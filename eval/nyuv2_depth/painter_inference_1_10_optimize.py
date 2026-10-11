#8.03_optimize
import warnings
import logging
# 屏蔽各种运行时警告
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

import sys
import os
import argparse
import math
from contextlib import nullcontext
import torch
import torch.fft
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import glob
import tqdm
from PIL import Image, ImageFile
import traceback

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# [优化6] 启用 cudnn.benchmark，固定输入尺寸下可缓存最快卷积算法
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    torch.set_float32_matmul_precision('high')

ImageFile.LOAD_TRUNCATED_IMAGES = True

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

import models_nyu as models_vpt_FA2_new

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
IMAGENET_MEAN_np = IMAGENET_MEAN.reshape(1, 1, 3)
IMAGENET_STD_np = IMAGENET_STD.reshape(1, 1, 3)


def sdp_ctx():
    """Compatible SDP kernel context; eval_drift 需 FP32 路径，避免 autocast 下 backward 全失败。"""
    if not torch.cuda.is_available():
        return nullcontext()
    try:
        return torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_mem_efficient=True, enable_math=True
        )
    except TypeError:
        try:
            return torch.backends.cuda.sdp_kernel(True, True, True)
        except Exception:
            return nullcontext()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--model', type=str, default='painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1')
    parser.add_argument('--prompt', type=str, default='study_room_0005b/rgb_00094')
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)

    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--lr_backbone', type=float, default=0.0)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.0,
        help='AdamW weight decay (L2 on weights). Default 0; PyTorch default would be 0.01.',
    )
    parser.add_argument('--steps', type=int, default=1)

    parser.add_argument('--vpt_type', type=str, default='shallow')
    parser.add_argument('--prompt_length', type=int, default=100)
    parser.add_argument('--vpt_target_layer', nargs='+', type=int, default=[0])

    parser.add_argument('--mode', type=str, choices=['inference', 'anchor', 'eval_drift', 'extract_drift'],
                        default='inference')
    parser.add_argument('--sample_list', type=str, default=None)
    parser.add_argument(
        '--inference_sample_list',
        type=str,
        default=None,
        help='Restrict inference to these image paths (newline-separated); used for fixed-seed 1/10 runs.',
    )
    parser.add_argument('--anchor_stats_path', type=str, default=None)
    parser.add_argument('--global_winner_path', type=str, default=None)
    parser.add_argument('--calibration_path', type=str, default='calibration_baseline.pt')
    parser.add_argument('--prompt_corruption', type=str, default='', help='提示图片的损坏类型，为空时使用 clean 版本')

    parser.add_argument('--num_bands', type=int, default=10, help='径向频带划分数量B')
    parser.add_argument('--lambda_gate', type=float, default=0.5, help='频域与梯度效用的融合权重')

    # =========================================================================
    # FC-VICT 新增参数
    # =========================================================================
    parser.add_argument('--use_fc_vict', action='store_true', help='是否开启 Frequency-Calibrated VICT')
    parser.add_argument('--use_fallback', action='store_true', help='是否开启策略二：Self-Selection Fallback')
    parser.add_argument('--r_L', type=float, default=0.03, help='低频遮罩半径 (0~1.0)')
    parser.add_argument('--r_H', type=float, default=0.15, help='高频遮罩半径 (0~1.0)')
    parser.add_argument('--rho', type=float, default=0.5, help='低频混合权重')
    parser.add_argument('--alpha_min', type=float, default=0.5, help='高频能量校准最小值')
    parser.add_argument('--alpha_max', type=float, default=1.5, help='高频能量校准最大值')
    parser.add_argument('--lambda_f', type=float, default=0.01, help='频域正则化 Loss 权重')
    parser.add_argument(
        '--mfm_fft_cpu',
        action='store_true',
        help='MFM 频域损失在 CPU 上 FFT（长跑更稳，避免 cuFFT 碎片导致变慢）。默认使用 GPU/cuFFT（与 pred 同设备）。',
    )

    return parser.parse_args()


def prepare_model(chkpt_dir, arch, args):
    if not hasattr(models_vpt_FA2_new, arch):
        raise AttributeError(f"Model {arch} not found")

    model = getattr(models_vpt_FA2_new, arch)(
        vpt_on=True, vpt_type=args.vpt_type, prompt_length=args.prompt_length,
        vpt_target_layer=args.vpt_target_layer, use_act_checkpoint=False
    )

    checkpoint = torch.load(chkpt_dir, map_location='cuda:0')
    model.load_state_dict(checkpoint['model'], strict=False)
    model.cuda()
    model.eval()

    return model

# =========================================================================
# FC-VICT 频域核心函数
# =========================================================================

# [优化4] 模块级缓存，避免重复计算 get_freq_masks
_freq_masks_cache = {}

def get_freq_masks(h, w, r_L=0.03, r_H=0.15, device='cpu'):
    cache_key = (h, w, r_L, r_H, str(device))
    if cache_key in _freq_masks_cache:
        return _freq_masks_cache[cache_key]

    c_h, c_w = h // 2, w // 2
    try:
        y, x = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing='ij')
    except TypeError:
        y, x = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32))
    y = y.to(device)
    x = x.to(device)

    rho = (1.0 / math.sqrt(2.0)) * torch.sqrt(((y - c_h) / (h / 2.0)) ** 2 + ((x - c_w) / (w / 2.0)) ** 2)

    M_L = (rho < r_L).float()
    M_H = (rho > r_H).float()
    M_M = ((rho >= r_L) & (rho <= r_H)).float()

    result = (M_L.unsqueeze(0).unsqueeze(0), M_M.unsqueeze(0).unsqueeze(0), M_H.unsqueeze(0).unsqueeze(0))
    _freq_masks_cache[cache_key] = result
    return result


def frequency_calibrate_prompt(prompt_img, test_img, args):
    device = 'cpu'
    x = torch.from_numpy(prompt_img).permute(2, 0, 1).unsqueeze(0).float()
    xt = torch.from_numpy(test_img).permute(2, 0, 1).unsqueeze(0).float()

    B, C, H, W = x.shape
    M_L, M_M, M_H = get_freq_masks(H, W, args.r_L, args.r_H, device)

    Fx = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
    Ft = torch.fft.fftshift(torch.fft.fft2(xt), dim=(-2, -1))

    Ax = torch.abs(Fx)
    Px = torch.angle(Fx)
    At = torch.abs(Ft)

    A_L = (1 - args.rho) * Ax + args.rho * At

    sum_M_H = M_H.sum() + 1e-8
    mean_At_H = (At * M_H).sum(dim=(-2, -1), keepdim=True) / sum_M_H
    mean_Ax_H = (Ax * M_H).sum(dim=(-2, -1), keepdim=True) / sum_M_H

    alpha_H = mean_At_H / (mean_Ax_H + 1e-8)
    alpha_H = torch.clamp(alpha_H, min=args.alpha_min, max=args.alpha_max)
    A_H = alpha_H * Ax

    Af = M_L * A_L + M_M * Ax + M_H * A_H

    Ff = torch.polar(Af, Px)
    Ff = torch.fft.ifftshift(Ff, dim=(-2, -1))
    xf = torch.fft.ifft2(Ff).real

    xf = torch.clamp(xf, 0, 1)
    return xf.squeeze(0).permute(1, 2, 0).numpy()


_mfm_gpu_fft_fallback_warned = False
_mfm_use_cpu_after_cufft_fail = False


def compute_mfm_frequency_loss(pred, target, M_L, M_H, fft_on_cpu=False):
    """FFT for MFM loss: prefer same device as ``pred`` (GPU cuFFT when CUDA).

    On cuFFT failures (e.g. ``CUFFT_INTERNAL_ERROR``), falls back to CPU FFT and stays on CPU
    for the rest of the process. Force CPU from the start with ``--mfm_fft_cpu``.
    """
    global _mfm_gpu_fft_fallback_warned, _mfm_use_cpu_after_cufft_fail

    def _spectral_dist(pred_in, tgt_in):
        fp = torch.fft.fftshift(torch.fft.fft2(pred_in, norm="forward"), dim=(-2, -1))
        ft = torch.fft.fftshift(torch.fft.fft2(tgt_in, norm="forward"), dim=(-2, -1))
        return torch.abs(fp - ft)

    use_cpu = fft_on_cpu or _mfm_use_cpu_after_cufft_fail

    if use_cpu:
        dist = _spectral_dist(pred.float().cpu(), target.float().cpu()).to(pred.device)
    elif not pred.is_cuda:
        dist = _spectral_dist(pred.float().contiguous(), target.float().contiguous())
    else:
        try:
            dist = _spectral_dist(pred.float().contiguous(), target.float().contiguous())
        except RuntimeError as e:
            err = str(e).lower()
            if "cufft" in err or "cuda" in err:
                _mfm_use_cpu_after_cufft_fail = True
                dist = _spectral_dist(pred.float().cpu(), target.float().cpu()).to(pred.device)
                if not _mfm_gpu_fft_fallback_warned:
                    print(
                        f"[warn] GPU cuFFT failed ({e}); using CPU FFT for MFM loss for remaining steps "
                        f"(or pass --mfm_fft_cpu to avoid trying GPU)."
                    )
                    _mfm_gpu_fft_fallback_warned = True
            else:
                raise

    mask = M_L if torch.rand(1).item() < 0.5 else M_H
    loss_freq = (dist * mask).sum() / (mask.expand_as(dist).sum() + 1e-8)
    return loss_freq

# =========================================================================
# FC-VICT 策略二：Self-Selection Fallback 相关函数
# =========================================================================
def calculate_cycle_loss(model, img_cat_np, tgt_cat_np, device):
    """计算 Cycle Reconstruction Loss 用于选择较好的 Prompt"""
    x = torch.tensor(img_cat_np).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)
    tgt = torch.tensor(tgt_cat_np).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)

    num_patches = model.patch_embed.num_patches
    h_img = 448
    mask_query = torch.zeros(num_patches).to(device)
    mask_query[num_patches // 2:] = 1
    mask_query = mask_query.unsqueeze(0)

    mask_support = torch.zeros(num_patches).to(device)
    mask_support[: num_patches // 2] = 1
    mask_support = mask_support.unsqueeze(0)

    valid = torch.ones_like(tgt)

    # [优化3] torch.no_grad() -> torch.inference_mode()
    with torch.inference_mode():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # 预测 test 图像 target
            _, y_tokens_1, _, _ = model(x, tgt, mask_query, valid)
            pred_query = model.unpatchify(y_tokens_1)[:, :, h_img:, :]

            # 使用预测的 target 做 cycle 重建 prompt 图像 target
            tgt_cycle = torch.cat([torch.zeros_like(pred_query), pred_query.detach()], dim=2)
            _, y_tokens_2, _, _ = model(x, tgt_cycle, mask_support, valid)
            recon_support = model.unpatchify(y_tokens_2)[:, :, :h_img, :]

            loss_cycle = F.smooth_l1_loss(recon_support, tgt[:, :, :h_img, :])
    return loss_cycle.item()


def get_best_prompt_image(model, img, prompt_img, prompt_img_f, prompt_tgt, device):
    """根据 Cycle Loss 动态选择使用原图还是频域校准后的图。

    Returns:
        (selected_prompt_img, chose_freq_calibrated): 若 chose_freq_calibrated 为 True 表示选中频域校准提示图。
    """
    img_cat_clean = (np.concatenate((prompt_img, img), axis=0) - IMAGENET_MEAN_np) / IMAGENET_STD_np
    tgt_cat = (np.concatenate((prompt_tgt, np.zeros((448, 448, 3))), axis=0) - IMAGENET_MEAN_np) / IMAGENET_STD_np

    # 评估时屏蔽 VPT layers
    orig_active_layers = getattr(model, 'active_vpt_layers', [])
    model.active_vpt_layers = []

    loss_clean = calculate_cycle_loss(model, img_cat_clean, tgt_cat, device)

    img_cat_f = (np.concatenate((prompt_img_f, img), axis=0) - IMAGENET_MEAN_np) / IMAGENET_STD_np
    loss_f = calculate_cycle_loss(model, img_cat_f, tgt_cat, device)

    # 恢复原 VPT layers
    model.active_vpt_layers = orig_active_layers

    if loss_f < loss_clean:
        return prompt_img_f, True
    else:
        return prompt_img, False


# =========================================================================

def get_spectral_features(x_tokens, h, w, num_bands=10):
    B_sz, L, D = x_tokens.shape
    device = x_tokens.device

    x_grid = x_tokens.transpose(1, 2).reshape(B_sz, D, h, w)
    # 在 CPU 上计算 FFT，避免 cuFFT 稳定性问题
    x_grid_cpu = x_grid.detach().cpu().float()
    fft_out_cpu = torch.fft.fft2(x_grid_cpu)

    energy_cpu = torch.abs(fft_out_cpu) ** 2
    energy_centered_cpu = torch.fft.fftshift(energy_cpu, dim=(-2, -1))
    energy_centered = energy_centered_cpu.to(device)

    c_h, c_w = (h - 1) / 2.0, (w - 1) / 2.0
    try:
        y, x = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing='ij')
    except TypeError:
        y, x = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32))

    y = y.to(device)
    x = x.to(device)
    rho = (1.0 / math.sqrt(2.0)) * torch.sqrt(((y - c_h) / (h / 2.0)) ** 2 + ((x - c_w) / (w / 2.0)) ** 2)

    e_l = torch.zeros(num_bands, device=device)
    for b in range(num_bands):
        r_prev = b / num_bands
        r_curr = (b + 1) / num_bands

        if b == num_bands - 1:
            mask = (rho >= r_prev) & (rho <= 1.0)
        else:
            mask = (rho >= r_prev) & (rho < r_curr)

        mask_float = mask.float()
        area = mask_float.sum() + 1e-6
        band_energy = (torch.log(1.0 + energy_centered) * mask_float).sum(dim=(-3, -2, -1))
        e_l[b] = band_energy / (D * area)

    e_hat = e_l / (e_l.sum() + 1e-6)
    return e_hat


def extract_query_tokens(layer_feat, num_patches):
    if layer_feat.dim() == 3:
        tokens = layer_feat[0]
    elif layer_feat.dim() == 2:
        tokens = layer_feat
    else:
        raise ValueError(f"Unsupported feature dim: {layer_feat.dim()}")
    return tokens[num_patches // 2:, :].unsqueeze(0)


def run_anchor_mode(args, model, device, prompt_img, prompt_tgt):
    print(f"--- Running Anchor Mode: Extracting Clean Data Anchor Stats ({args.num_bands} Bands) ---")
    with open(args.sample_list, 'r') as f:
        img_paths = [line.strip() for line in f.readlines() if line.strip()]

    num_patches = model.patch_embed.num_patches
    mask_query = torch.zeros(num_patches).to(device)
    mask_query[num_patches // 2:] = 1
    mask_query = mask_query.unsqueeze(0)

    anchor_layers = [0, 1, 2, 10, 11, 12, 21, 22, 23]
    S_accum = {l: [] for l in anchor_layers}
    count = 0
    skipped = 0

    model.eval()
    model.active_vpt_layers = []

    # [优化3] torch.no_grad() -> torch.inference_mode()
    with torch.inference_mode(), torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=True):
        for img_path in tqdm.tqdm(img_paths, desc="Processing Anchor Mode"):
            try:
                img = np.array(Image.open(img_path).convert("RGB").resize((448, 448))) / 255.

                # ----------------- Fallback 选择逻辑 -----------------
                if getattr(args, 'use_fc_vict', False):
                    prompt_img_f = frequency_calibrate_prompt(prompt_img, img, args)
                    if getattr(args, 'use_fallback', False):
                        current_prompt_img, chose_freq = get_best_prompt_image(
                            model, img, prompt_img, prompt_img_f, prompt_tgt, device
                        )
                        tqdm.tqdm.write(
                            f"[动态回退] {'选中频域校准提示图' if chose_freq else '保留原始提示图 (未回退至频域校准)'} | {img_path}"
                        )
                    else:
                        current_prompt_img = prompt_img_f
                else:
                    current_prompt_img = prompt_img
                # ---------------------------------------------------

                img_cat = (np.concatenate((current_prompt_img, img), axis=0) - IMAGENET_MEAN_np) / IMAGENET_STD_np
                tgt_cat = (np.concatenate((prompt_tgt, np.zeros((448, 448, 3))),
                                          axis=0) - IMAGENET_MEAN_np) / IMAGENET_STD_np

                x = torch.tensor(img_cat).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)
                tgt = torch.tensor(tgt_cat).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    _, _, _, inter_feats = model(x, tgt, mask_query, torch.ones_like(tgt))

                for l in anchor_layers:
                    query_feat = extract_query_tokens(inter_feats[l], num_patches)
                    e_hat = get_spectral_features(query_feat, 28, 28, num_bands=args.num_bands)
                    S_accum[l].append(e_hat.cpu())
                count += 1
            except Exception as e:
                skipped += 1
                if skipped <= 3:
                    print(f"[Anchor Warning] Skipping {img_path}: {e}")
                continue

    if count > 0:
        stats_accum = {}
        for l in anchor_layers:
            S_tensor = torch.stack(S_accum[l], dim=0)
            stats_accum[l] = {
                'mu': S_tensor.mean(dim=0),
                'std': S_tensor.std(dim=0, unbiased=False)
            }
        torch.save(stats_accum, args.anchor_stats_path)
        print(f"✅ Saved global clean spectral anchor stats to {args.anchor_stats_path}")
    else:
        raise RuntimeError(f"Failed to compute anchor stats; skipped {skipped}/{len(img_paths)} samples.")


def run_eval_drift_mode(args, model, device, prompt_img, prompt_tgt):
    print(f"--- Running {args.mode.upper()} Mode: Small Sample Voting (Freq Drift + Relative Gradient Utility) ---")
    anchor_stats = torch.load(args.anchor_stats_path, map_location='cpu')

    with open(args.sample_list, 'r') as f:
        img_paths = [line.strip() for line in f.readlines() if line.strip()]

    if not img_paths:
        raise RuntimeError(
            f"eval_drift: --sample_list is empty or has no valid lines: {args.sample_list!r}\n"
            "常见原因：损坏测试目录下没有 jpg/png；find 结果为空时仍 shuf 会导致空列表；或工作目录不对导致路径无效。"
        )

    num_patches = model.patch_embed.num_patches
    h_img = 448
    mask_query = torch.zeros(num_patches).to(device)
    mask_query[num_patches // 2:] = 1
    mask_query = mask_query.unsqueeze(0)

    mask_support = torch.zeros(num_patches).to(device)
    mask_support[: num_patches // 2] = 1
    mask_support = mask_support.unsqueeze(0)

    valid = torch.ones(1, 3, h_img * 2, h_img, device=device)

    layer_keys = [0, 1, 2, 10, 11, 12, 21, 22, 23]
    total_sample_scores = {l: 0.0 for l in layer_keys}
    valid_count = 0
    first_error_reported = False
    recorded_errors = []

    if getattr(args, 'use_fc_vict', False):
        M_L, _, M_H = get_freq_masks(h_img, h_img, args.r_L, args.r_H, device)

    print(f"Evaluating {len(img_paths)} samples to vote for the winner layer...")

    for img_path in tqdm.tqdm(img_paths, desc="Voting Winners"):
        try:
            img = np.array(Image.open(img_path).convert("RGB").resize((448, 448))) / 255.

            # ----------------- Fallback 选择逻辑 -----------------
            if getattr(args, 'use_fc_vict', False):
                prompt_img_f = frequency_calibrate_prompt(prompt_img, img, args)
                if getattr(args, 'use_fallback', False):
                    current_prompt_img, chose_freq = get_best_prompt_image(
                        model, img, prompt_img, prompt_img_f, prompt_tgt, device
                    )
                    tqdm.tqdm.write(
                        f"[动态回退] {'选中频域校准提示图' if chose_freq else '保留原始提示图 (未回退至频域校准)'} | {img_path}"
                    )
                else:
                    current_prompt_img = prompt_img_f
            else:
                current_prompt_img = prompt_img
            # ---------------------------------------------------

            img_cat = (np.concatenate((current_prompt_img, img), axis=0) - IMAGENET_MEAN_np) / IMAGENET_STD_np
            tgt_cat = (np.concatenate((prompt_tgt, np.zeros((448, 448, 3))), axis=0) - IMAGENET_MEAN_np) / IMAGENET_STD_np

            x = torch.tensor(img_cat).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)
            tgt = torch.tensor(tgt_cat).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)

            x.requires_grad_(False)

            model.eval()
            model.active_vpt_layers = []

            # [优化3] torch.no_grad() -> torch.inference_mode()
            with torch.inference_mode():
                _, _, _, inter_feats = model(x, tgt, mask_query, torch.ones_like(tgt))

            d_freq = {}
            for l in layer_keys:
                query_feat = extract_query_tokens(inter_feats[l], num_patches)
                e_hat_test = get_spectral_features(query_feat, 28, 28, num_bands=args.num_bands)

                mu_train = anchor_stats[l]['mu'].to(device)
                sigma_train = anchor_stats[l]['std'].to(device)

                d_l_freq = torch.mean(torch.abs((e_hat_test - mu_train) / (sigma_train + 1e-6))).item()
                d_freq[l] = d_l_freq

            model.train()
            model.active_vpt_layers = layer_keys

            for param in model.parameters():
                param.requires_grad = False

            prompt_params = []
            if hasattr(model, 'prompt_embeddings'):
                for idx, layer_idx in enumerate(sorted(args.vpt_target_layer)):
                    if layer_idx in layer_keys:
                        model.prompt_embeddings[idx].requires_grad = True
                        prompt_params.append(model.prompt_embeddings[idx])

            model.zero_grad(set_to_none=True)

            with sdp_ctx():
                # [优化] BF16 混合精度：前向用 BF16，loss/反向用 FP32
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    _, y_tokens_1, _, _ = model(x, tgt, mask_query, valid)
                    pred_query = model.unpatchify(y_tokens_1)[:, :, h_img:, :]

                    tgt_cycle = torch.cat([torch.zeros_like(pred_query), pred_query.detach()], dim=2)
                    _, y_tokens_2, _, _ = model(x, tgt_cycle, mask_support, valid)
                    recon_support = model.unpatchify(y_tokens_2)[:, :, :h_img, :]

                loss_cycle = F.smooth_l1_loss(recon_support.float(), tgt[:, :, :h_img, :].float())
                loss = loss_cycle

                if getattr(args, 'use_fc_vict', False):
                    loss_freq = compute_mfm_frequency_loss(
                        recon_support, tgt[:, :, :h_img, :], M_L, M_H, fft_on_cpu=args.mfm_fft_cpu
                    )
                    loss = loss + args.lambda_f * loss_freq

                loss.backward()

                d_util = {}
                raw_grads = {}
                if hasattr(model, 'prompt_embeddings'):
                    for idx, layer_idx in enumerate(sorted(args.vpt_target_layer)):
                        if layer_idx in layer_keys:
                            grad = model.prompt_embeddings[idx].grad
                            if grad is not None:
                                raw_grads[layer_idx] = torch.norm(grad, p=2).item()
                            else:
                                raw_grads[layer_idx] = 0.0

                mean_grad = sum(raw_grads.values()) / (len(raw_grads) + 1e-8)
                for layer_idx in layer_keys:
                    d_util[layer_idx] = raw_grads.get(layer_idx, 0.0) / (mean_grad + 1e-8)

            model.zero_grad(set_to_none=True)

            d_freq_log = {k: math.log(v + 1e-6) for k, v in d_freq.items()}
            d_util_log = {k: math.log(v + 1e-8) for k, v in d_util.items()}

            def shift_to_positive(scores_dict):
                min_val = min(scores_dict.values())
                return {k: (v - min_val + 1.0) for k, v in scores_dict.items()}

            freq_shifted = shift_to_positive(d_freq_log)
            util_shifted = shift_to_positive(d_util_log)

            def get_sum_norm(scores_dict):
                total = sum(scores_dict.values()) + 1e-6
                return {k: v / total for k, v in scores_dict.items()}

            freq_scores = get_sum_norm(freq_shifted)
            util_scores = get_sum_norm(util_shifted)

            S_dict = {}
            for l in layer_keys:
                S_dict[l] = args.lambda_gate * freq_scores[l] + (1.0 - args.lambda_gate) * util_scores[l]

            tau = 1.0
            s_tensor = torch.tensor([S_dict[l] for l in layer_keys], dtype=torch.float32, device=device)
            w_tensor = torch.nn.functional.softmax(s_tensor / tau, dim=0)

            for i, l in enumerate(layer_keys):
                total_sample_scores[l] += w_tensor[i].item()

            valid_count += 1
        except Exception as e:
            print(f"[eval_drift] Skip image due to error: {e}")
            if len(recorded_errors) < 8:
                recorded_errors.append((img_path, f"{type(e).__name__}: {e}"))
            if not first_error_reported:
                traceback.print_exc()
                first_error_reported = True
            continue

    if valid_count == 0:
        detail = "; ".join(f"{p} -> {msg}" for p, msg in recorded_errors[:5])
        raise RuntimeError(
            "No valid images processed in eval_drift. "
            f"sample_list={args.sample_list!r}, listed_paths={len(img_paths)}, "
            f"first_errors=[{detail}]"
        )

    print("\n--- 预选拔平均融合得分 (Relative Gradient Utility Based Fusion Scores) ---")
    for l in layer_keys:
        avg_score = total_sample_scores[l] / valid_count
        print(f"  Layer {l:2d}: {avg_score:.4f}")

    group1_score = total_sample_scores[0] + total_sample_scores[1] + total_sample_scores[2]
    group2_score = total_sample_scores[10] + total_sample_scores[11] + total_sample_scores[12]
    group3_score = total_sample_scores[21] + total_sample_scores[22] + total_sample_scores[23]

    print("\n--- 预选拔分组汇总得分 ---")
    print(f"  Group 1 (Layer 0,1,2) Score: {group1_score:.4f}")
    print(f"  Group 2 (Layer 10,11,12) Score: {group2_score:.4f}")
    print(f"  Group 3 (Layer 21,22,23) Score: {group3_score:.4f}")

    final_scores = {'group1': group1_score, 'group2': group2_score, 'group3': group3_score}
    global_winner_group = max(final_scores, key=final_scores.get)

    print(f"\n=> 🎯 Final Voted Winner Group: {global_winner_group}")

    with open(args.global_winner_path, 'w') as f:
        f.write(global_winner_group)

def run_one_image_mt3(img, tgt, size, model, out_path, device, args, global_winner_layer,
                      optimizer=None, prompt_embeddings_backup=None):
    group_mapping = {
        'group1': [0, 1, 2],
        'group2': [10, 11, 12],
        'group3': [21, 22, 23]
    }
    # 处理 "all" 表示更新全部层
    if global_winner_layer == 'all':
        active_target_layers = [0, 1, 2, 10, 11, 12, 21, 22, 23]
    else:
        active_target_layers = group_mapping[global_winner_layer]

    model.eval()
    model.active_vpt_layers = active_target_layers

    x = torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)
    tgt = torch.tensor(tgt).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)

    x.requires_grad_(False)

    # [优化8] 使用预存的 prompt_embeddings 备份，避免每张图重新 clone
    if prompt_embeddings_backup is None and hasattr(model, 'prompt_embeddings'):
        prompt_embeddings_backup = [p.data.clone() for p in model.prompt_embeddings]

    modules_to_optimize = [model.blocks[-1], model.norm]
    if hasattr(model, 'decoder_embed'): modules_to_optimize.append(model.decoder_embed)
    if hasattr(model, 'decoder_pred'): modules_to_optimize.append(model.decoder_pred)
    # [优化2] state_dict() 本身已是浅拷贝，无需 deepcopy
    backup_module_states = [m.state_dict() for m in modules_to_optimize]

    num_patches = model.patch_embed.num_patches
    h_img = 448
    mask_support = torch.zeros(num_patches).to(device)
    mask_support[: num_patches // 2] = 1
    mask_support = mask_support.unsqueeze(0)

    mask_query = torch.zeros(num_patches).to(device)
    mask_query[num_patches // 2:] = 1
    mask_query = mask_query.unsqueeze(0)
    valid = torch.ones_like(tgt)

    for param in model.parameters(): param.requires_grad = False
    prompt_params, backbone_params = [], []

    if hasattr(model, 'prompt_embeddings'):
        for idx, layer_idx in enumerate(sorted(args.vpt_target_layer)):
            if layer_idx in active_target_layers:
                model.prompt_embeddings[idx].requires_grad = True
                prompt_params.append(model.prompt_embeddings[idx])
            else:
                model.prompt_embeddings[idx].requires_grad = False

    if args.lr_backbone > 0:
        for m in modules_to_optimize:
            for param in m.parameters():
                param.requires_grad = True
                backbone_params.append(param)

    # [优化5] 复用外部传入的优化器，或首次创建
    if optimizer is None:
        optimizer = optim.AdamW(
            [
                {'params': prompt_params, 'lr': args.lr},
                {'params': backbone_params, 'lr': args.lr_backbone},
            ],
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
    else:
        # 更新参数组（prompt_params 可能因 active_target_layers 不同而变化）
        optimizer.param_groups[0]['params'] = prompt_params
        optimizer.param_groups[1]['params'] = backbone_params

    if getattr(args, 'use_fc_vict', False):
        M_L, _, M_H = get_freq_masks(h_img, h_img, args.r_L, args.r_H, device)

    model.train()
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=True):
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)

            # [优化] BF16 混合精度：前向用 BF16，loss/反向用 FP32
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                _, y_tokens_1, _, _ = model(x, tgt, mask_query, valid)
                pred_query = model.unpatchify(y_tokens_1)[:, :, h_img:, :]

                tgt_cycle = torch.cat([torch.zeros_like(pred_query), pred_query.detach()], dim=2)
                _, y_tokens_2, _, _ = model(x, tgt_cycle, mask_support, valid)
                recon_support = model.unpatchify(y_tokens_2)[:, :, :h_img, :]

            loss_cycle = F.smooth_l1_loss(recon_support.float(), tgt[:, :, :h_img, :].float())
            loss = loss_cycle

            if getattr(args, 'use_fc_vict', False):
                loss_freq = compute_mfm_frequency_loss(
                    recon_support, tgt[:, :, :h_img, :], M_L, M_H, fft_on_cpu=args.mfm_fft_cpu
                )
                loss = loss + args.lambda_f * loss_freq

            loss.backward()
            optimizer.step()

    model.eval()
    # [优化3] torch.no_grad() -> torch.inference_mode()
    # [优化] BF16 autocast 加速最终推理（无反向传播）
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16), torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=True):
        _, y_tokens_final, _, _ = model(x, tgt, mask_query, valid)

        image_1 = model.unpatchify(y_tokens_final)[:, :, h_img:, :][0].permute(1, 2, 0).float().cpu().numpy()

        output = torch.from_numpy(image_1).permute(2, 0, 1).unsqueeze(0)
        output = torch.clip((output.permute(0, 2, 3, 1) * IMAGENET_STD + IMAGENET_MEAN) * 10000, 0, 10000)
        output = F.interpolate(output.permute(0, 3, 1, 2), size=[size[1], size[0]], mode='bilinear').squeeze(0).permute(
            1, 2, 0)
        output = output.mean(-1).int().numpy().astype(np.uint16)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        Image.fromarray(output).save(out_path)

    # [优化8] 恢复 prompt_embeddings
    if hasattr(model, 'prompt_embeddings') and prompt_embeddings_backup is not None:
        with torch.inference_mode():
            for p, backup_p in zip(model.prompt_embeddings, prompt_embeddings_backup):
                p.copy_(backup_p)
    for m, state in zip(modules_to_optimize, backup_module_states):
        m.load_state_dict(state)

    for param in model.parameters(): param.requires_grad = False
    # [优化5] 清空优化器状态而非销毁
    optimizer.zero_grad(set_to_none=True)
    optimizer.state.clear()

    return global_winner_layer, optimizer, prompt_embeddings_backup


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = prepare_model(args.ckpt_path, args.model, args)

    if args.prompt_corruption:
        prompt_rgb_path = f"datasets/nyu_depth_v2/corrupted_prompts/{args.prompt_corruption}/5/{args.prompt}.jpg"
    else:
        prompt_rgb_path = f"datasets/nyu_depth_v2/sync/{args.prompt}.jpg"
    prompt_depth_path = f"datasets/nyu_depth_v2/sync/{args.prompt.replace('rgb', 'sync_depth')}.png"

    prompt_img = np.array(Image.open(prompt_rgb_path).convert("RGB").resize((448, 448))) / 255.
    if os.path.exists(prompt_depth_path):
        prompt_tgt = np.array(Image.open(prompt_depth_path).resize((448, 448))) / 10000.
        if len(prompt_tgt.shape) == 2:
            prompt_tgt = np.repeat(prompt_tgt[:, :, np.newaxis], 3, axis=2)
        elif prompt_tgt.shape[2] == 1:
            prompt_tgt = np.repeat(prompt_tgt, 3, axis=2)
    else:
        prompt_tgt = np.zeros((448, 448, 3))

    if args.mode == 'anchor':
        run_anchor_mode(args, model, device, prompt_img, prompt_tgt)
        return
    elif args.mode in ['eval_drift', 'extract_drift']:
        run_eval_drift_mode(args, model, device, prompt_img, prompt_tgt)
        return

    with open(args.global_winner_path, 'r') as f:
        global_winner_layer = f.read().strip()

    print(f"\n[{args.mode.upper()}] Voted Winner Group: {global_winner_layer}")

    if args.inference_sample_list:
        with open(args.inference_sample_list, 'r') as f:
            img_list = [line.strip() for line in f.readlines() if line.strip()]
        print(f"[Inference] Using {len(img_list)} images from --inference_sample_list")
    else:
        img_list = glob.glob(os.path.join(args.input_dir, "**/*.jpg"), recursive=True) + glob.glob(
            os.path.join(args.input_dir, "**/*.png"), recursive=True)

    # [优化5] 预分配优化器和 prompt 备份，在循环中复用
    shared_optimizer = None
    shared_prompt_backup = None

    for img_path in tqdm.tqdm(img_list, desc="Running Inference Loop"):
        try:
            img_raw = Image.open(img_path).convert("RGB")
            orig_size = img_raw.size
            img = np.array(img_raw.resize((448, 448))) / 255.

            # ----------------- Fallback 选择逻辑 -----------------
            if getattr(args, 'use_fc_vict', False):
                prompt_img_f = frequency_calibrate_prompt(prompt_img, img, args)
                if getattr(args, 'use_fallback', False):
                    current_prompt_img, chose_freq = get_best_prompt_image(
                        model, img, prompt_img, prompt_img_f, prompt_tgt, device
                    )
                    tqdm.tqdm.write(
                        f"[动态回退] {'选中频域校准提示图' if chose_freq else '保留原始提示图 (未回退至频域校准)'} | {img_path}"
                    )
                else:
                    current_prompt_img = prompt_img_f
            else:
                current_prompt_img = prompt_img
            # ---------------------------------------------------

            img_cat = (np.concatenate((current_prompt_img, img), axis=0) - IMAGENET_MEAN_np) / IMAGENET_STD_np
            tgt_cat = (np.concatenate((prompt_tgt, np.zeros((448, 448, 3))),
                                      axis=0) - IMAGENET_MEAN_np) / IMAGENET_STD_np

            out_path = os.path.join(args.save_dir, os.path.basename(os.path.dirname(img_path)),
                                    os.path.basename(img_path).replace(".jpg", ".png"))
            # [优化5] 传入并接收复用的优化器和 prompt 备份
            _, shared_optimizer, shared_prompt_backup = run_one_image_mt3(
                img_cat, tgt_cat, orig_size, model, out_path, device, args, global_winner_layer,
                optimizer=shared_optimizer, prompt_embeddings_backup=shared_prompt_backup
            )
        except Exception as e:
            print(f"[Error processing {img_path}]: {e}")
            continue
        # [优化1] 移除逐图像的 gc.collect() + torch.cuda.empty_cache()

    if args.mode == 'inference':
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, "winning_groups.txt"), "w") as f:
            f.write(f"Global Winner Layer: {global_winner_layer}")


if __name__ == '__main__':
    main()
