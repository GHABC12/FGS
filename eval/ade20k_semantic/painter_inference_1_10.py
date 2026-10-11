# 8.03对齐版本 (修复 ADE20K TTA 崩溃问题，引入调色板离散化吸附 + FC-VICT 频域对齐)
import warnings
import logging
# 屏蔽各种运行时警告
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

import sys
import os
import argparse
import copy
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
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    torch.set_float32_matmul_precision('high')

ImageFile.LOAD_TRUNCATED_IMAGES = True

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

import models_nyu as models_vpt_FA2_new

# 引入官方 ADE20K 调色板
try:
    from data.ade20k.gen_color_ade20k_sem import define_colors_per_location_mean_sep
    ADE20K_PALETTE = define_colors_per_location_mean_sep()
except ImportError:
    print("Warning: Could not import ADE20K palette. Make sure data/ade20k/gen_color_ade20k_sem.py exists.")
    ADE20K_PALETTE = []

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
IMAGENET_MEAN_np = IMAGENET_MEAN.reshape(1, 1, 3)
IMAGENET_STD_np = IMAGENET_STD.reshape(1, 1, 3)


def sdp_ctx():
    """Return a compatible SDP kernel context across torch versions (aligned with eval/lol)."""
    try:
        return torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=True
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
    parser.add_argument('--prompt', type=str, default='ADE_train_00014165')
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)

    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--lr_backbone', type=float, default=0.0)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--steps', type=int, default=1)

    parser.add_argument('--vpt_type', type=str, default='shallow')
    parser.add_argument('--prompt_length', type=int, default=100)
    parser.add_argument('--vpt_target_layer', nargs='+', type=int, default=[0])

    parser.add_argument('--mode', type=str, choices=['inference', 'anchor', 'eval_drift', 'extract_drift'],
                        default='inference')
    parser.add_argument('--sample_list', type=str, default=None)
    parser.add_argument('--anchor_stats_path', type=str, default=None)
    parser.add_argument('--global_winner_path', type=str, default=None)
    parser.add_argument('--calibration_path', type=str, default='calibration_baseline.pt')

    # 联合门控机制超参数
    parser.add_argument('--num_bands', type=int, default=10, help='径向频带划分数量B')
    parser.add_argument('--lambda_gate', type=float, default=0.5, help='频域与梯度效用的融合权重')

    parser.add_argument('--num_images', type=int, default=None, help='限制推理图片数量')
    parser.add_argument('--seed', type=int, default=42, help='随机种子，保证采样可复现')

    parser.add_argument('--prompt_mode', type=str, choices=['zero-shot', 'one-shot'],
                        default='zero-shot',
                        help='zero-shot: 使用干净 prompt 图; one-shot: 使用与当前 corruption 对应的 severity 损坏 prompt 图')

    # =========================================================================
    # FC-VICT 新增参数
    # =========================================================================
    parser.add_argument('--use_fc_vict', action='store_true', help='是否开启 Frequency-Calibrated VICT')
    parser.add_argument('--r_L', type=float, default=0.03, help='低频遮罩半径 (0~1.0)')
    parser.add_argument('--r_H', type=float, default=0.15, help='高频遮罩半径 (0~1.0)')
    parser.add_argument('--rho', type=float, default=0.5, help='低频混合权重')
    parser.add_argument('--alpha_min', type=float, default=0.5, help='高频能量校准最小值')
    parser.add_argument('--alpha_max', type=float, default=1.5, help='高频能量校准最大值')
    parser.add_argument('--lambda_f', type=float, default=0.005, help='频域正则化 Loss 权重')

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
def get_freq_masks(h, w, r_L=0.03, r_H=0.15, device='cpu'):
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
    
    return M_L.unsqueeze(0).unsqueeze(0), M_M.unsqueeze(0).unsqueeze(0), M_H.unsqueeze(0).unsqueeze(0)


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


def compute_mfm_frequency_loss(pred, target, M_L, M_H):
    """频域正则：与 eval/lol 一致，FFT 在 CPU 上算以避免 cuFFT/CUFFT_INTERNAL_ERROR。"""
    pred_cpu = pred.float().cpu()
    target_cpu = target.float().cpu()

    F_pred = torch.fft.fftshift(torch.fft.fft2(pred_cpu, norm="forward"), dim=(-2, -1))
    F_target = torch.fft.fftshift(torch.fft.fft2(target_cpu, norm="forward"), dim=(-2, -1))

    dist = torch.abs(F_pred - F_target).to(pred.device)
    mask = M_L if torch.rand(1).item() < 0.5 else M_H

    loss_freq = (dist * mask).sum() / (mask.expand_as(dist).sum() + 1e-8)
    return loss_freq

# =========================================================================

def get_spectral_features(x_tokens, h, w, num_bands=10):
    B_sz, L, D = x_tokens.shape
    device = x_tokens.device

    x_grid = x_tokens.transpose(1, 2).reshape(B_sz, D, h, w)
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
    
    with torch.no_grad(), sdp_ctx():
        for img_path in tqdm.tqdm(img_paths, desc="Processing Anchor Mode"):
            try:
                img = np.array(Image.open(img_path).convert("RGB").resize((448, 448))) / 255.
                
                if getattr(args, 'use_fc_vict', False):
                    current_prompt_img = frequency_calibrate_prompt(prompt_img, img, args)
                else:
                    current_prompt_img = prompt_img
                    
                img_cat = (np.concatenate((current_prompt_img, img), axis=0) - IMAGENET_MEAN_np) / IMAGENET_STD_np
                tgt_cat = (np.concatenate((prompt_tgt, np.zeros((448, 448, 3))),
                                          axis=0) - IMAGENET_MEAN_np) / IMAGENET_STD_np

                x = torch.tensor(img_cat).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)
                tgt = torch.tensor(tgt_cat).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)

                _, _, _, inter_feats = model(x, tgt, mask_query, torch.ones_like(tgt))

                for l in anchor_layers:
                    query_feat = extract_query_tokens(inter_feats[l], num_patches)
                    e_hat = get_spectral_features(query_feat, 28, 28, num_bands=args.num_bands)
                    S_accum[l].append(e_hat.cpu())
                count += 1
            except Exception as e:
                skipped += 1
                continue

    if count > 0:
        stats_accum = {}
        for l in anchor_layers:
            S_tensor = torch.stack(S_accum[l], dim=0)
            stats_accum[l] = {'mu': S_tensor.mean(dim=0), 'std': S_tensor.std(dim=0, unbiased=False)}
        torch.save(stats_accum, args.anchor_stats_path)
        print(f"✅ Saved global clean spectral anchor stats to {args.anchor_stats_path}")
    else:
        raise RuntimeError(f"Failed to compute anchor stats; skipped {skipped}/{len(img_paths)} samples.")


def run_eval_drift_mode(args, model, device, prompt_img, prompt_tgt):
    print(f"--- Running {args.mode.upper()} Mode: Small Sample Voting (Freq Drift + Relative Gradient Utility) ---")
    anchor_stats = torch.load(args.anchor_stats_path, map_location='cpu')

    with open(args.sample_list, 'r') as f:
        img_paths = [line.strip() for line in f.readlines() if line.strip()]

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
    
    if getattr(args, 'use_fc_vict', False):
        M_L, _, M_H = get_freq_masks(h_img, h_img, args.r_L, args.r_H, device)

    print(f"Evaluating {len(img_paths)} samples to vote for the winner layer...")

    for img_path in tqdm.tqdm(img_paths, desc="Voting Winners"):
        try:
            img = np.array(Image.open(img_path).convert("RGB").resize((448, 448))) / 255.
            
            if getattr(args, 'use_fc_vict', False):
                current_prompt_img = frequency_calibrate_prompt(prompt_img, img, args)
            else:
                current_prompt_img = prompt_img
                
            img_cat = (np.concatenate((current_prompt_img, img), axis=0) - IMAGENET_MEAN_np) / IMAGENET_STD_np
            tgt_cat = (np.concatenate((prompt_tgt, np.zeros((448, 448, 3))),
                                      axis=0) - IMAGENET_MEAN_np) / IMAGENET_STD_np

            x = torch.tensor(img_cat).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)
            tgt = torch.tensor(tgt_cat).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)
            
            x.requires_grad_(False)

            model.eval()
            model.active_vpt_layers = []
            with torch.no_grad():
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
            for param in model.parameters(): param.requires_grad = False
            
            prompt_params = []
            if hasattr(model, 'prompt_embeddings'):
                for idx, layer_idx in enumerate(sorted(args.vpt_target_layer)):
                    if layer_idx in layer_keys:
                        model.prompt_embeddings[idx].requires_grad = True
                        prompt_params.append(model.prompt_embeddings[idx])

            model.zero_grad(set_to_none=True)

            with sdp_ctx():
                _, y_tokens_1, _, _ = model(x, tgt, mask_query, valid)
                pred_query = model.unpatchify(y_tokens_1)[:, :, h_img:, :]

                tgt_cycle = torch.cat([torch.zeros_like(pred_query), pred_query.detach()], dim=2)
                _, y_tokens_2, _, _ = model(x, tgt_cycle, mask_support, valid)
                recon_support = model.unpatchify(y_tokens_2)[:, :, :h_img, :]

                loss_cycle = F.smooth_l1_loss(recon_support, tgt[:, :, :h_img, :])
                loss = loss_cycle

                if getattr(args, 'use_fc_vict', False):
                    loss_freq = compute_mfm_frequency_loss(recon_support, tgt[:, :, :h_img, :], M_L, M_H)
                    loss = loss + args.lambda_f * loss_freq

                loss.backward()

                d_util = {}
                raw_grads = {}

                for idx, layer_idx in enumerate(sorted(args.vpt_target_layer)):
                    if layer_idx in layer_keys:
                        grad = model.prompt_embeddings[idx].grad
                        if grad is not None:
                            raw_grads[layer_idx] = torch.norm(grad, p=2).item()
                        else:
                            raw_grads[layer_idx] = 0.0

                mean_grad = sum(raw_grads.values()) / (len(raw_grads) + 1e-8)

                for layer_idx in layer_keys:
                    d_util[layer_idx] = raw_grads.get(layer_idx, 0.0) / mean_grad

            model.zero_grad(set_to_none=True)

            d_freq_log = {k: math.log(v + 1e-6) for k, v in d_freq.items()}
            d_util_log = {k: math.log(v + 1e-8) for k, v in d_util.items()}

            def shift_to_positive(scores_dict):
                min_val = min(scores_dict.values())
                return {k: (v - min_val + 1.0) for k, v in scores_dict.items()}

            def get_sum_norm(scores_dict):
                total = sum(scores_dict.values()) + 1e-6
                return {k: v / total for k, v in scores_dict.items()}

            freq_scores = get_sum_norm(shift_to_positive(d_freq_log))
            util_scores = get_sum_norm(shift_to_positive(d_util_log))

            S_dict = {l: args.lambda_gate * freq_scores[l] + (1.0 - args.lambda_gate) * util_scores[l] for l in layer_keys}

            tau = 1.0
            s_tensor = torch.tensor([S_dict[l] for l in layer_keys], dtype=torch.float32, device=device)
            w_tensor = torch.nn.functional.softmax(s_tensor / tau, dim=0)

            for i, l in enumerate(layer_keys): total_sample_scores[l] += w_tensor[i].item()
            valid_count += 1
        except Exception as e:
            continue

    if valid_count == 0:
        raise RuntimeError("No valid images processed in eval_drift.")

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

    scores_path = args.global_winner_path.replace('.txt', '_scores.txt')
    with open(scores_path, 'w') as f:
        f.write(f"group1_score={group1_score:.6f}\n")
        f.write(f"group2_score={group2_score:.6f}\n")
        f.write(f"group3_score={group3_score:.6f}\n")
        f.write(f"winner={global_winner_group}\n")


def run_one_image_mt3(img, tgt, size, model, out_path, device, args, global_winner_layer):
    group_mapping = {'group1': [0, 1, 2], 'group2': [10, 11, 12], 'group3': [21, 22, 23]}
    active_target_layers = group_mapping[global_winner_layer]

    model.eval()
    model.active_vpt_layers = active_target_layers

    x = torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)
    tgt = torch.tensor(tgt).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)
    
    x.requires_grad_(False)

    backup_states = {}
    if hasattr(model, 'prompt_embeddings'):
        backup_states['prompt_embeddings'] = [p.data.clone() for p in model.prompt_embeddings]

    modules_to_optimize = [model.blocks[-1], model.norm]
    if hasattr(model, 'decoder_embed'): modules_to_optimize.append(model.decoder_embed)
    if hasattr(model, 'decoder_pred'): modules_to_optimize.append(model.decoder_pred)
    backup_module_states = [copy.deepcopy(m.state_dict()) for m in modules_to_optimize]

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

    optimizer = optim.AdamW([
        {'params': prompt_params, 'lr': args.lr},
        {'params': backbone_params, 'lr': args.lr_backbone}
    ], betas=(args.beta1, args.beta2))
    
    if getattr(args, 'use_fc_vict', False):
        M_L, _, M_H = get_freq_masks(h_img, h_img, args.r_L, args.r_H, device)

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # =============== 核心修改区域：准备调色板和常数张量 ===============
    if len(ADE20K_PALETTE) > 0:
        palette_tensor = torch.tensor(ADE20K_PALETTE, dtype=torch.float32, device=device)
    else:
        palette_tensor = None
        
    imagenet_mean_t = torch.tensor(IMAGENET_MEAN_np, dtype=torch.float32, device=device)
    imagenet_std_t = torch.tensor(IMAGENET_STD_np, dtype=torch.float32, device=device)
    # =================================================================

    model.train()
    with sdp_ctx():
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                _, y_tokens_1, _, _ = model(x, tgt, mask_query, valid)
                pred_query = model.unpatchify(y_tokens_1)[:, :, h_img:, :]
                
                # =================== [新增逻辑：调色板离散化吸附] ===================
                if palette_tensor is not None:
                    with torch.no_grad():
                        # (1, 3, h, w) -> (1, h, w, 3) 
                        pred_rgb = pred_query.detach().float().permute(0, 2, 3, 1)
                        # 逆归一化到 0-255 像素域
                        pred_rgb = (pred_rgb * imagenet_std_t + imagenet_mean_t) * 255.0
                        
                        B, H, W, C = pred_rgb.shape
                        K = palette_tensor.shape[0]
                        
                        # 广播计算 L1 距离 (B, H, W, 1, 3) - (1, 1, 1, K, 3)
                        dist = torch.abs(pred_rgb.unsqueeze(3) - palette_tensor.view(1, 1, 1, K, 3))
                        dist = torch.sum(dist, dim=-1)  # (B, H, W, K)
                        
                        # 取距离最近的调色板语义色彩 (Palette matching)
                        pred_idx = dist.argmin(dim=-1)  # (B, H, W)
                        clean_pred_rgb = palette_tensor[pred_idx]  # (B, H, W, 3)
                        
                        # 重新归一化回 ImageNet 空间，作为下一次推理纯净合法的 Target
                        clean_pred = (clean_pred_rgb / 255.0 - imagenet_mean_t) / imagenet_std_t
                        clean_pred = clean_pred.permute(0, 3, 1, 2)  # (1, 3, H, W)
                        
                        # 替换原始模糊的连续回归输出
                        pred_query_cycle = clean_pred
                else:
                    pred_query_cycle = pred_query.detach()
                # ====================================================================

                # 将离散化清洗后的预测作为循环的 Prompt Target
                tgt_cycle = torch.cat([torch.zeros_like(pred_query_cycle), pred_query_cycle], dim=2)
                
                _, y_tokens_2, _, _ = model(x, tgt_cycle, mask_support, valid)
                recon_support = model.unpatchify(y_tokens_2)[:, :, :h_img, :]
                
                loss_cycle = F.smooth_l1_loss(recon_support, tgt[:, :, :h_img, :])
                loss = loss_cycle
                
                if getattr(args, 'use_fc_vict', False):
                    loss_freq = compute_mfm_frequency_loss(recon_support, tgt[:, :, :h_img, :], M_L, M_H)
                    loss = loss + args.lambda_f * loss_freq
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    model.eval()
    with torch.no_grad(), sdp_ctx():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            _, y_tokens_final, _, _ = model(x, tgt, mask_query, valid)

        # --- ADE20k 后处理：恢复为彩色图像格式 ---
        output = model.unpatchify(y_tokens_final)[:, :, h_img:, :].detach().cpu()
        output = output[0]  # (3, H, W)
        output = torch.clip((output.permute(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN) * 255, 0, 255)  # 恢复到0-255

        # 尺寸插值还原
        output = F.interpolate(
            output[None, ...].permute(0, 3, 1, 2),
            size=[size[1], size[0]],
            mode='nearest',
        ).permute(0, 2, 3, 1)[0]
        output = output.int()

        # 存为 uint8 的彩色 PNG 图片
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        Image.fromarray(output.numpy().astype(np.uint8)).save(out_path)

    if hasattr(model, 'prompt_embeddings'):
        for p, backup_p in zip(model.prompt_embeddings, backup_states['prompt_embeddings']):
            with torch.no_grad(): p.copy_(backup_p)
    for m, state in zip(modules_to_optimize, backup_module_states): m.load_state_dict(state)
    for param in model.parameters(): param.requires_grad = False
    optimizer.zero_grad(set_to_none=True)
    del optimizer

    return global_winner_layer


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = prepare_model(args.ckpt_path, args.model, args)

    # --- ADE20k Prompt 加载逻辑 ---
    prompt_depth_path = f"datasets/ade20k/annotations_with_color/training/{args.prompt}.png"

    if args.prompt_mode == 'one-shot':
        parts = os.path.normpath(args.input_dir).split(os.sep)
        try:
            idx = parts.index('ade20k-c')
            corruption_name = parts[idx + 1]
            severity = parts[idx + 2]
        except (ValueError, IndexError):
            raise ValueError(
                f"--prompt_mode one-shot 要求 --input_dir 形如 datasets/ade20k-c/<corruption>/<severity>/...，"
                f"实际收到: {args.input_dir}"
            )
        prompt_rgb_path = (
            f"datasets/ade20k-c/{corruption_name}/{severity}/"
            f"ade/ADEChallengeData2016/images/training/{args.prompt}.jpg"
        )
        print(f"[Prompt] one-shot mode -> {prompt_rgb_path}")
    else:
        prompt_rgb_path = f"datasets/ade20k/images/training/{args.prompt}.jpg"
        print(f"[Prompt] zero-shot mode -> {prompt_rgb_path}")

    prompt_img = np.array(Image.open(prompt_rgb_path).convert("RGB").resize((448, 448))) / 255.

    if os.path.exists(prompt_depth_path):
        prompt_tgt_img = Image.open(prompt_depth_path)
        if prompt_tgt_img.mode == 'P':  # 如果是调色板模式，正确转成 RGB 以防颜色错乱
            prompt_tgt_img = prompt_tgt_img.convert('RGB')

        prompt_tgt = np.array(prompt_tgt_img.resize((448, 448), Image.NEAREST)) / 255.

        if len(prompt_tgt.shape) == 2:
            prompt_tgt = np.repeat(prompt_tgt[:, :, np.newaxis], 3, axis=2)
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

    img_list = glob.glob(os.path.join(args.input_dir, "**/*.jpg"), recursive=True) + glob.glob(
        os.path.join(args.input_dir, "**/*.png"), recursive=True)
    img_list = sorted(img_list)

    if args.num_images is not None and args.num_images < len(img_list):
        total_before = len(img_list)
        rng = np.random.RandomState(args.seed)
        indices = rng.choice(total_before, size=args.num_images, replace=False)
        img_list = [img_list[i] for i in sorted(indices)]
        print(f"[INFERENCE] Sampled {args.num_images} images (seed={args.seed}) from {total_before} total.")

    for img_path in tqdm.tqdm(img_list, desc="Running Inference Loop"):
        try:
            img_raw = Image.open(img_path).convert("RGB")
            orig_size = img_raw.size
            img = np.array(img_raw.resize((448, 448))) / 255.

            if getattr(args, 'use_fc_vict', False):
                current_prompt_img = frequency_calibrate_prompt(prompt_img, img, args)
            else:
                current_prompt_img = prompt_img

            img_cat = (np.concatenate((current_prompt_img, img), axis=0) - IMAGENET_MEAN_np) / IMAGENET_STD_np
            tgt_cat = (np.concatenate((prompt_tgt, np.zeros((448, 448, 3))),
                                      axis=0) - IMAGENET_MEAN_np) / IMAGENET_STD_np

            out_path = os.path.join(args.save_dir, os.path.basename(img_path).replace(".jpg", ".png"))
            run_one_image_mt3(img_cat, tgt_cat, orig_size, model, out_path, device, args, global_winner_layer)
        except Exception as e:
            print(f"[Error processing {img_path}]: {e}")
            continue

    if args.mode == 'inference':
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, "winning_groups.txt"), "w") as f:
            f.write(f"Global Winner Layer: {global_winner_layer}")


if __name__ == '__main__':
    main()