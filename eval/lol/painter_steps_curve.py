# --------------------------------------------------------
# Based on Painter, MAE, BEiT, detectron2, Mask2Former, bts, mmcv, mmdetetection, mmpose, MIRNet, MPRNet, and Uformer codebases
# Adapted for Test-Time Adaptation (TTA) with Gradient Utility & Freq Drift
# Added: Frequency-Calibrated Visual In-Context Tuning (FC-VICT)
# --------------------------------------------------------
import warnings
import logging
from contextlib import nullcontext
# 屏蔽各种运行时警告
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

import sys
import os
import argparse
import copy
import csv
import json
import math
import random
import time
import torch
import torch.fft
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import glob
import tqdm
from PIL import Image, ImageFile
import traceback

from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    torch.set_float32_matmul_precision('high')

ImageFile.LOAD_TRUNCATED_IMAGES = True

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

# 注意：这里导入你的模型库
import models_nyu as models_vpt_FA2_new

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
IMAGENET_MEAN_np = IMAGENET_MEAN.reshape(1, 1, 3)
IMAGENET_STD_np = IMAGENET_STD.reshape(1, 1, 3)


def sdp_ctx():
    """Return a compatible SDP kernel context across torch versions."""
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
    parser.add_argument('--prompt', type=str, default='100')
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)

    # TTA params
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--lr_backbone', type=float, default=0.0)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--curve_csv', type=str, required=True)
    parser.add_argument('--failures_file', type=str, required=True)
    parser.add_argument('--max_images', type=int, default=0)

    parser.add_argument('--vpt_type', type=str, default='shallow')
    parser.add_argument('--prompt_length', type=int, default=100)
    parser.add_argument('--vpt_target_layer', nargs='+', type=int, default=[0])

    parser.add_argument('--mode', type=str, choices=['inference', 'anchor', 'eval_drift', 'extract_drift'], default='inference')
    parser.add_argument('--sample_list', type=str, default=None)
    parser.add_argument('--anchor_stats_path', type=str, default=None)
    parser.add_argument('--global_winner_path', type=str, default=None)
    parser.add_argument('--calibration_path', type=str, default='calibration_baseline.pt')

    # LOL-specific params
    parser.add_argument('--split', type=str, default='val', choices=['train', 'train-c', 'val'])
    parser.add_argument('--perturbation', type=str, default='clean')
    parser.add_argument('--level', type=str, default='0')
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=2026, help='Random seed for reproducible TTA runs.')
    parser.add_argument('--allow_cpu_fallback', action='store_true',
                        help='Fallback to CPU if CUDA OOM happens during model move.')

    # 联合门控机制超参数
    parser.add_argument('--num_bands', type=int, default=10, help='径向频带划分数量B')
    parser.add_argument('--lambda_gate', type=float, default=0.5, help='频域与梯度效用的融合权重')

    parser.add_argument('--prompt_mode', type=str, choices=['zero-shot', 'one-shot'],
                        default='zero-shot',
                        help='zero-shot: our485 干净 low/high; one-shot: 低光侧为 corruption 下 our485/low，高光为干净 our485/high')

    # =========================================================================
    # FC-VICT 新增参数
    # =========================================================================
    parser.add_argument('--use_fc_vict', action='store_true', help='是否开启 Frequency-Calibrated VICT')
    parser.add_argument('--r_L', type=float, default=0.03, help='低频遮罩半径 (0~1.0)')
    parser.add_argument('--r_H', type=float, default=0.15, help='高频遮罩半径 (0~1.0)')
    parser.add_argument('--rho', type=float, default=0.5, help='低频混合权重')
    parser.add_argument('--alpha_min', type=float, default=0.5, help='高频能量校准最小值')
    parser.add_argument('--alpha_max', type=float, default=1.5, help='高频能量校准最大值')
    parser.add_argument('--lambda_f', type=float, default=0.01, help='频域正则化 Loss 权重')

    return parser.parse_args()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(device_arg):
    if device_arg == 'cpu':
        return torch.device('cpu')
    if device_arg == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA, but torch.cuda.is_available() is False.")
        return torch.device('cuda')
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_model(chkpt_dir, arch, args, device):
    if not hasattr(models_vpt_FA2_new, arch):
        raise AttributeError(f"Model {arch} not found")
    
    model = getattr(models_vpt_FA2_new, arch)(
        vpt_on=True, vpt_type=args.vpt_type, prompt_length=args.prompt_length,
        vpt_target_layer=args.vpt_target_layer, use_act_checkpoint=False 
    )
        
    # Load checkpoint on CPU first to avoid early CUDA OOM.
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    try:
        model.to(device)
    except torch.cuda.OutOfMemoryError:
        if device.type == 'cuda' and args.allow_cpu_fallback:
            print("[Warn] CUDA OOM while moving model to GPU. Falling back to CPU.")
            device = torch.device('cpu')
            model.to(device)
        else:
            raise
    model.eval()

    return model, device


# =========================================================================
# FC-VICT 频域核心函数
# =========================================================================
def get_freq_masks(h, w, r_L=0.03, r_H=0.15, device='cpu'):
    """生成低频、中频、高频 Mask"""
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
    """
    策略一: Frequency-Calibrated Prompt
    低频风格对齐 + 高频能量校准
    """
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
    
    # 1.1 低频风格对齐
    A_L = (1 - args.rho) * Ax + args.rho * At
    
    # 1.2 高频能量校准
    sum_M_H = M_H.sum() + 1e-8
    mean_At_H = (At * M_H).sum(dim=(-2, -1), keepdim=True) / sum_M_H
    mean_Ax_H = (Ax * M_H).sum(dim=(-2, -1), keepdim=True) / sum_M_H
    
    alpha_H = mean_At_H / (mean_Ax_H + 1e-8)
    alpha_H = torch.clamp(alpha_H, min=args.alpha_min, max=args.alpha_max)
    A_H = alpha_H * Ax
    
    # 1.3 合成频谱
    Af = M_L * A_L + M_M * Ax + M_H * A_H
    
    # 构建复数并还原
    Ff = torch.polar(Af, Px)
    Ff = torch.fft.ifftshift(Ff, dim=(-2, -1))
    xf = torch.fft.ifft2(Ff).real
    
    xf = torch.clamp(xf, 0, 1)
    return xf.squeeze(0).permute(1, 2, 0).numpy()

def compute_mfm_frequency_loss(pred, target, M_L, M_H):
    """
    策略三: Frequency Regularization
    计算频带复数距离
    """
    # [FIX]: 避免 cuFFT 底层显存碎片报错，移动到 CPU 计算
    pred_cpu = pred.float().cpu()
    target_cpu = target.float().cpu()
    
    # [FIX]: 致命Bug修复！必须加入 norm="forward" 避免频域幅度成十万倍放大导致的梯度爆炸！
    F_pred = torch.fft.fftshift(torch.fft.fft2(pred_cpu, norm="forward"), dim=(-2, -1))
    F_target = torch.fft.fftshift(torch.fft.fft2(target_cpu, norm="forward"), dim=(-2, -1))
    
    # 算完距离立即转回设备所在 GPU 保证梯度相连
    dist = torch.abs(F_pred - F_target).to(pred.device)
    mask = M_L if torch.rand(1).item() < 0.5 else M_H
    
    loss_freq = (dist * mask).sum() / (mask.expand_as(dist).sum() + 1e-8)
    return loss_freq


# =========================================================================
# 原有基线函数
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

    c_h, c_w = h // 2, w // 2
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

    model.eval()
    if hasattr(model, 'active_vpt_layers'):
        model.active_vpt_layers = []
    
    with torch.no_grad(), sdp_ctx():
        for img_path in tqdm.tqdm(img_paths, desc="Processing Anchor Mode"):
            try:
                img = np.array(Image.open(img_path).convert("RGB").resize((args.input_size, args.input_size))) / 255.
                
                # [FC-VICT 替换] 策略一：无条件应用
                if getattr(args, 'use_fc_vict', False):
                    current_prompt_img = frequency_calibrate_prompt(prompt_img, img, args)
                else:
                    current_prompt_img = prompt_img
                    
                img_cat = (np.concatenate((current_prompt_img, img), axis=0) - IMAGENET_MEAN_np) / IMAGENET_STD_np
                tgt_cat = (np.concatenate((prompt_tgt, prompt_tgt), axis=0) - IMAGENET_MEAN_np) / IMAGENET_STD_np

                x = torch.tensor(img_cat).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)
                tgt = torch.tensor(tgt_cat).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)

                _, _, _, inter_feats = model(x, tgt, mask_query, torch.ones_like(tgt))

                for l in anchor_layers:
                    query_feat = extract_query_tokens(inter_feats[l], num_patches)
                    e_hat = get_spectral_features(query_feat, args.input_size//16, args.input_size//16, num_bands=args.num_bands)
                    S_accum[l].append(e_hat.cpu())
                count += 1
            except Exception as e:
                print(f"[Anchor][Skip] {img_path}: {e}")
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
        print("❌ Failed to compute anchor stats.")


def run_eval_drift_mode(args, model, device, prompt_img, prompt_tgt):
    print(f"--- Running {args.mode.upper()} Mode: Small Sample Voting (Freq Drift + Relative Gradient Utility) ---")
    anchor_stats = torch.load(args.anchor_stats_path, map_location='cpu')

    with open(args.sample_list, 'r') as f:
        img_paths = [line.strip() for line in f.readlines() if line.strip()]

    num_patches = model.patch_embed.num_patches
    h_img = args.input_size
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
            img = np.array(Image.open(img_path).convert("RGB").resize((h_img, h_img))) / 255.
            
            # [FC-VICT 替换] 策略一：无条件应用
            if getattr(args, 'use_fc_vict', False):
                current_prompt_img = frequency_calibrate_prompt(prompt_img, img, args)
            else:
                current_prompt_img = prompt_img

            img_cat = (np.concatenate((current_prompt_img, img), axis=0) - IMAGENET_MEAN_np) / IMAGENET_STD_np
            tgt_cat = (np.concatenate((prompt_tgt, prompt_tgt), axis=0) - IMAGENET_MEAN_np) / IMAGENET_STD_np

            x = torch.tensor(img_cat).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)
            tgt = torch.tensor(tgt_cat).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)
            
            x.requires_grad_(False)

            model.eval()
            if hasattr(model, 'active_vpt_layers'):
                model.active_vpt_layers = []

            with torch.no_grad():
                _, _, _, inter_feats = model(x, tgt, mask_query, torch.ones_like(tgt))

            d_freq = {}
            for l in layer_keys:
                query_feat = extract_query_tokens(inter_feats[l], num_patches)
                e_hat_test = get_spectral_features(query_feat, h_img//16, h_img//16, num_bands=args.num_bands)

                mu_train = anchor_stats[l]['mu'].to(device)
                sigma_train = anchor_stats[l]['std'].to(device)

                d_l_freq = torch.mean(torch.abs((e_hat_test - mu_train) / (sigma_train + 1e-6))).item()
                d_freq[l] = d_l_freq

            model.train()
            if hasattr(model, 'active_vpt_layers'):
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
                _, y_tokens_1, _, _ = model(x, tgt, mask_query, valid)
                pred_query = model.unpatchify(y_tokens_1)[:, :, h_img:, :]

                tgt_cycle = torch.cat([torch.zeros_like(pred_query), pred_query.detach()], dim=2)
                _, y_tokens_2, _, _ = model(x, tgt_cycle, mask_support, valid)
                recon_support = model.unpatchify(y_tokens_2)[:, :, :h_img, :]

                loss_cycle = F.smooth_l1_loss(recon_support, tgt[:, :, :h_img, :])
                loss = loss_cycle
                
                # 策略三：加一个很小权重的频域正则
                if getattr(args, 'use_fc_vict', False):
                    loss_freq = compute_mfm_frequency_loss(recon_support, tgt[:, :, :h_img, :], M_L, M_H)
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
                    d_util[layer_idx] = raw_grads.get(layer_idx, 0.0) / mean_grad

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


def run_one_image_mt3(
    img, tgt, size, rgb_gt, img_name, model, device, args,
    global_winner_layer, experiment_start
):
    group_mapping = {
        'group1': [0, 1, 2],
        'group2': [10, 11, 12],
        'group3': [21, 22, 23]
    }
    active_target_layers = group_mapping[global_winner_layer]

    model.eval()
    if hasattr(model, 'active_vpt_layers'):
        if args.steps == 0:
            model.active_vpt_layers = []
        else:
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
    h_img = args.input_size
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

    records = []
    model.train()
    with sdp_ctx():
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            
            _, y_tokens_1, _, _ = model(x, tgt, mask_query, valid)
            pred_query = model.unpatchify(y_tokens_1)[:, :, h_img:, :]

            tgt_cycle = torch.cat([torch.zeros_like(pred_query), pred_query.detach()], dim=2)
            _, y_tokens_2, _, _ = model(x, tgt_cycle, mask_support, valid)
            recon_support = model.unpatchify(y_tokens_2)[:, :, :h_img, :]

            # 主 Loss
            loss_cycle = F.smooth_l1_loss(recon_support, tgt[:, :, :h_img, :])
            loss = loss_cycle
            
            # [FC-VICT] 策略三：加一个很小权重的频域正则
            if getattr(args, 'use_fc_vict', False):
                loss_freq = compute_mfm_frequency_loss(recon_support, tgt[:, :, :h_img, :], M_L, M_H)
                loss = loss + args.lambda_f * loss_freq
                
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                _, y_tokens_final, _, _ = model(x, tgt, mask_query, valid)
                image_1 = (
                    model.unpatchify(y_tokens_final)[:, :, h_img:, :][0]
                    .permute(1, 2, 0).cpu().numpy()
                )
                output = image_1 * IMAGENET_STD + IMAGENET_MEAN
                output = F.interpolate(
                    torch.tensor(output)[None, ...].permute(0, 3, 1, 2),
                    size=[size[1], size[0]],
                    mode='bicubic',
                ).permute(0, 2, 3, 1)[0].numpy()
                rgb_restored = np.clip(output, 0, 1)
                records.append({
                    'image': img_name,
                    'step': step + 1,
                    'psnr': float(psnr_loss(rgb_restored, rgb_gt)),
                    'elapsed_s': time.monotonic() - experiment_start,
                })
            model.train()

    if hasattr(model, 'prompt_embeddings'):
        for p, backup_p in zip(model.prompt_embeddings, backup_states['prompt_embeddings']):
            with torch.no_grad(): p.copy_(backup_p)
    for m, state in zip(modules_to_optimize, backup_module_states):
        m.load_state_dict(state)

    for param in model.parameters(): param.requires_grad = False
    optimizer.zero_grad(set_to_none=True)
    del optimizer

    return records


def _our485_clean_pair(dataset_dir, p):
    """Zero-shot: LOL 提示图使用 our485 干净 low + high。"""
    our_low = os.path.join(dataset_dir, "light_enhance", "our485", "low", "{}.png".format(p))
    our_high = os.path.join(dataset_dir, "light_enhance", "our485", "high", "{}.png".format(p))
    if os.path.isfile(our_low) and os.path.isfile(our_high):
        print(f"[Prompt] zero-shot (our485) -> low={our_low}")
        return our_low, our_high
    raise FileNotFoundError(
        "our485 prompt pair not found for id {!r}: {} and {}.".format(p, our_low, our_high)
    )


def resolve_prompt_paths(args):
    """Resolve LOL prompt image paths (low / high) for VPT. Anchor always uses clean our485 pair."""
    dataset_dir = "datasets/"
    p = args.prompt

    if args.mode == 'anchor':
        img2_path = dataset_dir + "light_enhance/our485/low/{}.png".format(p)
        tgt2_path = dataset_dir + "light_enhance/our485/high/{}.png".format(p)
        print(f"[Prompt] anchor mode (clean our485) -> low={img2_path}")
        return img2_path, tgt2_path

    if args.split == 'train':
        img2_path = dataset_dir + "light_enhance/our485/low/{}.png".format(p)
        tgt2_path = dataset_dir + "light_enhance/our485/high/{}.png".format(p)
        print(f"[Prompt] train split -> low={img2_path}")
        return img2_path, tgt2_path

    if args.split == 'train-c':
        if args.prompt_mode == 'zero-shot':
            return _our485_clean_pair(dataset_dir, p)

        parts = os.path.normpath(args.input_dir).split(os.sep)
        try:
            idx = parts.index('light_enhance-c')
            corruption_name = parts[idx + 1]
            severity = parts[idx + 2]
        except (ValueError, IndexError):
            corruption_name = args.perturbation
            severity = args.level
        if corruption_name == 'clean':
            return _our485_clean_pair(dataset_dir, p)

        img2_path = os.path.join(
            dataset_dir, "light_enhance-c", corruption_name, str(severity),
            "light_enhance", "our485", "low", "{}.png".format(p)
        )
        tgt2_path = os.path.join(dataset_dir, "light_enhance", "our485", "high", "{}.png".format(p))
        print(f"[Prompt] one-shot (our485 corrupt low + clean high) -> low={img2_path}")
        return img2_path, tgt2_path

    assert args.split == 'val'
    if args.perturbation == 'clean':
        return _our485_clean_pair(dataset_dir, p)
    if args.prompt_mode == 'zero-shot':
        return _our485_clean_pair(dataset_dir, p)

    img2_path = os.path.join(
        dataset_dir, "light_enhance-c", args.perturbation, args.level,
        "light_enhance", "our485", "low", "{}.png".format(p)
    )
    tgt2_path = os.path.join(dataset_dir, "light_enhance", "our485", "high", "{}.png".format(p))
    print(f"[Prompt] val one-shot (our485) -> low={img2_path}")
    return img2_path, tgt2_path


def main():
    args = get_args()
    set_random_seed(args.seed)
    device = resolve_device(args.device)
    model, device = prepare_model(args.ckpt_path, args.model, args, device)

    img2_path, tgt2_path = resolve_prompt_paths(args)

    prompt_img = np.array(Image.open(img2_path).convert("RGB").resize((args.input_size, args.input_size))) / 255.
    prompt_tgt = np.array(Image.open(tgt2_path).convert("RGB").resize((args.input_size, args.input_size))) / 255.

    if args.mode == 'anchor':
        run_anchor_mode(args, model, device, prompt_img, prompt_tgt)
        return
    elif args.mode in ['eval_drift', 'extract_drift']:
        run_eval_drift_mode(args, model, device, prompt_img, prompt_tgt)
        return

    with open(args.global_winner_path, 'r') as f:
        global_winner_layer = f.read().strip()

    print(f"\n[{args.mode.upper()}] Voted Winner Group: {global_winner_layer}")

    img_list = sorted(glob.glob(os.path.join(args.input_dir, "*.png")))

    psnr_val_rgb = []
    ssim_val_rgb = []
    
    os.makedirs(args.save_dir, exist_ok=True)

    for img_path in tqdm.tqdm(img_list, desc="Running Inference Loop"):
        try:
            img_name = os.path.basename(img_path)
            out_path = os.path.join(args.save_dir, img_name)

            img_org = Image.open(img_path).convert("RGB")
            size = img_org.size
            img = np.array(img_org.resize((args.input_size, args.input_size))) / 255.

            gt_path = img_path.replace('low', 'high').replace(f"light_enhance-c/{args.perturbation}/{args.level}/", '')
            rgb_gt = np.array(Image.open(gt_path).convert("RGB")) / 255.

            # [FC-VICT 替换] 策略一：无条件应用 (不使用策略二)
            if getattr(args, 'use_fc_vict', False):
                current_prompt_img = frequency_calibrate_prompt(prompt_img, img, args)
            else:
                current_prompt_img = prompt_img

            img_cat = (np.concatenate((current_prompt_img, img), axis=0) - IMAGENET_MEAN_np) / IMAGENET_STD_np
            tgt_cat = (np.concatenate((prompt_tgt, prompt_tgt), axis=0) - IMAGENET_MEAN_np) / IMAGENET_STD_np

            output = run_one_image_mt3(img_cat, tgt_cat, size, model, device, args, global_winner_layer)
            
            rgb_restored = np.clip(output, 0, 1)

            psnr = psnr_loss(rgb_restored, rgb_gt)
            ssim = ssim_loss(rgb_restored, rgb_gt, multichannel=True)
            psnr_val_rgb.append(psnr)
            ssim_val_rgb.append(ssim)

            output_img = rgb_restored * 255
            Image.fromarray(output_img.astype(np.uint8)).save(out_path)

        except Exception as e:
            print(f"[Inference][Skip] Error on {img_path}: {e}")
            continue

    if args.mode == 'inference':
        avg_psnr = sum(psnr_val_rgb) / len(img_list) if psnr_val_rgb else 0
        avg_ssim = sum(ssim_val_rgb) / len(img_list) if ssim_val_rgb else 0
        
        with open(os.path.join(args.save_dir, "winning_groups.txt"), "w") as f:
            f.write(f"Global Winner Layer: {global_winner_layer}")
            
        print(f"\nFINAL_METRICS: PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")

def curve_main():
    args = get_args()
    if args.mode != 'inference':
        raise ValueError("The curve entry point only supports --mode inference.")
    if args.steps < 1:
        raise ValueError("--steps must be at least 1.")
    set_random_seed(args.seed)
    device = resolve_device(args.device)
    model, device = prepare_model(args.ckpt_path, args.model, args, device)

    img2_path, tgt2_path = resolve_prompt_paths(args)
    prompt_img = np.array(
        Image.open(img2_path).convert("RGB").resize(
            (args.input_size, args.input_size)
        )
    ) / 255.
    prompt_tgt = np.array(
        Image.open(tgt2_path).convert("RGB").resize(
            (args.input_size, args.input_size)
        )
    ) / 255.

    with open(args.global_winner_path, 'r', encoding='utf-8') as winner_file:
        global_winner_layer = winner_file.read().strip()
    if global_winner_layer != 'group2':
        raise ValueError(
            f"This controlled curve experiment requires group2, "
            f"got {global_winner_layer!r}."
        )
    print(f"[CURVE] Fixed Winner Group: {global_winner_layer}")
    print(
        f"[CURVE] steps={args.steps}, prompt_length={args.prompt_length}, "
        f"seed={args.seed}, perturbation={args.perturbation}, level={args.level}"
    )

    img_list = sorted(glob.glob(os.path.join(args.input_dir, "*.png")))
    if args.max_images > 0:
        img_list = img_list[:args.max_images]
    if not img_list:
        raise RuntimeError(f"No PNG images found in {args.input_dir}")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(
        os.path.dirname(os.path.abspath(args.curve_csv)), exist_ok=True
    )
    os.makedirs(
        os.path.dirname(os.path.abspath(args.failures_file)), exist_ok=True
    )

    failures = []
    successful_images = 0
    experiment_start = time.monotonic()
    with open(args.curve_csv, 'w', newline='', encoding='utf-8') as curve_file:
        writer = csv.DictWriter(
            curve_file, fieldnames=['image', 'step', 'psnr', 'elapsed_s']
        )
        writer.writeheader()
        for img_path in tqdm.tqdm(img_list, desc="Running Step Curves"):
            try:
                img_name = os.path.basename(img_path)
                img_org = Image.open(img_path).convert("RGB")
                size = img_org.size
                img = np.array(
                    img_org.resize((args.input_size, args.input_size))
                ) / 255.
                gt_path = img_path.replace('low', 'high').replace(
                    f"light_enhance-c/{args.perturbation}/{args.level}/", ''
                )
                rgb_gt = np.array(Image.open(gt_path).convert("RGB")) / 255.
                if getattr(args, 'use_fc_vict', False):
                    current_prompt_img = frequency_calibrate_prompt(
                        prompt_img, img, args
                    )
                else:
                    current_prompt_img = prompt_img
                img_cat = (
                    np.concatenate((current_prompt_img, img), axis=0)
                    - IMAGENET_MEAN_np
                ) / IMAGENET_STD_np
                tgt_cat = (
                    np.concatenate((prompt_tgt, prompt_tgt), axis=0)
                    - IMAGENET_MEAN_np
                ) / IMAGENET_STD_np
                image_records = run_one_image_mt3(
                    img_cat, tgt_cat, size, rgb_gt, img_name, model, device,
                    args, global_winner_layer, experiment_start
                )
                if len(image_records) != args.steps:
                    raise RuntimeError(
                        f"{img_name}: expected {args.steps} records, "
                        f"got {len(image_records)}"
                    )
                writer.writerows(image_records)
                curve_file.flush()
                successful_images += 1
                print(
                    f"[CURVE] completed {img_name}: "
                    f"{successful_images}/{len(img_list)}"
                )
            except Exception as exc:
                failures.append({
                    'image': img_path,
                    'exception_type': type(exc).__name__,
                    'message': str(exc),
                    'traceback': traceback.format_exc(),
                })
                print(f"[CURVE][FAIL] {img_path}: {exc}")
                break

    status = {
        'valid': not failures and successful_images == len(img_list),
        'expected_images': len(img_list),
        'successful_images': successful_images,
        'steps': args.steps,
        'expected_rows': len(img_list) * args.steps,
        'failures': failures,
        'elapsed_s': time.monotonic() - experiment_start,
    }
    with open(args.failures_file, 'w', encoding='utf-8') as failure_file:
        json.dump(status, failure_file, indent=2, ensure_ascii=False)
    print("CURVE_STATUS: " + json.dumps(status, ensure_ascii=False))
    if not status['valid']:
        raise RuntimeError("Curve experiment is invalid; see failures_file.")


if __name__ == '__main__':
    curve_main()
