import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

import sys
import os
import argparse
import copy
import math
import torch
import torch.fft
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import glob
import tqdm
from PIL import Image, ImageFile
import traceback
from contextlib import nullcontext

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

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
IMAGENET_MEAN_np = IMAGENET_MEAN.reshape(1, 1, 3)
IMAGENET_STD_np = IMAGENET_STD.reshape(1, 1, 3)


def sdp_ctx():
    """与 eval/lol 一致：兼容的 SDP 上下文，eval_drift / TTA 前向用 FP32 避免 cuFFT 与显存尖峰。"""
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

    parser.add_argument('--num_bands', type=int, default=10, help='径向频带划分数量B')
    parser.add_argument('--lambda_gate', type=float, default=0.5, help='频域与梯度效用的融合权重')

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
    """
    策略三: Frequency Regularization（与 eval/lol 一致）
    在 CPU 上 FFT，避免 GPU cuFFT 在显存紧张时 CUFFT_INTERNAL_ERROR / OOM。
    """
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

    model.eval()
    if hasattr(model, 'active_vpt_layers'):
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
                print(f"\n[Anchor Mode Error] Failed on image {img_path}: {e}")
                traceback.print_exc()
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
            tgt_cat = (np.concatenate((prompt_tgt, np.zeros((448, 448, 3))), axis=0) - IMAGENET_MEAN_np) / IMAGENET_STD_np

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
                e_hat_test = get_spectral_features(query_feat, 28, 28, num_bands=args.num_bands)

                mu_train = anchor_stats[l]['mu'].to(device)
                sigma_train = anchor_stats[l]['std'].to(device)

                d_l_freq = torch.mean(torch.abs((e_hat_test - mu_train) / (sigma_train + 1e-6))).item()
                d_freq[l] = d_l_freq

            del inter_feats

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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
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

    print("\n--- 预选拔分组汇总得分 (Group Total Scores) ---")
    print(f"  Group 1 (Layer 0,1,2) Score: {group1_score:.4f}")
    print(f"  Group 2 (Layer 10,11,12) Score: {group2_score:.4f}")
    print(f"  Group 3 (Layer 21,22,23) Score: {group3_score:.4f}")

    final_scores = {'group1': group1_score, 'group2': group2_score, 'group3': group3_score}
    global_winner_group = max(final_scores, key=final_scores.get)

    print(f"\n=> 🎯 Final Voted Winner Group: {global_winner_group}")

    with open(args.global_winner_path, 'w') as f:
        f.write(global_winner_group)


def run_one_image_mt3(img, tgt, size, model, out_path, device, args, global_winner_layer):
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

    model.train()
    with sdp_ctx():
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)

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
            optimizer.step()

    model.eval()
    with torch.no_grad(), sdp_ctx():
        _, y_tokens_final, _, _ = model(x, tgt, mask_query, valid)

        image_1 = model.unpatchify(y_tokens_final)[:, :, h_img:, :][0].permute(1, 2, 0).float().cpu().numpy()

        output = torch.from_numpy(image_1).permute(2, 0, 1).unsqueeze(0)
        output = torch.clip((output.permute(0, 2, 3, 1) * IMAGENET_STD + IMAGENET_MEAN) * 255.0, 0, 255)
        output = F.interpolate(output.permute(0, 3, 1, 2), size=[size[1], size[0]], mode='bicubic').squeeze(0).permute(
            1, 2, 0)
        output = output.numpy().astype(np.uint8)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        Image.fromarray(output).save(out_path)

    if hasattr(model, 'prompt_embeddings'):
        for p, backup_p in zip(model.prompt_embeddings, backup_states['prompt_embeddings']):
            with torch.no_grad(): p.copy_(backup_p)
    for m, state in zip(modules_to_optimize, backup_module_states):
        m.load_state_dict(state)

    for param in model.parameters(): param.requires_grad = False
    optimizer.zero_grad(set_to_none=True)
    del optimizer

    return global_winner_layer


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = prepare_model(args.ckpt_path, args.model, args)

    prompt_rgb_path = f"datasets/derain/train/input/{args.prompt}.jpg"
    prompt_tgt_path = f"datasets/derain/train/target/{args.prompt}.jpg"

    prompt_img = np.array(Image.open(prompt_rgb_path).convert("RGB").resize((448, 448))) / 255.
    if os.path.exists(prompt_tgt_path):
        prompt_tgt = np.array(Image.open(prompt_tgt_path).convert("RGB").resize((448, 448))) / 255.
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

    img_list = glob.glob(os.path.join(args.input_dir, "**/*.jpg"), recursive=True) + glob.glob(
        os.path.join(args.input_dir, "**/*.png"), recursive=True)

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

            rel_path = os.path.relpath(img_path, args.input_dir)
            rel_parts = rel_path.split(os.sep)
            subset = rel_parts[0] if len(rel_parts) > 1 else "default"
            out_name = os.path.basename(img_path)
            if out_name.lower().endswith('.jpg'):
                out_name = out_name[:-4] + '.png'
            out_path = os.path.join(args.save_dir, subset, out_name)

            run_one_image_mt3(img_cat, tgt_cat, orig_size, model, out_path, device, args, global_winner_layer)
        except Exception as e:
            print(f"\n[Inference Error] Failed on image {img_path}: {e}")
            traceback.print_exc()
            continue

    if args.mode == 'inference':
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, "winning_groups.txt"), "w") as f:
            f.write(f"Global Winner Layer: {global_winner_layer}")


if __name__ == '__main__':
    main()