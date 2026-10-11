import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

import sys
import os
import argparse
import copy
import math
import random
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

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
IMAGENET_MEAN_np = IMAGENET_MEAN.reshape(1, 1, 3)
IMAGENET_STD_np = IMAGENET_STD.reshape(1, 1, 3)

def sdp_ctx():
    """Return a compatible SDP kernel context across torch versions."""
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
    parser.add_argument('--prompt', type=str, default='000000391460')
    parser.add_argument('--prompt_image_path', type=str, default=None)
    parser.add_argument('--prompt_target_path', type=str, default=None)
    parser.add_argument('--prompt_corruption', type=str, default='', help='提示图片的损坏类型，为空时使用 clean 版本')
    parser.add_argument('--prompt_level', type=int, default=5, help='one-shot 提示图片的损坏强度')
    parser.add_argument('--task', type=str, choices=['semseg', 'inst'], default='semseg')
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)

    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--lr_backbone', type=float, default=0.0)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--weight_decay', type=float, default=0.0)
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
    parser.add_argument('--seed', type=int, default=20260426, help='随机种子，保证 COCO 子集实验可复现')

    return parser.parse_args()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_model(chkpt_dir, arch, args):
    model = getattr(models_vpt_FA2_new, arch)(
        vpt_on=True, vpt_type=args.vpt_type, prompt_length=args.prompt_length,
        vpt_target_layer=args.vpt_target_layer, use_act_checkpoint=False
    )
    checkpoint = torch.load(chkpt_dir, map_location='cuda:0')
    model.load_state_dict(checkpoint['model'], strict=False)
    model.cuda()
    model.eval()
    return model


def resolve_prompt_paths(args):
    if args.task == 'semseg':
        prompt_rgb_path = f"datasets/coco/train2017/{args.prompt}.jpg"
        prompt_tgt_path = f"datasets/coco/pano_sem_seg/panoptic_segm_train2017_with_color/{args.prompt}.png"
    elif args.task == 'inst':
        prompt_rgb_path = f"datasets/coco/pano_ca_inst/train_org/{args.prompt}_image_train_org.png"
        prompt_tgt_path = f"datasets/coco/pano_ca_inst/train_org/{args.prompt}_label_train_org.png"
    else:
        raise ValueError(f"Unsupported task: {args.task}")

    if args.prompt_corruption:
        root = f"datasets/coco-c/{args.prompt_corruption}/{args.prompt_level}/coco"
        if args.task == 'semseg':
            prompt_rgb_path = f"{root}/train2017/{args.prompt}.jpg"
        else:
            prompt_rgb_path = f"{root}/pano_ca_inst/train_org/{args.prompt}_image_train_org.png"

    if args.prompt_image_path:
        prompt_rgb_path = args.prompt_image_path
    if args.prompt_target_path:
        prompt_tgt_path = args.prompt_target_path

    print(f"[Prompt] task={args.task} image={os.path.abspath(prompt_rgb_path)}")
    print(f"[Prompt] task={args.task} target={os.path.abspath(prompt_tgt_path)}")
    if not os.path.isfile(prompt_rgb_path):
        raise FileNotFoundError(prompt_rgb_path)
    if not os.path.isfile(prompt_tgt_path):
        raise FileNotFoundError(prompt_tgt_path)
    return prompt_rgb_path, prompt_tgt_path

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
    # Avoid cuFFT internal memory fragmentation errors, move to CPU to compute
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
    """防范中间层特征维度返回不一致的安全补丁"""
    if layer_feat.dim() == 3:
        tokens = layer_feat[0]
    elif layer_feat.dim() == 2:
        tokens = layer_feat
    else:
        raise ValueError(f"Unsupported feature dim: {layer_feat.dim()}")
    return tokens[num_patches // 2:, :].unsqueeze(0)


def run_anchor_mode(args, model, device, prompt_img, prompt_tgt):
    print(f"--- Running Anchor Mode ({args.num_bands} Bands) ---")
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
                
                # [FC-VICT] Frequency Calibrated Prompt
                if getattr(args, 'use_fc_vict', False):
                    current_prompt_img = frequency_calibrate_prompt(prompt_img, img, args)
                else:
                    current_prompt_img = prompt_img

                img_cat = (np.concatenate((current_prompt_img, img), axis=0) - IMAGENET_MEAN_np) / IMAGENET_STD_np
                tgt_cat = (np.concatenate((prompt_tgt, np.zeros((448, 448, 3))),
                                          axis=0) - IMAGENET_MEAN_np) / IMAGENET_STD_np

                x = torch.tensor(img_cat).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)
                tgt = torch.tensor(tgt_cat).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)

                with torch.autocast(device_type='cuda', dtype=torch.float16):
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
            stats_accum[l] = {'mu': S_tensor.mean(dim=0), 'std': S_tensor.std(dim=0, unbiased=False)}
        torch.save(stats_accum, args.anchor_stats_path)
        print(f"✅ Saved global clean spectral anchor stats to {args.anchor_stats_path}")
    else:
        raise RuntimeError(f"Failed to compute anchor stats; skipped {skipped}/{len(img_paths)} samples.")


def run_eval_drift_mode(args, model, device, prompt_img, prompt_tgt):
    print(f"--- Running Eval Drift Mode: Small Sample Voting ---")
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
    first_error_reported = False

    # 预先生成掩码以防在循环中复用重复计算
    if getattr(args, 'use_fc_vict', False):
        M_L, _, M_H = get_freq_masks(h_img, h_img, args.r_L, args.r_H, device)

    for img_path in tqdm.tqdm(img_paths, desc="Voting Winners"):
        try:
            img = np.array(Image.open(img_path).convert("RGB").resize((448, 448))) / 255.
            
            # [FC-VICT] Frequency Calibrated Prompt
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
            model.active_vpt_layers = []
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    _, _, _, inter_feats = model(x, tgt, mask_query, torch.ones_like(tgt))

            d_freq = {}
            for l in layer_keys:
                query_feat = extract_query_tokens(inter_feats[l], num_patches)
                e_hat_test = get_spectral_features(query_feat, 28, 28, num_bands=args.num_bands)
                mu_train = anchor_stats[l]['mu'].to(device)
                sigma_train = anchor_stats[l]['std'].to(device)
                d_freq[l] = torch.mean(torch.abs((e_hat_test - mu_train) / (sigma_train + 1e-6))).item()

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
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    _, y_tokens_1, _, _ = model(x, tgt, mask_query, valid)
                    pred_query = model.unpatchify(y_tokens_1)[:, :, h_img:, :]
                    tgt_cycle = torch.cat([torch.zeros_like(pred_query), pred_query], dim=2)
                    _, y_tokens_2, _, _ = model(x, tgt_cycle, mask_support, valid)
                    recon_support = model.unpatchify(y_tokens_2)[:, :, :h_img, :]
                    
                    loss_cycle = F.smooth_l1_loss(recon_support, tgt[:, :, :h_img, :], beta=0.01)
                    loss = loss_cycle
                    
                    # [FC-VICT] Frequency Regularization Loss
                    if getattr(args, 'use_fc_vict', False):
                        loss_freq = compute_mfm_frequency_loss(recon_support, tgt[:, :, :h_img, :], M_L, M_H)
                        loss = loss + args.lambda_f * loss_freq

                loss.backward()

                d_util = {}
                raw_grads = {}
                for idx, layer_idx in enumerate(sorted(args.vpt_target_layer)):
                    if layer_idx in layer_keys:
                        grad = model.prompt_embeddings[idx].grad
                        # 【重要修复】：明确判断 grad 是否为 None，避免布尔判别模糊错误
                        raw_grads[layer_idx] = torch.norm(grad, p=2).item() if grad is not None else 0.0

                mean_grad = sum(raw_grads.values()) / (len(raw_grads) + 1e-8)
                for layer_idx in layer_keys:
                    d_util[layer_idx] = raw_grads.get(layer_idx, 0.0) / (mean_grad + 1e-8)

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
            tau = 1.0 

            s_tensor = torch.tensor([args.lambda_gate * freq_scores[l] + (1.0 - args.lambda_gate) * util_scores[l] for l in layer_keys], dtype=torch.float32, device=device)
            w_tensor = torch.nn.functional.softmax(s_tensor / tau, dim=0)

            for i, l in enumerate(layer_keys):
                total_sample_scores[l] += w_tensor[i].item()
            valid_count += 1
            
        except Exception as e:
            if not first_error_reported:
                traceback.print_exc()
                first_error_reported = True
            continue

    if valid_count == 0:
        raise RuntimeError("No valid images processed in eval_drift.")

    group1_score = total_sample_scores[0] + total_sample_scores[1] + total_sample_scores[2]
    group2_score = total_sample_scores[10] + total_sample_scores[11] + total_sample_scores[12]
    group3_score = total_sample_scores[21] + total_sample_scores[22] + total_sample_scores[23]

    final_scores = {'group1': group1_score, 'group2': group2_score, 'group3': group3_score}
    global_winner_group = max(final_scores, key=final_scores.get)

    print(f"\n=> 🎯 Final Voted Winner Group: {global_winner_group}")
    with open(args.global_winner_path, 'w') as f:
        f.write(global_winner_group)

    scores_path = args.global_winner_path.replace('.txt', '_scores.txt')
    with open(scores_path, 'w') as f:
        f.write(f"group1_score={group1_score:.6f}\ngroup2_score={group2_score:.6f}\ngroup3_score={group3_score:.6f}\nwinner={global_winner_group}\n")


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
    ], betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=True)
    model.train()
    
    if getattr(args, 'use_fc_vict', False):
        M_L, _, M_H = get_freq_masks(h_img, h_img, args.r_L, args.r_H, device)

    with sdp_ctx():
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                _, y_tokens_1, _, _ = model(x, tgt, mask_query, valid)
                pred_query = model.unpatchify(y_tokens_1)[:, :, h_img:, :]
                tgt_cycle = torch.cat([torch.zeros_like(pred_query), pred_query], dim=2)
                _, y_tokens_2, _, _ = model(x, tgt_cycle, mask_support, valid)
                recon_support = model.unpatchify(y_tokens_2)[:, :, :h_img, :]
                
                loss_cycle = F.smooth_l1_loss(recon_support, tgt[:, :, :h_img, :], beta=0.01)
                loss = loss_cycle
                
                # [FC-VICT] Frequency Regularization Loss
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

        image_1 = model.unpatchify(y_tokens_final)[:, :, h_img:, :][0].permute(1, 2, 0).cpu().numpy()
        output = torch.from_numpy(image_1).permute(2, 0, 1).unsqueeze(0)
        
        # 兼容 COCO 提取特征的方式；Instance 分支必须保持离散 RGB 编码，避免双线性混色
        output = torch.clip((output.permute(0, 2, 3, 1) * IMAGENET_STD + IMAGENET_MEAN) * 255, 0, 255)
        resize_mode = 'nearest' if args.task == 'inst' else 'bilinear'
        output = F.interpolate(output.permute(0, 3, 1, 2), size=[size[1], size[0]], mode=resize_mode).squeeze(0).permute(1, 2, 0)
        output = output.int().numpy().astype(np.uint8)

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
    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prompt_rgb_path, prompt_tgt_path = resolve_prompt_paths(args)
    prompt_img = np.array(Image.open(prompt_rgb_path).convert("RGB").resize((448, 448))) / 255.
    prompt_tgt = np.array(Image.open(prompt_tgt_path).convert("RGB").resize((448, 448))) / 255.

    model = prepare_model(args.ckpt_path, args.model, args)

    if args.mode == 'anchor':
        run_anchor_mode(args, model, device, prompt_img, prompt_tgt)
        return
    elif args.mode in ['eval_drift', 'extract_drift']:
        run_eval_drift_mode(args, model, device, prompt_img, prompt_tgt)
        return

    with open(args.global_winner_path, 'r') as f:
        global_winner_layer = f.read().strip()

    print(f"\n[{args.mode.upper()} | Task: {args.task}] Voted Winner Group: {global_winner_layer}")

    img_list = []
    img_list.extend(glob.glob(os.path.join(args.input_dir, "**/*.jpg"), recursive=True))
    img_list.extend(glob.glob(os.path.join(args.input_dir, "**/*.png"), recursive=True))
    img_list = sorted(set(img_list))

    failed_images = []
    for img_path in tqdm.tqdm(img_list, desc="Running Inference Loop"):
        try:
            img_raw = Image.open(img_path).convert("RGB")
            orig_size = img_raw.size
            img = np.array(img_raw.resize((448, 448))) / 255.
            
            # [FC-VICT] Frequency Calibrated Prompt
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
            failed_images.append((img_path, str(e), traceback.format_exc()))
            continue

    if args.mode == 'inference':
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, "winning_groups.txt"), "w") as f:
            f.write(f"Global Winner Layer: {global_winner_layer}")

        if failed_images:
            print(f"[Error] Inference failed on {len(failed_images)} / {len(img_list)} images.")
            for img_path, err, tb in failed_images[:10]:
                print(f"  - {img_path}: {err}")
                print(tb)
            raise RuntimeError("Inference has failed samples; stop before evaluation.")

if __name__ == '__main__':
    main()
