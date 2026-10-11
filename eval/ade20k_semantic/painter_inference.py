# 8.03对齐版本 (修复 ADE20K TTA 崩溃问题，引入调色板离散化吸附)
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
    model.active_vpt_layers = []
    
    with torch.no_grad(), torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=True):
        for img_path in tqdm.tqdm(img_paths, desc="Processing Anchor Mode"):
            try:
                img = np.array(Image.open(img_path).convert("RGB").resize((448, 448))) / 255.
                img_cat = (np.concatenate((prompt_img, img), axis=0) - IMAGENET_MEAN_np) / IMAGENET_STD_np
                tgt_cat = (np.concatenate((prompt_tgt, np.zeros((448, 448, 3))),
                                          axis=0) - IMAGENET_MEAN_np) / IMAGENET_STD_np

                x = torch.tensor(img_cat).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)
                tgt = torch.tensor(tgt_cat).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    _, _, _, inter_feats = model(x, tgt, mask_query, torch.ones_like(tgt))

                for l in anchor_layers:
                    query_feat = inter_feats[l][0][num_patches // 2:, :].unsqueeze(0)
                    e_hat = get_spectral_features(query_feat, 28, 28, num_bands=args.num_bands)
                    S_accum[l].append(e_hat.cpu())
                count += 1
            except Exception as e:
                continue

    if count > 0:
        stats_accum = {}
        for l in anchor_layers:
            S_tensor = torch.stack(S_accum[l], dim=0)
            stats_accum[l] = {'mu': S_tensor.mean(dim=0), 'std': S_tensor.std(dim=0, unbiased=False)}
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

    print(f"Evaluating {len(img_paths)} samples to vote for the winner layer...")

    # [此处与原代码相同，省略预选拔部分改动，主要看下一段]
    for img_path in tqdm.tqdm(img_paths, desc="Voting Winners"):
        try:
            img = np.array(Image.open(img_path).convert("RGB").resize((448, 448))) / 255.
            img_cat = (np.concatenate((prompt_img, img), axis=0) - IMAGENET_MEAN_np) / IMAGENET_STD_np
            tgt_cat = (np.concatenate((prompt_tgt, np.zeros((448, 448, 3))),
                                      axis=0) - IMAGENET_MEAN_np) / IMAGENET_STD_np

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
                query_feat = inter_feats[l][0][num_patches // 2:, :].unsqueeze(0)
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

            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=True):
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    _, y_tokens_1, _, _ = model(x, tgt, mask_query, valid)
                    pred_query = model.unpatchify(y_tokens_1)[:, :, h_img:, :]

                    tgt_cycle = torch.cat([torch.zeros_like(pred_query), pred_query.detach()], dim=2)
                    _, y_tokens_2, _, _ = model(x, tgt_cycle, mask_support, valid)
                    recon_support = model.unpatchify(y_tokens_2)[:, :, :h_img, :]

                    loss = F.smooth_l1_loss(recon_support, tgt[:, :, :h_img, :])
                
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
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=True):
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
                
                loss = F.smooth_l1_loss(recon_support, tgt[:, :, :h_img, :])
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    model.eval()
    with torch.no_grad(), torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=True):
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
    prompt_rgb_path = f"datasets/ade20k/images/training/{args.prompt}.jpg"
    prompt_depth_path = f"datasets/ade20k/annotations_with_color/training/{args.prompt}.png"

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

    for img_path in tqdm.tqdm(img_list, desc="Running Inference Loop"):
        try:
            img_raw = Image.open(img_path).convert("RGB")
            orig_size = img_raw.size
            img = np.array(img_raw.resize((448, 448))) / 255.

            img_cat = (np.concatenate((prompt_img, img), axis=0) - IMAGENET_MEAN_np) / IMAGENET_STD_np
            tgt_cat = (np.concatenate((prompt_tgt, np.zeros((448, 448, 3))),
                                      axis=0) - IMAGENET_MEAN_np) / IMAGENET_STD_np

            out_path = os.path.join(args.save_dir, os.path.basename(img_path).replace(".jpg", ".png"))
            run_one_image_mt3(img_cat, tgt_cat, orig_size, model, out_path, device, args, global_winner_layer)
        except Exception as e:
            continue

    if args.mode == 'inference':
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, "winning_groups.txt"), "w") as f:
            f.write(f"Global Winner Layer: {global_winner_layer}")


if __name__ == '__main__':
    main()