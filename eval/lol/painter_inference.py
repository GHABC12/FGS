# --------------------------------------------------------
# Based on Painter, MAE, BEiT, detectron2, Mask2Former, bts, mmcv, mmdetetection, mmpose, MIRNet, MPRNet, and Uformer codebases
# Adapted for Test-Time Adaptation (TTA) with Gradient Utility & Freq Drift
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
    parser.add_argument('--allow_cpu_fallback', action='store_true',
                        help='Fallback to CPU if CUDA OOM happens during model move.')

    # 联合门控机制超参数
    parser.add_argument('--num_bands', type=int, default=10, help='径向频带划分数量B')
    parser.add_argument('--lambda_gate', type=float, default=0.5, help='频域与梯度效用的融合权重')

    return parser.parse_args()


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
                img = np.array(Image.open(img_path).convert("RGB").resize((args.input_size, args.input_size))) / 255.
                img_cat = (np.concatenate((prompt_img, img), axis=0) - IMAGENET_MEAN_np) / IMAGENET_STD_np
                
                # [FIX]: 完全回归原版 LOL，Query区域使用 prompt_tgt，而非 np.zeros
                tgt_cat = (np.concatenate((prompt_tgt, prompt_tgt), axis=0) - IMAGENET_MEAN_np) / IMAGENET_STD_np

                x = torch.tensor(img_cat).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)
                tgt = torch.tensor(tgt_cat).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)

                # [FIX]: 移除 autcast，全程使用稳定 FP32 防止暗光噪声溢出
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

    print(f"Evaluating {len(img_paths)} samples to vote for the winner layer...")

    for img_path in tqdm.tqdm(img_paths, desc="Voting Winners"):
        try:
            img = np.array(Image.open(img_path).convert("RGB").resize((h_img, h_img))) / 255.
            img_cat = (np.concatenate((prompt_img, img), axis=0) - IMAGENET_MEAN_np) / IMAGENET_STD_np
            
            # [FIX]: 匹配原版推理行为
            tgt_cat = (np.concatenate((prompt_tgt, prompt_tgt), axis=0) - IMAGENET_MEAN_np) / IMAGENET_STD_np

            x = torch.tensor(img_cat).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)
            tgt = torch.tensor(tgt_cat).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)
            
            x.requires_grad_(False)

            model.eval()
            if hasattr(model, 'active_vpt_layers'):
                model.active_vpt_layers = []

            # [FIX]: 移除 autcast
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
                # [FIX]: 移除 autcast FP16，强制 FP32 计算 Loss 和梯度
                _, y_tokens_1, _, _ = model(x, tgt, mask_query, valid)
                pred_query = model.unpatchify(y_tokens_1)[:, :, h_img:, :]

                tgt_cycle = torch.cat([torch.zeros_like(pred_query), pred_query.detach()], dim=2)
                _, y_tokens_2, _, _ = model(x, tgt_cycle, mask_support, valid)
                recon_support = model.unpatchify(y_tokens_2)[:, :, :h_img, :]

                loss = F.smooth_l1_loss(recon_support, tgt[:, :, :h_img, :])
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


def run_one_image_mt3(img, tgt, size, model, device, args, global_winner_layer):
    group_mapping = {
        'group1': [0, 1, 2],
        'group2': [10, 11, 12],
        'group3': [21, 22, 23]
    }
    active_target_layers = group_mapping[global_winner_layer]

    model.eval()
    
    # ================= [ 修改点：纯 Baseline 推理时不激活随机参数 ] =================
    if hasattr(model, 'active_vpt_layers'):
        if args.steps == 0:
            model.active_vpt_layers = []
        else:
            model.active_vpt_layers = active_target_layers
    # ==============================================================================

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

    # [FIX]: 移除使用 FP16 强相关的 GradScaler
    model.train()
    with sdp_ctx():
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            
            # [FIX]: 移除 autocast
            _, y_tokens_1, _, _ = model(x, tgt, mask_query, valid)
            pred_query = model.unpatchify(y_tokens_1)[:, :, h_img:, :]

            tgt_cycle = torch.cat([torch.zeros_like(pred_query), pred_query.detach()], dim=2)
            _, y_tokens_2, _, _ = model(x, tgt_cycle, mask_support, valid)
            recon_support = model.unpatchify(y_tokens_2)[:, :, :h_img, :]

            loss = F.smooth_l1_loss(recon_support, tgt[:, :, :h_img, :])
            
            # [FIX]: 直接后向传播
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad(), sdp_ctx():
        # [FIX]: 移除 autocast
        _, y_tokens_final, _, _ = model(x, tgt, mask_query, valid)
            
        image_1 = model.unpatchify(y_tokens_final)[:, :, h_img:, :][0].permute(1, 2, 0).cpu().numpy()

        output = image_1 * IMAGENET_STD + IMAGENET_MEAN
        output = F.interpolate(torch.tensor(output)[None, ...].permute(0, 3, 1, 2), size=[size[1], size[0]], mode='bicubic').permute(0, 2, 3, 1)[0]
        output = output.numpy()

    if hasattr(model, 'prompt_embeddings'):
        for p, backup_p in zip(model.prompt_embeddings, backup_states['prompt_embeddings']):
            with torch.no_grad(): p.copy_(backup_p)
    for m, state in zip(modules_to_optimize, backup_module_states):
        m.load_state_dict(state)

    for param in model.parameters(): param.requires_grad = False
    optimizer.zero_grad(set_to_none=True)
    del optimizer

    return output


def main():
    args = get_args()
    device = resolve_device(args.device)
    model, device = prepare_model(args.ckpt_path, args.model, args, device)
    
    dataset_dir = "datasets/"
    
    if args.split == 'train':
        img2_path = dataset_dir + "light_enhance/our485/low/{}.png".format(args.prompt)
        tgt2_path = dataset_dir + "light_enhance/our485/high/{}.png".format(args.prompt)
    elif args.split == 'train-c':
        img2_path = os.path.join(dataset_dir, "light_enhance-c", args.perturbation, args.level, "light_enhance/our485/low/{}.png".format(args.prompt))
        tgt2_path = dataset_dir + "light_enhance/our485/high/{}.png".format(args.prompt)
    else:
        assert args.split == 'val'
        if args.perturbation == 'clean':
            img2_path = dataset_dir + "light_enhance/eval15/low/{}.png".format(args.prompt)
        else:
            img2_path = os.path.join(dataset_dir, "light_enhance-c", args.perturbation, args.level, "light_enhance/eval15/low/{}.png".format(args.prompt))
        tgt2_path = dataset_dir + "light_enhance/eval15/high/{}.png".format(args.prompt)

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

    img_list = glob.glob(os.path.join(args.input_dir, "*.png"))

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

            img_cat = (np.concatenate((prompt_img, img), axis=0) - IMAGENET_MEAN_np) / IMAGENET_STD_np
            
            # [FIX]: 完全回归原版 LOL，这里使用 prompt_tgt 填补 query 区域
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

if __name__ == '__main__':
    main()