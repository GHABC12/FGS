# train_painter_MT3.py
# ==============================================================================
# MT3 Meta-Training for Painter (Option 1: functional_call / Exact MAML)
# ==============================================================================

import os
import argparse
import glob
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import tqdm


# [关键引入] PyTorch 2.0+ 的函数式调用工具
try:
    from torch.func import functional_call
except ImportError:
    print("Error: torch.func not found. Please upgrade to PyTorch >= 2.0")
    exit(1)

import sys
import os

# [修复] 获取当前脚本的绝对路径，并将其所在目录加入 python 搜索路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 再次尝试导入，如果失败直接报错退出，不再往下跑
try:
    import models_painter
except ImportError as e:
    print(f"Error: {e}")
    print(f"Python could not find 'models_painter.py'.")
    print(f"We added this directory to path: {current_dir}")
    print("Please check if 'models_painter.py' exists in that directory.")
    exit(1)
# ================= 配置区 =================
# [请确认] 这里指向 sync 文件夹的根目录
TRAIN_DATA_ROOT = "datasets/nyu_depth_v2/sync/"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


# ================= 1. 数据增强 (MT3 Task Simulator) =================
class MT3Augmentation:
    def __init__(self):
        self.aug = transforms.Compose([
            transforms.RandomResizedCrop(448, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.2),
        ])

    def add_noise(self, x):
        if torch.rand(1) < 0.3:
            noise = torch.randn_like(x) * 0.05
            return torch.clamp(x + noise, 0, 1)
        return x

    def __call__(self, img):
        return self.add_noise(self.aug(img))


# ================= 2. 数据集加载器 =================
class NYUTrainDataset(Dataset):
    def __init__(self, root_dir):
        # 递归搜索
        rgb_pattern = os.path.join(root_dir, "**", "rgb_*.jpg")
        depth_pattern = os.path.join(root_dir, "**", "sync_depth_*.png")

        print(f"Scanning files in {root_dir}...")
        self.rgb_files = sorted(glob.glob(rgb_pattern, recursive=True))
        self.depth_files = sorted(glob.glob(depth_pattern, recursive=True))

        if len(self.rgb_files) == 0:
            print(f"Error: No RGB images found! Checked pattern: {rgb_pattern}")
            rgb_pattern_png = os.path.join(root_dir, "**", "rgb_*.png")
            self.rgb_files = sorted(glob.glob(rgb_pattern_png, recursive=True))
            if len(self.rgb_files) > 0:
                print(f"Found {len(self.rgb_files)} RGB images with .png extension.")

        if len(self.rgb_files) != len(self.depth_files):
            print(f"Warning: Count Mismatch! RGB: {len(self.rgb_files)}, Depth: {len(self.depth_files)}")
            min_len = min(len(self.rgb_files), len(self.depth_files))
            self.rgb_files = self.rgb_files[:min_len]
            self.depth_files = self.depth_files[:min_len]

        print(f"Train Dataset Ready: {len(self.rgb_files)} pairs found.")

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        try:
            # 读取 RGB
            rgb = Image.open(self.rgb_files[idx]).convert("RGB").resize((448, 448))
            rgb = np.array(rgb) / 255.0
            # [修复] 显式转为 float (float32)
            rgb = torch.from_numpy(rgb).permute(2, 0, 1).float()

            # 读取 Depth
            depth = Image.open(self.depth_files[idx]).resize((448, 448))
            depth = np.array(depth).astype(np.float32) / 10000.0 * 255.0
            # [修复] 显式转为 float (float32)
            depth = torch.from_numpy(depth).unsqueeze(0).float() / 255.0

            return rgb, depth
        except Exception as e:
            print(f"Error loading index {idx}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))


# ================= 3. 辅助函数 =================
def prepare_model(ckpt_path, args):
    model = getattr(models_painter, args.model)()
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        print("Warning: Checkpoint path invalid or not found. Initializing random weights.")
    return model


def normalize_batch(x):
    # [关键修复] 强制转换为 float32 (float)，避免 float64 (double) 引起的类型错误
    mean = torch.tensor(IMAGENET_MEAN).float().view(1, 3, 1, 1).to(x.device)
    std = torch.tensor(IMAGENET_STD).float().view(1, 3, 1, 1).to(x.device)
    return (x - mean) / std


def get_masked_input(prompt_img, prompt_tgt, query_img, query_tgt, mask_type, model):
    # 1. 拼接
    imgs = torch.cat([prompt_img, query_img], dim=2)
    tgts = torch.cat([prompt_tgt, query_tgt], dim=2)

    # Normalize
    imgs = normalize_batch(imgs)
    tgts = normalize_batch(tgts.repeat(1, 3, 1, 1))

    # 2. Mask 设置
    num_patches = model.patch_embed.num_patches
    mask = torch.zeros(num_patches).to(prompt_img.device)

    if mask_type == 'query':
        mask[num_patches // 2:] = 1
    elif mask_type == 'prompt':
        mask[:num_patches // 2] = 1

    mask = mask.unsqueeze(0)
    valid = torch.ones_like(tgts)

    return imgs, tgts, mask, valid


# ================= 4. MAML/MT3 核心逻辑 =================
def train_mt3(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 模型与数据
    model = prepare_model(args.ckpt_path, args)
    model.to(device)
    # [关键] 转为 float32，防止模型本身有 double 参数
    model.float()
    model.train()

    dataset = NYUTrainDataset(TRAIN_DATA_ROOT)
    if len(dataset) == 0:
        print("Error: Dataset is empty. Exiting.")
        return

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    meta_optimizer = optim.AdamW(model.parameters(), lr=args.meta_lr, weight_decay=0.05)
    augmentor = MT3Augmentation()

    print(f"Start Meta-Training (Option 1: functional_call) on {len(dataset)} images...")

    for epoch in range(args.epochs):
        pbar = tqdm.tqdm(dataloader)
        total_loss_inner = 0
        total_loss_outer = 0

        for i, (imgs, depths) in enumerate(pbar):
            # [关键] 确保输入也是 float32
            imgs, depths = imgs.to(device).float(), depths.to(device).float()
            B = imgs.shape[0]
            if B < 2: continue

            half = B // 2
            prompt_img = imgs[:half]
            prompt_gt = depths[:half]
            clean_query_img = imgs[half:]
            clean_query_gt = depths[half:]

            # --- Inner Loop ---
            corrupted_query_img = augmentor(clean_query_img)

            imgs_in, tgts_in, mask_in, valid_in = get_masked_input(
                prompt_img, prompt_gt, corrupted_query_img, torch.zeros_like(clean_query_gt), 'query', model
            )

            # Forward 1
            _, y_tokens, _ = model(imgs_in, tgts_in, mask_in, valid_in)
            y_out = model.unpatchify(y_tokens)
            pred_depth_aug = y_out[:, 0:1, 448:, :]

            # Forward 2 (Cycle)
            cycle_tgt_prompt = torch.zeros_like(prompt_gt)
            imgs_cycle, tgts_cycle, mask_cycle, valid_cycle = get_masked_input(
                prompt_img, cycle_tgt_prompt, corrupted_query_img, pred_depth_aug, 'prompt', model
            )

            _, y_tokens_cycle, _ = model(imgs_cycle, tgts_cycle, mask_cycle, valid_cycle)
            y_out_cycle = model.unpatchify(y_tokens_cycle)
            recon_prompt_depth = y_out_cycle[:, 0:1, :448, :]

            loss_inner = F.smooth_l1_loss(recon_prompt_depth, prompt_gt)

            # Gradients
            grads = torch.autograd.grad(loss_inner, model.parameters(), create_graph=True)

            fast_weights = {}
            for (name, param), grad in zip(model.named_parameters(), grads):
                if grad is None:
                    fast_weights[name] = param
                else:
                    fast_weights[name] = param - args.inner_lr * grad

            # --- Outer Loop ---
            imgs_out, tgts_out, mask_out, valid_out = get_masked_input(
                prompt_img, prompt_gt, clean_query_img, clean_query_gt, 'query', model
            )

            # [修复] 确保所有 Tensor 都在同一个 device 且是 float
            # functional_call 会自动处理 device，但以防万一

            _, y_tokens_final, _ = functional_call(
                model,
                fast_weights,
                (imgs_out, tgts_out, mask_out, valid_out)
            )

            pred_clean = model.unpatchify(y_tokens_final)[:, 0:1, 448:, :]
            loss_outer = F.smooth_l1_loss(pred_clean, clean_query_gt)

            meta_optimizer.zero_grad()
            loss_outer.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            meta_optimizer.step()

            total_loss_inner += loss_inner.item()
            total_loss_outer += loss_outer.item()
            pbar.set_description(f"Ep {epoch + 1} | L_In: {loss_inner.item():.4f} | L_Out: {loss_outer.item():.4f}")
            # [新增] 手动清理显存碎片
            del grads, fast_weights, y_tokens, y_tokens_cycle, y_tokens_final
            torch.cuda.empty_cache()

        # Save Checkpoint
        if not os.path.exists("models"): os.makedirs("models")
        save_path = os.path.join("models", f"mt3_functional_epoch_{epoch + 1}.pth")
        torch.save({'model': model.state_dict()}, save_path)
        print(f"Saved: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='models/painter_vit_large/painter_vit_large.pth')
    parser.add_argument('--model', type=str, default='painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--meta_lr', type=float, default=1e-5)
    parser.add_argument('--inner_lr', type=float, default=1e-2)

    args = parser.parse_args()
    train_mt3(args)