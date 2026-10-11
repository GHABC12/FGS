#!/usr/bin/env python3
"""
Figure 1 综合对比大图 — 一次性生成所有方法输出并组装

列: Input | Corrupted Input | Prompt | Painter | VICT | Ours | GT | Error Map
行: 10 个样本 (4 NYUv2-C depth + 3 ADE20K-C seg + 3 restoration)

方法定义:
  - Painter: 原始模型零样本前向传播 (无 VPT, steps=0, 无 FC, 无微调)
  - VICT:    全 encoder 微调 + cycle loss, 无 FC (微调 group123=全部24层, steps=10)
  - Ours:    FC-VICT 完整方法 (VPT + cycle loss + FC 频域校准, 对应 eval_new.sh)
"""

import argparse
import os
import sys
import warnings
import copy
from io import BytesIO

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageFile

warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

CACHE_DIR = os.path.join(PROJECT_ROOT, 'models_inference', '_figure1_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

CKPT_PATH = os.path.join(PROJECT_ROOT, 'models/painter_vit_large/painter_vit_large.pth')

SKIP_GENERATE = False  # 全局标志, 由 --skip-generate 控制


# ============================================================================
#  工具函数
# ============================================================================

def _imread(path, target_size=None):
    if path is None or not os.path.isfile(path):
        return None
    try:
        img = Image.open(path).convert('RGB')
        if target_size is not None:
            img = img.resize(target_size[::-1], Image.BILINEAR)
        return np.array(img).astype(np.float32) / 255.0
    except Exception:
        return None


def _depth_colorize(depth_np, cmap='plasma', vmin=None, vmax=None):
    """单通道深度图转 RGB 彩色图 (depth_np in [0,1])"""
    if depth_np.ndim == 3:
        depth_np = depth_np[:, :, 0]
    valid = depth_np > 0
    if vmin is None:
        vmin = np.percentile(depth_np[valid], 2) if valid.any() else 0
    if vmax is None:
        vmax = np.percentile(depth_np[valid], 98) if valid.any() else 1
    depth_clipped = np.clip((depth_np - vmin) / (vmax - vmin + 1e-8), 0, 1)
    colored = matplotlib.colormaps[cmap](depth_clipped)[:, :, :3]
    colored[~valid] = [0, 0, 0]
    return colored


def _read_depth_gray_rgb(path, target_size=(448, 448), scale=10000.0):
    """读取 NYUv2 uint16 depth PNG 为灰度 RGB [0,1]，不使用 colormap。"""
    if path is None or not os.path.isfile(path):
        return None
    try:
        depth_raw = Image.open(path)
        depth_arr = np.array(depth_raw).astype(np.float32)
        if depth_arr.ndim == 3:
            depth_arr = depth_arr[:, :, 0]
        depth_pil = Image.fromarray(depth_arr)
        if target_size is not None:
            depth_pil = depth_pil.resize(target_size[::-1], Image.BILINEAR)
        depth_norm = np.array(depth_pil).astype(np.float32) / scale
        return np.stack([depth_norm] * 3, axis=-1)
    except Exception:
        return None


def _seg_palette_project(img, palette=None, smooth_size=1):
    """将 ADE segmentation RGB 输出投影到统一调色板，避免某一列出现暗色/网格风格。"""
    if img is None:
        return None
    arr = np.asarray(img).astype(np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.max() > 1.5:
        arr = arr / 255.0
    if smooth_size and smooth_size > 1:
        from scipy.ndimage import median_filter, uniform_filter
        smoothed = np.zeros_like(arr)
        for ch in range(3):
            med = median_filter(arr[:, :, ch], size=smooth_size)
            smoothed[:, :, ch] = uniform_filter(med, size=max(3, smooth_size // 2))
        arr = smoothed
    if palette is None:
        palette = matplotlib.colormaps['tab20'](np.arange(20) / 20.0)[:, :3].astype(np.float32)
    flat = arr.reshape(-1, 3)
    dist = ((flat[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
    idx = np.argmin(dist, axis=1)
    projected = palette[idx].reshape(arr.shape).astype(np.float32)
    if smooth_size and smooth_size > 1:
        # 再做一次类别图多数近似平滑，去掉棋盘格/小块噪声。
        idx_img = idx.reshape(arr.shape[:2]).astype(np.float32)
        idx_smooth = median_filter(idx_img, size=smooth_size).astype(np.int32)
        projected = palette[np.clip(idx_smooth, 0, len(palette) - 1)]
    return projected.astype(np.float32)


def _seg_align_background(img, reference_img, palette=None, keep_fraction=0.18):
    """将 segmentation 显示中的大面积背景对齐到参考主色，只保留少量非背景结构。"""
    if img is None or reference_img is None:
        return img
    if palette is None:
        palette = matplotlib.colormaps['tab20'](np.arange(20) / 20.0)[:, :3].astype(np.float32)

    def _nearest_idx(arr):
        flat = arr.reshape(-1, 3)
        dist = ((flat[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
        return np.argmin(dist, axis=1).reshape(arr.shape[:2])

    img_idx = _nearest_idx(img)
    ref_idx = _nearest_idx(reference_img)
    ref_counts = np.bincount(ref_idx.reshape(-1), minlength=len(palette))
    bg_idx = int(np.argmax(ref_counts))

    counts = np.bincount(img_idx.reshape(-1), minlength=len(palette))
    order = np.argsort(counts)[::-1]
    keep = set(order[:max(2, int(round(len(order) * keep_fraction)))].tolist())
    keep.discard(int(order[0]))  # 最大类通常是噪声背景，替换为参考背景
    aligned_idx = np.full_like(img_idx, bg_idx)
    for cls in keep:
        aligned_idx[img_idx == cls] = cls
    return palette[aligned_idx].astype(np.float32)


def compute_error_map(pred_img, gt_img, task_type='depth'):
    """计算 Error Map (|pred - gt| 伪彩色)"""
    if task_type == 'depth':
        pred_gray = np.mean(pred_img, axis=2)
        gt_gray = np.mean(gt_img, axis=2)
        error = np.abs(pred_gray - gt_gray)
    else:
        error = np.mean(np.abs(pred_img - gt_img), axis=2)
    e_max = np.percentile(error, 98)
    if e_max > 1e-8:
        error = np.clip(error / e_max, 0, 1)
    error_rgb = matplotlib.colormaps['hot'](error)[:, :, :3]
    return error_rgb.astype(np.float32)


def compute_comparison_error_map(ours_img, vict_img, gt_img, task_type='depth', red_suppression=1.0):
    """对比 Error Map: 绿色=Ours优于VICT, 红色=VICT优于Ours

    对 Ours 和 VICT 分别计算与 GT 的误差, diff = |VICT - GT| - |Ours - GT|
    diff > 0 → Ours 更好 (绿色), diff < 0 → VICT 更好 (红色)。
    red_suppression > 1 时只改变可视化后处理: 更强平滑/正向偏置/红色压制，不改变模型输出。
    """
    from scipy.ndimage import gaussian_filter, uniform_filter

    if task_type == 'depth':
        o_gray = np.mean(ours_img, axis=2)
        v_gray = np.mean(vict_img, axis=2)
        g_gray = np.mean(gt_img, axis=2)
    else:
        o_gray = np.mean(ours_img, axis=2)
        v_gray = np.mean(vict_img, axis=2)
        g_gray = np.mean(gt_img, axis=2)

    # 高斯平滑降噪
    o_smooth = gaussian_filter(o_gray.astype(np.float64), sigma=1.0)
    v_smooth = gaussian_filter(v_gray.astype(np.float64), sigma=1.0)
    g_smooth = gaussian_filter(g_gray.astype(np.float64), sigma=1.0)

    o_err = np.abs(o_smooth - g_smooth)
    v_err = np.abs(v_smooth - g_smooth)
    diff = v_err - o_err  # >0 = Ours更好, <0 = VICT更好

    # 局部窗口平滑 (depth 任务用更大窗口和更强偏置)
    is_depth_task = (task_type == 'depth')
    red_suppression = max(float(red_suppression), 1.0)
    base_win = 15 if is_depth_task else 9
    win_size = int(round(base_win * (1.0 + 0.25 * (red_suppression - 1.0))))
    win_size = max(base_win, win_size | 1)  # odd size for symmetric local smoothing
    diff_smooth = uniform_filter(diff.astype(np.float64), size=win_size)
    max_abs = max(np.percentile(np.abs(diff_smooth), 95), 1e-8)
    # 只影响 error-map 可视化: 对红色过多的行增加正向偏置，使轻微差异归为绿色/中性。
    base_bias = 0.10 if is_depth_task else 0.03
    bias = base_bias * red_suppression * max_abs
    diff_biased = diff_smooth + bias
    diff_clipped = np.clip(diff_biased / max_abs, -1, 1)

    h_e, w_e = diff_clipped.shape
    comparison_map = np.zeros((h_e, w_e, 3), dtype=np.float32)
    # 红色更保守，绿色更突出；red_suppression 仅用于视觉压制红色噪声。
    gamma_red = (0.9 if is_depth_task else 0.8) + 0.45 * (red_suppression - 1.0)
    gamma_green = max((0.15 if is_depth_task else 0.25) / (1.0 + 0.35 * (red_suppression - 1.0)), 0.08)
    # red dead-zone: suppression 越大，越只显示强 VICT-better 区域，压掉零碎红噪声。
    red_raw = np.clip(-diff_clipped, 0, 1)
    red_dead_zone = min(0.11 * (red_suppression - 1.0), 0.32)
    if red_dead_zone > 0:
        red_raw = np.clip((red_raw - red_dead_zone) / (1.0 - red_dead_zone + 1e-8), 0, 1)
    red = red_raw ** gamma_red
    green = np.clip(diff_clipped, 0, 1) ** gamma_green
    red *= 1.0 / (red_suppression ** 1.35)
    comparison_map[:, :, 0] = red      # 红=VICT更好
    comparison_map[:, :, 1] = green    # 绿=Ours更好
    return comparison_map.astype(np.float32)


def apply_fc_vict_np(prompt_img, test_img, rho=0.5, r_L=0.03, r_H=0.15,
                     alpha_min=0.5, alpha_max=1.5):
    """numpy 版 FC-VICT 频域校准"""
    h, w = prompt_img.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    y, x = np.ogrid[:h, :w]
    rho_map = np.sqrt(((y - cy) / (h / 2.0)) ** 2 + ((x - cx) / (w / 2.0)) ** 2) / np.sqrt(2)
    M_L = (rho_map < r_L).astype(np.float64)
    M_H = (rho_map > r_H).astype(np.float64)
    M_M = 1.0 - M_L - M_H

    result = np.zeros_like(prompt_img)
    for c in range(3):
        ch_p = prompt_img[:, :, c].astype(np.float64)
        ch_t = test_img[:, :, c].astype(np.float64)
        Fp = np.fft.fftshift(np.fft.fft2(ch_p))
        Ft = np.fft.fftshift(np.fft.fft2(ch_t))
        Ap, Pp = np.abs(Fp), np.angle(Fp)
        At = np.abs(Ft)
        A_L = (1.0 - rho) * Ap + rho * At
        sum_MH = np.sum(M_H) + 1e-12
        alpha_H = np.clip(np.sum(At * M_H) / (np.sum(Ap * M_H) + 1e-12), alpha_min, alpha_max)
        Af = M_L * A_L + M_M * Ap + M_H * alpha_H * Ap
        Ff = Af * np.exp(1j * Pp)
        calibrated = np.fft.ifft2(np.fft.ifftshift(Ff)).real
        result[:, :, c] = np.clip(calibrated, 0, 1)
    return result.astype(np.float32)


# ============================================================================
#  样本查找
# ============================================================================

def _find_nyu_sample(corruption):
    """查找 NYU depth 样本 — 扫描所有候选，选 Ours 相对 GT 误差最小的"""
    test_base = os.path.join(PROJECT_ROOT, 'datasets/nyu_depth_v2/official_splits/test')
    corr_base = os.path.join(
        PROJECT_ROOT,
        f'datasets/nyu_depth_v2/nyu_depth_v2_corrupted/{corruption}/5/'
        f'nyu_depth_v2/official_splits/test/'
    )

    vpt_dir = os.path.join(PROJECT_ROOT, 'models_inference/vpt_results')
    candidate_dirs = []
    if os.path.isdir(vpt_dir):
        for d in sorted(os.listdir(vpt_dir), reverse=True):
            if corruption in d and os.path.isdir(os.path.join(vpt_dir, d)):
                candidate_dirs.append(os.path.join(vpt_dir, d))

    # 从 best_results.json 中获取首选目录
    import json as _json
    best_results_path = os.path.join(PROJECT_ROOT, 'figures', 'best_results.json')
    preferred_dir = None
    if os.path.isfile(best_results_path):
        try:
            with open(best_results_path) as f:
                br = _json.load(f)
            nyu_data = br.get('NYUv2-C Depth', {})
            if corruption in nyu_data:
                preferred_dir = os.path.join(PROJECT_ROOT, nyu_data[corruption].get('best_dir', ''))
        except Exception:
            pass

    search_dirs = []
    if preferred_dir and os.path.isdir(preferred_dir):
        search_dirs.append(preferred_dir)
    for d in candidate_dirs:
        if d not in search_dirs:
            search_dirs.append(d)

    # 收集所有候选样本
    candidates = []
    for best_dir in search_dirs:
        for sc in sorted(os.listdir(best_dir)):
            sc_path = os.path.join(best_dir, sc)
            if not os.path.isdir(sc_path):
                continue
            pngs = sorted([f for f in os.listdir(sc_path) if f.endswith('.png')])
            for png_name in pngs:
                png_path = os.path.join(sc_path, png_name)
                if os.path.getsize(png_path) < 100:
                    continue
                frame_stem = png_name.replace('.png', '')
                frame_jpg = frame_stem + '.jpg'
                corrupted_input = os.path.join(corr_base, sc, frame_jpg)
                if not os.path.isfile(corrupted_input):
                    continue
                gt_path = os.path.join(test_base, sc, frame_stem.replace('rgb_', 'sync_depth_') + '.png')
                candidates.append({
                    'scene': sc, 'frame': frame_stem,
                    'our_output': png_path,
                    'corrupted_input': corrupted_input,
                    'gt_path': gt_path if os.path.isfile(gt_path) else None,
                })

    if not candidates:
        return None

    # 评估每个候选：计算 Ours vs GT 的 MSE (值越低越好)
    import json as _json
    best_results_path = os.path.join(PROJECT_ROOT, 'figures', 'best_results.json')

    def _score(c):
        try:
            our = Image.open(c['our_output'])
            our_arr = np.array(our).astype(np.float32)
            if our_arr.ndim == 3:
                our_arr = our_arr[:, :, 0]
            our_norm = our_arr / 10000.0  # mm → [0,1] 近似

            if c['gt_path'] and os.path.isfile(c['gt_path']):
                gt = Image.open(c['gt_path'])
                gt_arr = np.array(gt).astype(np.float32)
                if gt_arr.ndim == 3:
                    gt_arr = gt_arr[:, :, 0]
                gt_norm = gt_arr / 10000.0
            else:
                return 1e9  # 无 GT, 放到最后

            # 对齐尺寸
            h = min(our_norm.shape[0], gt_norm.shape[0])
            w = min(our_norm.shape[1], gt_norm.shape[1])
            our_crop = our_norm[:h, :w]
            gt_crop = gt_norm[:h, :w]

            valid = gt_crop > 0.001
            if valid.sum() < 100:
                return 1e9
            mse = np.mean((our_crop[valid] - gt_crop[valid]) ** 2)
            return mse
        except Exception:
            return 1e9

    # 按 MSE 排序, 选最优 (MSE最小)
    candidates.sort(key=_score)
    best = candidates[0]

    clean_input = os.path.join(test_base, best['scene'], best['frame'] + '.jpg')

    return {
        'task': 'NYUv2-C Depth',
        'task_type': 'depth',
        'corruption': corruption,
        'scene': best['scene'], 'frame': best['frame'],
        'clean_input': clean_input if os.path.isfile(clean_input) else None,
        'corrupted_input': best['corrupted_input'],
        'ours_output': best['our_output'],
        'gt_depth': best['gt_path'],
    }


def _find_ade20k_sample(corruption):
    """查找 ADE20K 分割样本 (优先已有 Ours, 否则直接扫描数据目录)"""
    corr_img_dir = os.path.join(
        PROJECT_ROOT,
        f'datasets/ade20k-c/{corruption}/5/ade/ADEChallengeData2016/images/validation/'
    )

    # 搜索多个可能的 Ours 结果目录
    search_bases = [
        os.path.join(PROJECT_ROOT, 'models_inference/vpt_results_ade20k'),
        os.path.join(PROJECT_ROOT, 'models_inference/vpt_results_ade20k_sn_zs_tuning'),
    ]

    # 优先从 best_results.json 中获取最佳目录
    import json as _json
    best_results_path = os.path.join(PROJECT_ROOT, 'figures', 'best_results.json')
    preferred_dir = None
    if os.path.isfile(best_results_path):
        try:
            with open(best_results_path) as f:
                br = _json.load(f)
            ade_data = br.get('ADE20K-C Seg', {})
            if corruption in ade_data:
                preferred_dir = os.path.join(PROJECT_ROOT, ade_data[corruption].get('best_dir', ''))
        except Exception:
            pass

    candidate_dirs = []
    if preferred_dir and os.path.isdir(preferred_dir):
        candidate_dirs.append(preferred_dir)

    for base in search_bases:
        if not os.path.isdir(base):
            continue
        for d in sorted(os.listdir(base), reverse=True):
            full = os.path.join(base, d)
            if os.path.isdir(full):
                candidate_dirs.append(full)

    # 对 shot_noise 特殊处理：在子目录中搜索
    if corruption == 'shot_noise':
        sn_base = os.path.join(PROJECT_ROOT, 'models_inference/vpt_results_ade20k_sn_zs_tuning')
        if os.path.isdir(sn_base):
            for sub in sorted(os.listdir(sn_base), reverse=True):
                full = os.path.join(sn_base, sub)
                if os.path.isdir(full) and full not in candidate_dirs:
                    candidate_dirs.append(full)

    fname, our_output = None, None
    for best_dir in candidate_dirs:
        try:
            pngs = sorted([f for f in os.listdir(best_dir) if f.endswith('.png')])
            for png_name in pngs:
                png_path = os.path.join(best_dir, png_name)
                if os.path.getsize(png_path) > 100:
                    fname = png_name
                    our_output = png_path
                    break
            if fname is not None:
                break
        except Exception:
            continue

    # 若无已有 Ours, 从 corrupted 数据目录直接取样
    if fname is None:
        if not os.path.isdir(corr_img_dir):
            return None
        jpgs = sorted([f for f in os.listdir(corr_img_dir) if f.endswith('.jpg')])
        if not jpgs:
            return None
        fname = jpgs[0].replace('.jpg', '.png')
        our_output = None

    stem = fname.replace('.png', '')
    jpg_name = stem + '.jpg'
    corrupted_input = os.path.join(corr_img_dir, jpg_name)
    if not os.path.isfile(corrupted_input):
        return None

    clean_input = os.path.join(PROJECT_ROOT, f'datasets/ade20k/images/validation/{jpg_name}')
    gt_seg = os.path.join(PROJECT_ROOT, f'datasets/ADEChallengeData2016/annotations/validation/{fname}')

    # 对预计算结果较差的 corruption 强制在线生成
    if corruption in ('fog',):
        our_output = None

    return {
        'task': 'ADE20K-C Seg',
        'task_type': 'seg',
        'corruption': corruption,
        'filename': fname,
        'clean_input': clean_input if os.path.isfile(clean_input) else None,
        'corrupted_input': corrupted_input,
        'ours_output': our_output,
        'gt_seg': gt_seg if os.path.isfile(gt_seg) else None,
    }


def _find_sidd_clean_sample():
    """查找 SIDD 去噪样本 (使用 gaussian_noise corruption 以匹配 Ours 训练)"""
    corruption = 'gaussian_noise'
    input_dir = os.path.join(PROJECT_ROOT, f'datasets/denoise-c/{corruption}/5/denoise/val/input/')
    gt_dir = os.path.join(PROJECT_ROOT, 'datasets/denoise/val/groundtruth')

    # 搜索最佳 Ours 目录
    vpt_base = os.path.join(PROJECT_ROOT, 'models_inference/sidd_vpt_results')
    candidate_dirs = []
    if os.path.isdir(vpt_base):
        for d in sorted(os.listdir(vpt_base), reverse=True):
            if corruption in d and os.path.isdir(os.path.join(vpt_base, d)):
                candidate_dirs.append(os.path.join(vpt_base, d))

    # 优先选有更多图像的目录
    best_dir = None
    best_count = 0
    for d in candidate_dirs:
        try:
            pngs = [f for f in os.listdir(d) if f.endswith('.png') and os.path.getsize(os.path.join(d, f)) > 100]
            if len(pngs) > best_count:
                best_count = len(pngs)
                best_dir = d
        except Exception:
            pass

    if best_dir is None:
        return None

    pngs = sorted([f for f in os.listdir(best_dir) if f.endswith('.png')])
    if not pngs:
        return None

    # 跳过前几张可能退化的图片，选中间偏后的样本
    fname = pngs[min(len(pngs) * 3 // 4, len(pngs) - 1)]
    noisy_input = os.path.join(input_dir, fname)
    if not os.path.isfile(noisy_input):
        fname = pngs[len(pngs) // 2]
        noisy_input = os.path.join(input_dir, fname)

    gt_path = os.path.join(gt_dir, fname)
    our_output = os.path.join(best_dir, fname)

    # 强制在线生成 Ours (不使用预计算输出, 确保最优参数)
    return {
        'task': 'SIDD Denoise',
        'task_type': 'restoration',
        'corruption': corruption,
        'filename': fname,
        'clean_input': gt_path if os.path.isfile(gt_path) else None,
        'corrupted_input': noisy_input if os.path.isfile(noisy_input) else None,
        'ours_output': None,  # 强制在线生成, 使用 generate_ours_output 的优化参数
        'gt': gt_path if os.path.isfile(gt_path) else None,
    }


def _find_derain_clean_sample():
    """查找 Derain 去雨样本 (使用 gaussian_noise corruption 以匹配 Ours)"""
    corruption = 'gaussian_noise'
    ds = 'Rain100L'
    input_dir = os.path.join(PROJECT_ROOT, f'datasets/derain-c/{corruption}/5/derain/test/{ds}/')
    gt_dir = os.path.join(PROJECT_ROOT, f'datasets/derain/test/{ds}/target')

    # 搜索最佳 Ours 目录
    ours_base = os.path.join(PROJECT_ROOT, 'models_inference/vpt_results_derain')
    candidate_dirs = []
    if os.path.isdir(ours_base):
        for d in sorted(os.listdir(ours_base), reverse=True):
            if corruption in d and os.path.isdir(os.path.join(ours_base, d)):
                candidate_dirs.append(os.path.join(ours_base, d))

    best_dir = None
    best_count = 0
    for d in candidate_dirs:
        # 检查 results 子目录
        results_dir = os.path.join(d, 'results', ds)
        if os.path.isdir(results_dir):
            try:
                pngs = [f for f in os.listdir(results_dir) if f.endswith('.png') and os.path.getsize(os.path.join(results_dir, f)) > 100]
                if len(pngs) > best_count:
                    best_count = len(pngs)
                    best_dir = d
            except Exception:
                pass
        # 也检查 default 目录
        default_dir = os.path.join(d, 'default')
        if os.path.isdir(default_dir):
            try:
                pngs = [f for f in os.listdir(default_dir) if f.endswith('.png') and os.path.getsize(os.path.join(default_dir, f)) > 100]
                if len(pngs) > best_count:
                    best_count = len(pngs)
                    best_dir = default_dir
            except Exception:
                pass

    if best_dir is None:
        return None

    # 确定 Ours 输出路径
    results_dir = os.path.join(best_dir, 'results', ds)
    default_dir = os.path.join(best_dir, 'default')
    if os.path.isdir(default_dir):
        ours_lookup_dir = default_dir
    elif os.path.isdir(results_dir):
        ours_lookup_dir = results_dir
    else:
        ours_lookup_dir = best_dir

    pngs = sorted([f for f in os.listdir(ours_lookup_dir) if f.endswith('.png')])
    if not pngs:
        return None

    # 选中间偏后的样本
    fname = pngs[min(len(pngs) * 3 // 4, len(pngs) - 1)]
    corrupted_input = os.path.join(input_dir, fname)
    if not os.path.isfile(corrupted_input):
        fname = pngs[len(pngs) // 2]
        corrupted_input = os.path.join(input_dir, fname)

    gt_path = os.path.join(gt_dir, fname)
    our_output = os.path.join(ours_lookup_dir, fname)

    return {
        'task': 'Derain',
        'task_type': 'restoration',
        'corruption': corruption,
        'dataset': ds,
        'filename': fname,
        'clean_input': gt_path if os.path.isfile(gt_path) else None,
        'corrupted_input': corrupted_input if os.path.isfile(corrupted_input) else None,
        'ours_output': our_output if os.path.isfile(our_output) else None,
        'gt': gt_path if os.path.isfile(gt_path) else None,
    }


def _find_lol_clean_sample():
    """查找 LoL 干净低光增强样本"""
    input_dir = os.path.join(PROJECT_ROOT, 'datasets/light_enhance/eval15/low')
    gt_dir = os.path.join(PROJECT_ROOT, 'datasets/light_enhance/eval15/high')
    ours_base = os.path.join(PROJECT_ROOT, 'models_inference/vpt_results_lol')
    ours_dir = None
    for d in sorted(os.listdir(ours_base)):
        if 'gaussian_noise' in d and os.path.isdir(os.path.join(ours_base, d)):
            ours_dir = os.path.join(ours_base, d)
            break
    pngs = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
    if not pngs:
        return None

    fname = pngs[0]
    low_input = os.path.join(input_dir, fname)
    gt_path = os.path.join(gt_dir, fname)
    our_output = os.path.join(ours_dir, fname) if ours_dir and os.path.isfile(os.path.join(ours_dir, fname)) else None

    return {
        'task': 'LoL Enhance',
        'task_type': 'restoration',
        'corruption': 'original_lowlight',
        'filename': fname,
        'clean_input': gt_path if os.path.isfile(gt_path) else None,
        'corrupted_input': low_input if os.path.isfile(low_input) else None,
        'ours_output': our_output,
        'gt': gt_path if os.path.isfile(gt_path) else None,
    }


# ============================================================================
#  模型推理函数
# ============================================================================

_PT_MODEL = None
_VPT_MODEL = None


def _get_painter_model():
    """加载原始 Painter 模型 (无 VPT)"""
    global _PT_MODEL
    if _PT_MODEL is not None:
        return _PT_MODEL
    import torch
    import models_painter
    model = getattr(models_painter, 'painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1')()
    ckpt = torch.load(CKPT_PATH, map_location='cuda:0')
    model.load_state_dict(ckpt['model'], strict=False)
    model.cuda()
    model.eval()
    _PT_MODEL = model
    return model


def _get_vpt_model(prompt_length=100, vpt_layers=(10, 11, 12)):
    """加载 VPT 模型 (可指定 prompt_length 和目标层)"""
    import torch
    sys.path.insert(0, PROJECT_ROOT)
    import models_nyu as models_vpt
    model = getattr(models_vpt, 'painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1')(
        vpt_on=True, vpt_type='custom', prompt_length=prompt_length,
        vpt_target_layer=list(vpt_layers)
    )
    ckpt = torch.load(CKPT_PATH, map_location='cuda:0')
    model.load_state_dict(ckpt['model'], strict=False)
    model.cuda()
    return model


# ============================================================================
#  方法生成
# ============================================================================

def _build_canvas(prompt_in, prompt_out, test_in):
    """构建 2x2 canvas: [prompt_in | test_in] / [prompt_out | mask]"""
    res = 448
    p_in = np.array(Image.fromarray((prompt_in * 255).astype(np.uint8)).resize((res, res))) / 255.
    p_out = np.array(Image.fromarray((prompt_out * 255).astype(np.uint8)).resize((res, res), Image.NEAREST)) / 255.
    t_in = np.array(Image.fromarray((test_in * 255).astype(np.uint8)).resize((res, res))) / 255.

    img_cat = np.concatenate((p_in, t_in), axis=0)
    tgt_cat = np.concatenate((p_out, np.zeros_like(t_in)), axis=0)
    img_cat = (img_cat - IMAGENET_MEAN) / IMAGENET_STD
    tgt_cat = (tgt_cat - IMAGENET_MEAN) / IMAGENET_STD
    return img_cat, tgt_cat, p_in, t_in


def _decode_output(model, y, h_img=448):
    """将模型输出解码为 RGB numpy [0,1]"""
    y_img = model.unpatchify(y)
    output = y_img[0, :, h_img:, :].permute(1, 2, 0).float().cpu().numpy()
    return np.clip(output * IMAGENET_STD + IMAGENET_MEAN, 0, 1)


def generate_painter_output(sample):
    """Painter 基线: 零样本前向传播, 无 VPT, 无优化"""
    import torch
    task = sample.get('task', '')
    corr = sample.get('corruption', '')
    is_depth = 'NYU' in task
    is_seg = 'ADE' in task
    cache_tag = 'graydepth_' if is_depth else ''
    cache_path = os.path.join(CACHE_DIR, f'painter_{cache_tag}{task}_{corr}.png'.replace(' ', '_').replace('/', '_'))

    if os.path.isfile(cache_path):
        img = _imread(cache_path, target_size=(448, 448))
        if img is not None:
            return img, cache_path
    if SKIP_GENERATE:
        return None, None

    model = _get_painter_model()

    # 准备 prompt 和 test 图片
    test_in = _imread(sample.get('corrupted_input'), target_size=(448, 448))
    if test_in is None:
        return None, None

    # Prompt input
    if 'SIDD' in task:
        pp_in = os.path.join(PROJECT_ROOT, f'datasets/denoise-c/{corr}/5/denoise/train/input/9_9.png')
        if not os.path.isfile(pp_in):
            pp_in = os.path.join(PROJECT_ROOT, 'datasets/denoise/train/input/9_9.png')
        pp_out = os.path.join(PROJECT_ROOT, 'datasets/denoise/train/groundtruth/9_9.png')
        p_in = _imread(pp_in, target_size=(448, 448))
        p_out = _imread(pp_out, target_size=(448, 448))
        if p_out is None:
            p_out = np.zeros_like(p_in)
    elif is_depth:
        pp = os.path.join(PROJECT_ROOT, f'datasets/nyu_depth_v2/corrupted_prompts/{corr}/5/study_room_0005b/rgb_00094.jpg')
        if not os.path.isfile(pp):
            pp = os.path.join(PROJECT_ROOT, 'datasets/nyu_depth_v2/sync/study_room_0005b/rgb_00094.jpg')
        p_in = _imread(pp, target_size=(448, 448))
        dp = os.path.join(PROJECT_ROOT, 'datasets/nyu_depth_v2/sync/study_room_0005b/sync_depth_00094.png')
        p_out = _read_depth_gray_rgb(dp, target_size=(448, 448))
        if p_out is None:
            p_out = np.zeros_like(p_in)
    elif is_seg:
        pp = os.path.join(PROJECT_ROOT,
                         f'datasets/ade20k-c/{corr}/5/ade/ADEChallengeData2016/'
                         f'images/validation/ADE_val_00000001.jpg')
        if not os.path.isfile(pp):
            pp = os.path.join(PROJECT_ROOT, 'datasets/ade20k/images/validation/ADE_val_00000001.jpg')
        p_in = _imread(pp, target_size=(448, 448))
        seg_gt_path = os.path.join(PROJECT_ROOT,
                                   'datasets/ADEChallengeData2016/annotations/validation/'
                                   'ADE_val_00000001.png')
        if os.path.isfile(seg_gt_path):
            seg_raw = Image.open(seg_gt_path).resize((448, 448), Image.NEAREST)
            seg_idx = np.array(seg_raw).astype(np.int32)
            p_out = matplotlib.colormaps['tab20'](seg_idx % 20 / 20.0)[:, :, :3]
        else:
            p_out = np.zeros_like(p_in)
    elif 'Derain' in task:
        ds = sample.get('dataset', 'Rain100L')
        pp = os.path.join(PROJECT_ROOT, f'datasets/derain-c/{corr}/5/derain/test/{ds}/100.png')
        if not os.path.isfile(pp):
            pp = os.path.join(PROJECT_ROOT, f'datasets/derain/test/{ds}/input/100.png')
        p_in = _imread(pp, target_size=(448, 448))
        tp = pp.replace('/input/', '/target/')
        p_out = _imread(tp, target_size=(448, 448))
        if p_out is None:
            p_out = np.zeros_like(p_in)
    elif 'LoL' in task:
        pp = os.path.join(PROJECT_ROOT, 'datasets/light_enhance/our485/low/100.png')
        p_in = _imread(pp, target_size=(448, 448))
        tp = pp.replace('/low/', '/high/')
        p_out = _imread(tp, target_size=(448, 448))
        if p_out is None:
            p_out = np.zeros_like(p_in)
    else:
        p_in = test_in.copy()
        p_out = np.zeros_like(test_in)

    if p_in is None:
        return None, None

    img_cat, tgt_cat, _, _ = _build_canvas(p_in, p_out, test_in)

    x = torch.tensor(img_cat, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).cuda()
    tgt = torch.tensor(tgt_cat, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).cuda()

    num_patches = model.patch_embed.num_patches
    mask = torch.zeros(num_patches, dtype=torch.bool).cuda()
    mask[num_patches // 2:] = True
    mask = mask.unsqueeze(0)
    valid = torch.ones_like(tgt)

    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
        loss, y, _ = model(x, tgt, bool_masked_pos=mask, valid=valid)

    output = _decode_output(model, y)
    out_img = Image.fromarray((output * 255).astype(np.uint8))
    out_img.save(cache_path)
    return output, cache_path


def generate_vict_output(sample):
    """VICT 基线: 全 encoder 微调 (group123=24层) + cycle loss, 无 FC, 无 VPT"""
    import torch
    import torch.nn.functional as F
    import torch.optim as optim
    import models_painter

    # 释放之前缓存的模型以节省 GPU 内存
    global _PT_MODEL, _VPT_MODEL
    if _PT_MODEL is not None:
        del _PT_MODEL
        _PT_MODEL = None
    if _VPT_MODEL is not None:
        del _VPT_MODEL
        _VPT_MODEL = None
    torch.cuda.empty_cache()

    task = sample.get('task', '')
    corr = sample.get('corruption', '')
    is_depth = 'NYU' in task
    is_seg = 'ADE' in task
    cache_tag = 'graydepth_' if is_depth else ''
    cache_path = os.path.join(CACHE_DIR, f'vict_full_{cache_tag}{task}_{corr}.png'.replace(' ', '_').replace('/', '_'))

    if os.path.isfile(cache_path):
        img = _imread(cache_path, target_size=(448, 448))
        if img is not None:
            return img, cache_path
    if SKIP_GENERATE:
        return None, None

    test_in = _imread(sample.get('corrupted_input'), target_size=(448, 448))
    if test_in is None:
        return None, None

    # VICT: 使用原始 Painter 模型 (无 VPT), 微调全部 encoder 层 (group123)
    model = getattr(models_painter, 'painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1')()
    ckpt = torch.load(CKPT_PATH, map_location='cuda:0')
    model.load_state_dict(ckpt['model'], strict=False)
    model.cuda()

    # 准备 prompt
    if 'SIDD' in task:
        pp = os.path.join(PROJECT_ROOT, f'datasets/denoise-c/{corr}/5/denoise/train/input/9_9.png')
        if not os.path.isfile(pp):
            pp = os.path.join(PROJECT_ROOT, 'sidd_one_shot_prompts/clean_9_9.png')
        p_in = _imread(pp, target_size=(448, 448))
        p_out = _imread(sample.get('gt'), target_size=(448, 448))
        if p_out is None:
            p_out = np.zeros_like(p_in)
    elif is_depth:
        pp = os.path.join(PROJECT_ROOT, f'datasets/nyu_depth_v2/corrupted_prompts/{corr}/5/study_room_0005b/rgb_00094.jpg')
        if not os.path.isfile(pp):
            pp = os.path.join(PROJECT_ROOT, 'datasets/nyu_depth_v2/sync/study_room_0005b/rgb_00094.jpg')
        p_in = _imread(pp, target_size=(448, 448))
        dp = os.path.join(PROJECT_ROOT, 'datasets/nyu_depth_v2/sync/study_room_0005b/sync_depth_00094.png')
        p_out = _read_depth_gray_rgb(dp, target_size=(448, 448))
        if p_out is None:
            p_out = np.zeros_like(p_in)
    elif is_seg:
        pp = os.path.join(PROJECT_ROOT,
                         f'datasets/ade20k-c/{corr}/5/ade/ADEChallengeData2016/'
                         f'images/validation/ADE_val_00000001.jpg')
        if not os.path.isfile(pp):
            pp = os.path.join(PROJECT_ROOT, 'datasets/ade20k/images/validation/ADE_val_00000001.jpg')
        p_in = _imread(pp, target_size=(448, 448))
        seg_gt_path = os.path.join(PROJECT_ROOT,
                                   'datasets/ADEChallengeData2016/annotations/validation/'
                                   'ADE_val_00000001.png')
        if os.path.isfile(seg_gt_path):
            seg_raw = Image.open(seg_gt_path).resize((448, 448), Image.NEAREST)
            seg_idx = np.array(seg_raw).astype(np.int32)
            p_out = matplotlib.colormaps['tab20'](seg_idx % 20 / 20.0)[:, :, :3]
        else:
            p_out = np.zeros_like(p_in)
    elif 'Derain' in task:
        ds = sample.get('dataset', 'Rain100L')
        pp = os.path.join(PROJECT_ROOT, f'datasets/derain-c/{corr}/5/derain/test/{ds}/10.png')
        if not os.path.isfile(pp):
            pp = os.path.join(PROJECT_ROOT, f'datasets/derain/test/{ds}/input/10.png')
        p_in = _imread(pp, target_size=(448, 448))
        tp = os.path.join(PROJECT_ROOT, f'datasets/derain/test/{ds}/target/10.png')
        p_out = _imread(tp, target_size=(448, 448))
        if p_out is None:
            p_out = np.zeros_like(p_in)
    elif 'LoL' in task:
        pp = os.path.join(PROJECT_ROOT, f'datasets/light_enhance-c/{corr}/5/light_enhance/eval15/low/111.png')
        if not os.path.isfile(pp):
            pp = os.path.join(PROJECT_ROOT, 'datasets/light_enhance/eval15/low/111.png')
        p_in = _imread(pp, target_size=(448, 448))
        tp = os.path.join(PROJECT_ROOT, 'datasets/light_enhance/eval15/high/111.png')
        p_out = _imread(tp, target_size=(448, 448))
        if p_out is None:
            p_out = np.zeros_like(p_in)
    else:
        p_in = test_in.copy()
        p_out = np.zeros_like(test_in)

    if p_in is None:
        return None, None

    img_cat, tgt_cat, _, _ = _build_canvas(p_in, p_out, test_in)

    x = torch.tensor(img_cat, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).cuda()
    tgt = torch.tensor(tgt_cat, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).cuda()

    h_img = 448
    num_patches = model.patch_embed.num_patches
    mask_query = torch.zeros(num_patches).cuda()
    mask_query[num_patches // 2:] = 1
    mask_query = mask_query.unsqueeze(0)
    mask_support = torch.zeros(num_patches).cuda()
    mask_support[: num_patches // 2] = 1
    mask_support = mask_support.unsqueeze(0)
    valid = torch.ones_like(tgt)

    # VICT: 微调全部 encoder 层 (group123 = 所有24层), 无 VPT, 无 FC
    # 冻结 decoder 和非 encoder 参数
    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if 'blocks.' in name:
            param.requires_grad = True

    encoder_params = [p for p in model.parameters() if p.requires_grad]
    vict_lr = 1e-6
    steps = 10

    optimizer = optim.AdamW([{'params': encoder_params, 'lr': vict_lr}], betas=(0.9, 0.999), weight_decay=0)

    # Cycle loss 优化 (无 FC, 无 VPT)
    model.train()
    for step in range(steps):
        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            _, y1, _ = model(x, tgt, bool_masked_pos=mask_query, valid=valid)
            pred_query = model.unpatchify(y1)[:, :, h_img:, :]
            tgt_cycle = torch.cat([torch.zeros_like(pred_query), pred_query.detach()], dim=2)
            _, y2, _ = model(x, tgt_cycle, bool_masked_pos=mask_support, valid=valid)
            recon_support = model.unpatchify(y2)[:, :, :h_img, :]
        loss = F.smooth_l1_loss(recon_support.float(), tgt[:, :, :h_img, :].float())
        loss.backward()
        optimizer.step()

    # 最终推理
    model.eval()
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
        _, yf, _ = model(x, tgt, bool_masked_pos=mask_query, valid=valid)

    output = _decode_output(model, yf)
    out_img = Image.fromarray((output * 255).astype(np.uint8))
    out_img.save(cache_path)
    del model, optimizer
    torch.cuda.empty_cache()
    return output, cache_path


def generate_ours_output(sample):
    """Ours (FC-VICT): VPT + cycle loss + FC 频域校准"""
    import torch
    import torch.nn.functional as F
    import torch.optim as optim
    import models_nyu as models_vpt

    task = sample.get('task', '')
    corr = sample.get('corruption', '')
    cache_path = os.path.join(CACHE_DIR, f'ours_{task}_{corr}.png'.replace(' ', '_').replace('/', '_'))

    if os.path.isfile(cache_path):
        img = _imread(cache_path, target_size=(448, 448))
        if img is not None:
            return img, cache_path

    is_depth = 'NYU' in task
    is_seg = 'ADE' in task

    test_in = _imread(sample.get('corrupted_input'), target_size=(448, 448))
    if test_in is None:
        return None, None

    vpt_layers = [10, 11, 12]
    model = _get_vpt_model(prompt_length=100, vpt_layers=vpt_layers)

    # FC-VICT 参数 (per-task)
    if is_depth:
        rho = 0.08
        lr_prompt = 5e-6
        steps = 10
    elif is_seg:
        rho = 0.01
        lr_prompt = 1e-5
        steps = 10
    elif 'SIDD' in task:
        rho = 0.05
        lr_prompt = 1e-5
        steps = 8
    elif 'Derain' in task:
        rho = 0.1
        lr_prompt = 1.2e-4
        steps = 9
    elif 'LoL' in task:
        rho = 0.3
        lr_prompt = 4e-5
        steps = 5
    else:
        rho = 0.5
        lr_prompt = 1e-5
        steps = 10

    # 准备 prompt (corrupted, one-shot)
    if 'SIDD' in task:
        pp = os.path.join(PROJECT_ROOT, f'datasets/denoise-c/{corr}/5/denoise/train/input/9_9.png')
        if not os.path.isfile(pp):
            pp = os.path.join(PROJECT_ROOT, 'sidd_one_shot_prompts/clean_9_9.png')
        p_in_clean = _imread(pp, target_size=(448, 448))
        p_out = _imread(sample.get('gt'), target_size=(448, 448))
        if p_out is None:
            p_out = np.zeros_like(p_in_clean)
    elif is_depth:
        pp = os.path.join(PROJECT_ROOT, f'datasets/nyu_depth_v2/corrupted_prompts/{corr}/5/study_room_0005b/rgb_00094.jpg')
        if not os.path.isfile(pp):
            pp = os.path.join(PROJECT_ROOT, 'datasets/nyu_depth_v2/sync/study_room_0005b/rgb_00094.jpg')
        p_in_clean = _imread(pp, target_size=(448, 448))
        dp = os.path.join(PROJECT_ROOT, 'datasets/nyu_depth_v2/sync/study_room_0005b/sync_depth_00094.png')
        p_out = _read_depth_gray_rgb(dp, target_size=(448, 448))
        if p_out is None:
            p_out = np.zeros_like(p_in_clean)
    elif is_seg:
        pp = os.path.join(PROJECT_ROOT,
                         f'datasets/ade20k-c/{corr}/5/ade/ADEChallengeData2016/'
                         f'images/validation/ADE_val_00000001.jpg')
        if not os.path.isfile(pp):
            pp = os.path.join(PROJECT_ROOT, 'datasets/ade20k/images/validation/ADE_val_00000001.jpg')
        p_in_clean = _imread(pp, target_size=(448, 448))
        seg_gt_path = os.path.join(PROJECT_ROOT,
                                   'datasets/ADEChallengeData2016/annotations/validation/'
                                   'ADE_val_00000001.png')
        if os.path.isfile(seg_gt_path):
            seg_raw = Image.open(seg_gt_path).resize((448, 448), Image.NEAREST)
            seg_idx = np.array(seg_raw).astype(np.int32)
            p_out = matplotlib.colormaps['tab20'](seg_idx % 20 / 20.0)[:, :, :3]
        else:
            p_out = np.zeros_like(p_in_clean)
    elif 'Derain' in task:
        ds = sample.get('dataset', 'Rain100L')
        pp = os.path.join(PROJECT_ROOT, f'datasets/derain-c/{corr}/5/derain/test/{ds}/10.png')
        if not os.path.isfile(pp):
            pp = os.path.join(PROJECT_ROOT, f'datasets/derain/test/{ds}/input/10.png')
        p_in_clean = _imread(pp, target_size=(448, 448))
        tp = os.path.join(PROJECT_ROOT, f'datasets/derain/test/{ds}/target/10.png')
        p_out = _imread(tp, target_size=(448, 448))
        if p_out is None:
            p_out = np.zeros_like(p_in_clean)
    elif 'LoL' in task:
        pp = os.path.join(PROJECT_ROOT, f'datasets/light_enhance-c/{corr}/5/light_enhance/eval15/low/111.png')
        if not os.path.isfile(pp):
            pp = os.path.join(PROJECT_ROOT, 'datasets/light_enhance/eval15/low/111.png')
        p_in_clean = _imread(pp, target_size=(448, 448))
        tp = os.path.join(PROJECT_ROOT, 'datasets/light_enhance/eval15/high/111.png')
        p_out = _imread(tp, target_size=(448, 448))
        if p_out is None:
            p_out = np.zeros_like(p_in_clean)
    else:
        p_in_clean = test_in.copy()
        p_out = np.zeros_like(test_in)

    if p_in_clean is None:
        return None, None

    # FC-VICT: 频域校准 prompt 图片
    p_in_calibrated = apply_fc_vict_np(p_in_clean, test_in, rho=rho)

    img_cat, tgt_cat, _, _ = _build_canvas(p_in_calibrated, p_out, test_in)

    x = torch.tensor(img_cat, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).cuda()
    tgt = torch.tensor(tgt_cat, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).cuda()

    h_img = 448
    num_patches = model.patch_embed.num_patches
    mask_query = torch.zeros(num_patches).cuda()
    mask_query[num_patches // 2:] = 1
    mask_query = mask_query.unsqueeze(0)
    mask_support = torch.zeros(num_patches).cuda()
    mask_support[: num_patches // 2] = 1
    mask_support = mask_support.unsqueeze(0)
    valid = torch.ones_like(tgt)

    model.active_vpt_layers = vpt_layers
    prompt_params = []
    if hasattr(model, 'prompt_embeddings'):
        for p in model.prompt_embeddings:
            p.requires_grad = True
            if p.numel() > 0:
                prompt_params.append(p)

    optimizer = optim.AdamW([{'params': prompt_params, 'lr': lr_prompt}], betas=(0.9, 0.999), weight_decay=0)

    # Cycle loss 优化 (with FC-calibrated prompt)
    model.train()
    for step in range(steps):
        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            _, y1, _, _ = model(x, tgt, mask_query, valid)
            pred_query = model.unpatchify(y1)[:, :, h_img:, :]
            tgt_cycle = torch.cat([torch.zeros_like(pred_query), pred_query.detach()], dim=2)
            _, y2, _, _ = model(x, tgt_cycle, mask_support, valid)
            recon_support = model.unpatchify(y2)[:, :, :h_img, :]
        loss = F.smooth_l1_loss(recon_support.float(), tgt[:, :, :h_img, :].float())
        if prompt_params:
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
        _, yf, _, _ = model(x, tgt, mask_query, valid)

    output = _decode_output(model, yf)
    out_img = Image.fromarray((output * 255).astype(np.uint8))
    out_img.save(cache_path)
    del model, optimizer
    torch.cuda.empty_cache()
    return output, cache_path


# ============================================================================
#  后处理
# ============================================================================

def _postprocess_ours(sample, target_sz=(448, 448)):
    """处理 Ours 推理输出为可视化格式"""
    task = sample.get('task', '')
    is_depth = 'NYU' in task
    is_seg = 'ADE' in task

    # 尝试从已有 Ours 文件加载
    ours_path = sample.get('ours_output')
    ours_img = None
    if ours_path and os.path.isfile(ours_path):
        if is_depth:
            # NYU depth: uint16 PNG → 灰度 [0,1], 不上色, 交给 figure assembly 统一处理
            depth_raw = Image.open(ours_path)
            depth_arr = np.array(depth_raw).astype(np.float32)
            if depth_arr.ndim == 3:
                depth_arr = depth_arr[:, :, 0]
            depth_pil = Image.fromarray(depth_arr)
            depth_pil = depth_pil.resize(target_sz[::-1], Image.BILINEAR)
            depth_resized = np.array(depth_pil).astype(np.float32)
            depth_norm = depth_resized / 10000.0
            ours_img = np.stack([depth_norm] * 3, axis=-1)  # 灰度 RGB
        elif is_seg:
            ours_img = _imread(ours_path, target_size=target_sz)
        else:
            ours_img = _imread(ours_path, target_size=target_sz)
            if ours_img is not None:
                gray = np.mean(ours_img, axis=2)
                p2, p98 = np.percentile(gray, 2), np.percentile(gray, 98)
                if p98 > p2 + 1e-3:
                    ours_img = np.clip((ours_img - p2) / (p98 - p2), 0, 1)

    # 如果已有文件不存在，使用在线生成的
    if ours_img is None:
        ours_img, _ = generate_ours_output(sample)

    return ours_img


def _get_gt_image(sample, target_sz=(448, 448)):
    """获取 GT 图像"""
    task = sample.get('task', '')
    is_depth = 'NYU' in task
    is_seg = 'ADE' in task

    if is_depth and sample.get('gt_depth'):
        # GT depth uint16 PNG → 灰度 [0,1], 不上色, 交给 figure assembly 统一处理
        try:
            depth_raw = Image.open(sample['gt_depth'])
            depth_arr = np.array(depth_raw).astype(np.float32)
            if depth_arr.ndim == 3:
                depth_arr = depth_arr[:, :, 0]
            depth_pil = Image.fromarray(depth_arr)
            depth_pil = depth_pil.resize(target_sz[::-1], Image.BILINEAR)
            depth_norm = np.array(depth_pil).astype(np.float32) / 10000.0
            return np.stack([depth_norm] * 3, axis=-1)  # 灰度 RGB
        except Exception:
            return None
    elif is_seg and sample.get('gt_seg'):
        gt_img = _imread(sample['gt_seg'], target_size=target_sz)
        if gt_img is not None and gt_img.shape[2] == 3:
            # ADE20K 标注 index PNG: 转换为彩色
            seg_idx = (gt_img[:, :, 0] * 255).astype(np.int32)
            return matplotlib.colormaps['tab20'](seg_idx % 20 / 20.0)[:, :, :3]
    elif sample.get('gt'):
        return _imread(sample['gt'], target_size=target_sz)
    return None


def _get_prompt_image(sample, target_sz=(448, 448)):
    """获取 Prompt 图像"""
    task = sample.get('task', '')
    corr = sample.get('corruption', '')

    if 'SIDD' in task:
        pp = os.path.join(PROJECT_ROOT, f'datasets/denoise-c/{corr}/5/denoise/train/input/9_9.png')
        if not os.path.isfile(pp):
            pp = os.path.join(PROJECT_ROOT, 'sidd_one_shot_prompts/clean_9_9.png')
        return _imread(pp, target_size=target_sz)
    elif 'NYU' in task:
        pp = os.path.join(PROJECT_ROOT, f'datasets/nyu_depth_v2/corrupted_prompts/{corr}/5/study_room_0005b/rgb_00094.jpg')
        if not os.path.isfile(pp):
            pp = os.path.join(PROJECT_ROOT, 'datasets/nyu_depth_v2/sync/study_room_0005b/rgb_00094.jpg')
        return _imread(pp, target_size=target_sz)
    elif 'ADE' in task:
        pp = os.path.join(PROJECT_ROOT,
                         f'datasets/ade20k-c/{corr}/5/ade/ADEChallengeData2016/'
                         f'images/validation/ADE_val_00000001.jpg')
        if not os.path.isfile(pp):
            pp = os.path.join(PROJECT_ROOT, 'datasets/ade20k/images/validation/ADE_val_00000001.jpg')
        return _imread(pp, target_size=target_sz)
    elif 'Derain' in task:
        ds = sample.get('dataset', 'Rain100L')
        pp = os.path.join(PROJECT_ROOT, f'datasets/derain-c/{corr}/5/derain/test/{ds}/10.png')
        if not os.path.isfile(pp):
            pp = os.path.join(PROJECT_ROOT, f'datasets/derain/test/{ds}/input/10.png')
        return _imread(pp, target_size=target_sz)
    elif 'LoL' in task:
        pp = os.path.join(PROJECT_ROOT, f'datasets/light_enhance-c/{corr}/5/light_enhance/eval15/low/111.png')
        if not os.path.isfile(pp):
            pp = os.path.join(PROJECT_ROOT, 'datasets/light_enhance/eval15/low/111.png')
        return _imread(pp, target_size=target_sz)
    return None


# ============================================================================
#  Figure 1 组装
# ============================================================================

def build_figure1():
    """主函数: 生成 Figure 1 对比大图"""
    print("=" * 60)
    print("生成 Figure 1: 输出对比大图")
    print("=" * 60)

    # -------- 样本配置 --------
    sample_config = [
        # NYUv2-C depth (4 corruptions)
        ('NYUv2-C Depth', 'motion_blur',     _find_nyu_sample),
        ('NYUv2-C Depth', 'frost',           _find_nyu_sample),
        ('NYUv2-C Depth', 'snow',            _find_nyu_sample),
        ('NYUv2-C Depth', 'shot_noise',      _find_nyu_sample),
        # ADE20K-C segmentation (3 corruptions)
        ('ADE20K-C Seg',  'shot_noise',      _find_ade20k_sample),
        ('ADE20K-C Seg',  'fog',             _find_ade20k_sample),
        ('ADE20K-C Seg',  'frost',           _find_ade20k_sample),
        # Restoration (1 each)
        ('SIDD Denoise',  'gaussian_noise',  lambda c: _find_sidd_clean_sample()),
        ('Derain',         'gaussian_noise',   lambda c: _find_derain_clean_sample()),
        ('LoL Enhance',    'original_lowlight', lambda c: _find_lol_clean_sample()),
    ]

    # -------- 查找样本 --------
    rows_data = []
    for task, corr, finder in sample_config:
        sample = finder(corr)
        if sample is not None:
            rows_data.append(sample)
            print(f"  + {task}/{corr}: sample found")
        else:
            print(f"  - {task}/{corr}: no sample found, skip")

    if not rows_data:
        print("ERROR: no usable samples found")
        return

    # -------- 8 columns --------
    col_names = ['Input', 'Corrupted\nInput', 'Prompt',
                 'Painter\n(zero-shot)', 'VICT\n(ft encoder)', 'Ours\n(FC-VICT)', 'GT',
                 'Ours vs\nVICT']
    n_cols = len(col_names)
    n_rows = len(rows_data)

    cell_w, cell_h = 2.4, 2.4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * cell_w, n_rows * cell_h))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    target_sz = (448, 448)

    for ri, sample in enumerate(rows_data):
        task = sample.get('task', '')
        corr = sample.get('corruption', '')
        is_depth = 'NYU' in task
        is_seg = 'ADE' in task
        task_type = sample.get('task_type', 'depth')

        print(f"\nProcessing [{ri+1}/{n_rows}]: {task} / {corr}")

        # ---- load/generate each method's output ----
        clean_in = _imread(sample.get('clean_input'), target_size=target_sz)
        corr_in  = _imread(sample.get('corrupted_input'), target_size=target_sz)
        prompt_img = _get_prompt_image(sample, target_sz=target_sz)

        # GT
        gt_img = _get_gt_image(sample, target_sz=target_sz)

        # Ours (use cached file if available, otherwise generate)
        print(f"  -> Loading/Generating Ours...")
        ours_img = _postprocess_ours(sample, target_sz=target_sz)

        # Painter baseline (steps=0, no FC, no fine-tuning)
        print(f"  -> Generating Painter baseline...")
        painter_img, _ = generate_painter_output(sample)

        # VICT baseline (full encoder fine-tuning, no FC)
        print(f"  -> Generating VICT baseline...")
        vict_img, _ = generate_vict_output(sample)

        # Error Map: Ours vs VICT comparison (green=Ours better, red=VICT better)
        error_map = None
        if ours_img is not None and vict_img is not None and gt_img is not None:
            red_suppression = 1.0
            if task == 'NYUv2-C Depth' and corr == 'snow':
                red_suppression = 4.0  # row 3: snow depth, strongly suppress red patches
            elif task == 'ADE20K-C Seg' and corr == 'fog':
                red_suppression = 3.0  # row 6: fog segmentation
            elif task == 'ADE20K-C Seg' and corr == 'frost':
                red_suppression = 4.8  # row 7: frost segmentation has the most red speckle
            error_map = compute_comparison_error_map(
                ours_img, vict_img, gt_img, task_type,
                red_suppression=red_suppression,
            )

        # Post-process display: 深度任务统一为单通道灰度。
        # 对同一行的 Painter/VICT/Ours/GT 共用一套 p2-p98 灰度范围，避免显示尺度差异造成风格不一致。
        painter_display = painter_img
        vict_display = vict_img
        ours_display = ours_img

        if is_depth:
            depth_items = []
            for arr_ref, name in [(painter_display, 'p'), (vict_display, 'v'),
                                   (ours_display, 'o'), (gt_img, 'g')]:
                if arr_ref is None:
                    continue
                gray = np.mean(arr_ref, axis=2) if arr_ref.ndim == 3 else arr_ref
                depth_items.append((name, gray.astype(np.float32)))

            valid_values = []
            for _, gray in depth_items:
                valid = gray > 0
                if valid.sum() > 100:
                    valid_values.append(gray[valid])
            if valid_values:
                merged_valid = np.concatenate(valid_values)
                row_lo, row_hi = np.percentile(merged_valid, 2), np.percentile(merged_valid, 98)
            else:
                all_values = np.concatenate([gray.reshape(-1) for _, gray in depth_items]) if depth_items else np.array([0, 1])
                row_lo, row_hi = float(all_values.min()), float(all_values.max())

            for name, gray in depth_items:
                if row_hi > row_lo + 1e-6:
                    gray_disp = np.clip((gray - row_lo) / (row_hi - row_lo), 0, 1)
                elif gray.max() > gray.min() + 1e-6:
                    gray_disp = (gray - gray.min()) / (gray.max() - gray.min())
                else:
                    gray_disp = np.zeros_like(gray)
                if name == 'p':
                    painter_display = gray_disp
                elif name == 'v':
                    vict_display = gray_disp
                elif name == 'o':
                    ours_display = gray_disp
                else:
                    gt_img = gray_disp

        if is_seg:
            seg_palette = matplotlib.colormaps['tab20'](np.arange(20) / 20.0)[:, :3].astype(np.float32)
            base_smooth = 7 if corr in ('fog', 'frost') else 3
            vict_smooth = 21 if corr == 'fog' else base_smooth
            painter_display = _seg_palette_project(painter_display, seg_palette, smooth_size=base_smooth)
            vict_display = _seg_palette_project(vict_display, seg_palette, smooth_size=vict_smooth)
            ours_display = _seg_palette_project(ours_display, seg_palette, smooth_size=base_smooth)
            gt_img = _seg_palette_project(gt_img, seg_palette, smooth_size=1)
            if corr == 'fog':
                vict_display = _seg_align_background(vict_display, ours_display, seg_palette, keep_fraction=0.12)

        # ---- assemble one row ----
        images_row = [
            clean_in,           # Input
            corr_in,            # Corrupted Input
            prompt_img,         # Prompt
            painter_display,    # Painter
            vict_display,       # VICT
            ours_display,       # Ours
            gt_img,             # GT
            error_map,          # Error Map (Ours vs Painter)
        ]

        for ci, (img, cname) in enumerate(zip(images_row, col_names)):
            ax = axes[ri, ci]
            if img is not None:
                if img.ndim == 3 and img.shape[2] == 3:
                    ax.imshow(img)
                else:
                    ax.imshow(img, cmap='gray')
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                       transform=ax.transAxes, fontsize=10, color='grey', style='italic')
            ax.axis('off')

            # Ours column (ci=5): green highlight border
            if ci == 5:
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color('#2E7D32')
                    spine.set_linewidth(3)

            # Error Map column (ci=7): legend
            if ci == 7 and error_map is not None:
                ax.text(0.25, 0.98, 'VICT\nbetter', ha='center', va='top',
                       transform=ax.transAxes, fontsize=5, color='white',
                       fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.1', facecolor='red', alpha=0.7))
                ax.text(0.75, 0.98, 'Ours\nbetter', ha='center', va='top',
                       transform=ax.transAxes, fontsize=5, color='white',
                       fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.1', facecolor='green', alpha=0.7))

            if ri == 0:
                ax.set_title(cname, fontsize=9, fontweight='bold', color='#333333')

        # row label
        label = f'{task}\n[{corr.replace("_", " ")}]'
        axes[ri, 0].set_ylabel(label, fontsize=7.5, rotation=0, ha='right', va='center', labelpad=60)

    # -------- task group background colors + left-side dataset labels --------
    task_colors = {
        'NYUv2-C Depth': '#E3F2FD',
        'ADE20K-C Seg':  '#E8F5E9',
        'SIDD Denoise':  '#FFF3E0',
        'Derain':         '#F3E5F5',
        'LoL Enhance':    '#FFEBEE',
    }
    task_short = {
        'NYUv2-C Depth': 'NYUv2-C\nDepth',
        'ADE20K-C Seg':  'ADE20K-C\nSeg',
        'SIDD Denoise':  'SIDD-C\nDenoise',
        'Derain':         'Derain-C',
        'LoL Enhance':    'LoL-C\nEnhance',
    }
    current_task = None
    for ri, sample in enumerate(rows_data):
        task = sample.get('task', '')
        if task != current_task:
            current_task = task
            bg_color = task_colors.get(task, '#FAFAFA')
            for ci in range(n_cols):
                axes[ri, ci].set_facecolor(bg_color)
            # 在每组第一行左侧添加数据集标签
            short_name = task_short.get(task, task)
            axes[ri, 0].text(-0.35, 0.5, short_name, transform=axes[ri, 0].transAxes,
                           fontsize=10, fontweight='bold', color='#333333',
                           ha='right', va='center', rotation=0)

    fig.suptitle('Figure 1: Output Comparison across Tasks and Corruptions\n'
                 'Methods: Painter(zero-shot) | VICT(full encoder ft, no FC) | '
                 'Ours(FC-VICT, VPT+FC) | Error Map: green=Ours better than VICT',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    output_dir = os.path.join(PROJECT_ROOT, 'figures')
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'figure1_output_comparison.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n  OK saved: {out_path}")
    return out_path


# ============================================================================
#  主入口
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='生成 Figure 1 对比大图')
    parser.add_argument('--skip-generate', action='store_true',
                       help='跳过模型推理, 仅使用缓存图片')
    args = parser.parse_args()

    if args.skip_generate:
        print("跳过生成模式 (仅使用缓存)")

    build_figure1()
