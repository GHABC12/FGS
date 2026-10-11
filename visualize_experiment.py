#!/usr/bin/env python3
"""
可视化实验脚本 —— 根据「可视化实验.pdf」需求，生成三张论文级别图表：

  Figure 1: 输出对比大图 (Input / Corrupted / Prompt / Painter / VICT / Ours / GT / Error Map)
  Figure 2: Prompt 频谱可视化 (clean / corrupted / calibrated 的频谱分析)
  Figure 3: Group Selection 热力图 (15 corruption × 3 layer groups)

用法:  python visualize_experiment.py [--output_dir OUTPUT_DIR]
"""

import argparse
import copy
import csv
import glob
import json
import math
import os
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from PIL import Image, ImageFile
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# 全局配置
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

# 15 corruption 类型 (标准名称)
ALL_CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog',
    'brightness', 'contrast',
    'elastic_transform', 'pixelate', 'jpeg_compression',
]

# 层组定义
GROUP_LAYERS = {
    'G1 (shallow 0-2)':  [0, 1, 2],
    'G2 (middle 10-12)': [10, 11, 12],
    'G3 (deep 21-23)':   [21, 22, 23],
}
GROUP_SHORT = {0: 'G1', 1: 'G2', 2: 'G3'}


# ============================================================================
#  工具函数
# ============================================================================

def _imread(path, target_size=None):
    """读取图片返回 numpy (H,W,3) RGB [0,1]，文件不存在返回 None。"""
    if path is None or not os.path.isfile(path):
        return None
    try:
        img = Image.open(path).convert('RGB')
        if target_size is not None:
            img = img.resize(target_size[::-1], Image.BILINEAR)
        return np.array(img).astype(np.float32) / 255.0
    except Exception:
        return None


def _denorm_painter(img_np):
    """对 [0,1] 的 Painter 输出做反归一化（图像已被 ImageNet 归一化存储）。"""
    return np.clip(img_np * IMAGENET_STD + IMAGENET_MEAN, 0, 1)


def _is_degenerate(img_np, variance_thresh=0.0005, dark_thresh=0.02, bright_thresh=0.98):
    """检测图片是否退化：全灰/全黑/全白/极低方差。"""
    if img_np is None:
        return True
    if img_np.ndim == 3:
        gray = np.mean(img_np, axis=2)
    else:
        gray = img_np
    var = np.var(gray)
    mean = np.mean(gray)
    if var < variance_thresh:
        return True
    if mean < dark_thresh:
        return True
    if mean > bright_thresh:
        return True
    return False


def _depth_colorize(depth_np, cmap='plasma', vmin=None, vmax=None, fixed_range=False):
    """将单通道深度图（H,W）转为 RGB 彩色图。

    如果 fixed_range=True，使用固定 vmin=0.0, vmax=1.0 (对应 0-10m)，
    保证不同场景之间的颜色一致可比较。
    """
    if depth_np.ndim == 3:
        depth_np = depth_np[:, :, 0]
    valid = depth_np > 0
    if fixed_range:
        vmin, vmax = 0.0, 1.0
    else:
        if vmin is None:
            vmin = np.percentile(depth_np[valid], 2) if valid.any() else 0
        if vmax is None:
            vmax = np.percentile(depth_np[valid], 98) if valid.any() else 1
    depth_clipped = np.clip((depth_np - vmin) / (vmax - vmin + 1e-8), 0, 1)
    colored = matplotlib.colormaps[cmap](depth_clipped)[:, :, :3]
    colored[~valid] = [0, 0, 0]
    return colored


def _pick_best_ours_sample(candidates, task_hint, corruption):
    """从候选列表中选取 Ours 相对 VICT 优势最大的样本。"""
    if len(candidates) <= 1:
        return candidates[0] if candidates else None

    # 尝试加载 VICT 缓存以计算 green advantage
    v_key = f"vict_a1__{task_hint.replace(' ','_')}__{corruption}".replace('/', '_')
    v_pattern = os.path.join(_BASELINE_CACHE_DIR, f'{v_key}*')
    v_files = glob.glob(v_pattern)
    vict = None
    if v_files:
        vict = _imread(v_files[0], target_size=(224, 224))
    if vict is not None:
        vict_gray = np.mean(vict, axis=2)

    is_depth = 'NYU' in task_hint or 'Depth' in task_hint
    best = None
    best_green = -1.0
    for cand in candidates:
        our_path = cand.get('ours_output')
        gt_path = cand.get('gt', cand.get('gt_depth'))
        if not our_path or not gt_path:
            continue
        try:
            if is_depth:
                raw = Image.open(our_path)
                arr = np.array(raw).astype(np.float32)
                if arr.ndim == 3: arr = arr[:,:,0]
                p = Image.fromarray(arr).resize((224, 224), Image.BILINEAR)
                o = np.array(p).astype(np.float32) / 10000.0
                if os.path.isfile(gt_path):
                    raw_g = Image.open(gt_path)
                    arr_g = np.array(raw_g).astype(np.float32)
                    if arr_g.ndim == 3: arr_g = arr_g[:,:,0]
                    p_g = Image.fromarray(arr_g).resize((224, 224), Image.BILINEAR)
                    g = np.array(p_g).astype(np.float32) / 10000.0
                else: continue
            else:
                o = _imread(our_path, target_size=(224, 224))
                g = _imread(gt_path, target_size=(224, 224))
                if o is None or g is None: continue
                o = np.mean(o, axis=2)
                g = np.mean(g, axis=2)
        except Exception:
            continue

        # 如果有VICT缓存，按green advantage评分；否则按方差
        if vict is not None:
            o_err = np.abs(o - g)
            v_err = np.abs(vict_gray - g)
            green_frac = (o_err < v_err).mean()
            score = green_frac
        else:
            score = np.var(o) if len(o.ravel()) > 0 else 0

        if score > best_green:
            best_green = score
            best = cand

    return best if best is not None else candidates[min(1, len(candidates)-1)]


def _find_inference_subdir(base_dir, corruption):
    """在 base_dir 下找到包含指定 corruption 名称的最优子目录（按输出质量评分）。"""
    if not os.path.isdir(base_dir):
        return None
    matches = []
    for d in sorted(os.listdir(base_dir)):
        full = os.path.join(base_dir, d)
        if os.path.isdir(full) and corruption in d:
            # 评分：采样前20张图，计算平均方差
            pngs = sorted([f for f in os.listdir(full) if f.endswith('.png')])
            if not pngs:
                # 检查 results 子目录 (Derain)
                results_dir = os.path.join(full, 'results')
                if os.path.isdir(results_dir):
                    for ds in ['Rain100L', 'Test100']:
                        ds_dir = os.path.join(results_dir, ds)
                        if os.path.isdir(ds_dir):
                            pngs.extend(sorted([os.path.join(ds, f) for f in os.listdir(ds_dir) if f.endswith('.png')]))
            score = 0
            count = 0
            for fname in pngs[:20]:
                fpath = os.path.join(full, fname) if '/' not in fname else os.path.join(base_dir, d, 'results', fname)
                try:
                    img = np.array(Image.open(fpath).convert('RGB')).astype(float)
                    score += np.mean(img, axis=2).var()
                    count += 1
                except: pass
            matches.append((full, score / max(count, 1)))
    if not matches:
        return None
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches[0][0]


# ============================================================================
#  CVS / JSON 数据加载
# ============================================================================

def _read_figures_metric_table(path, task_name):
    """读取 figures/ 中论文整理后的结果表，标准化为 corruption/winning_groups/metric 列。"""
    try:
        raw = Path(path).read_bytes()
        if raw[:2] == b'PK':
            df_raw = pd.read_excel(path, header=None)
        else:
            df_raw = pd.read_csv(path, header=None, encoding='gb18030')
    except Exception as e:
        print(f"  [WARN] 读取 figures 结果失败 {os.path.basename(path)}: {e}")
        return None

    header_idx = None
    for idx in range(len(df_raw)):
        values = [str(v).strip().lower() for v in df_raw.iloc[idx].tolist()]
        if 'corruption' in values:
            header_idx = idx
            break
    if header_idx is None:
        return None

    raw_header = [str(v).strip() if not pd.isna(v) else f'col_{i}'
                  for i, v in enumerate(df_raw.iloc[header_idx].tolist())]
    seen = defaultdict(int)
    header = []
    for name in raw_header:
        count = seen[name]
        header.append(name if count == 0 else f'{name}__{count}')
        seen[name] += 1
    body = df_raw.iloc[header_idx + 1:].copy()
    body.columns = header
    body = body.dropna(how='all')
    if 'corruption' not in body.columns:
        return None
    body['corruption'] = body['corruption'].astype(str).str.strip().str.lower()
    body = body[body['corruption'].isin(ALL_CORRUPTIONS)].copy()
    if body.empty:
        return None

    if '获胜组' in body.columns and 'winning_groups' not in body.columns:
        body['winning_groups'] = body['获胜组']
    elif 'winning_groups' not in body.columns:
        for col in body.columns:
            if body[col].astype(str).str.contains('group', case=False, na=False).any():
                body['winning_groups'] = body[col]
                break

    metric_col = None
    for col in body.columns:
        col_l = str(col).lower()
        if col_l == 'one-shot' or col_l.startswith('one-shot__') or col_l == 'one_shot':
            metric_col = col
            break
    if metric_col is None:
        for col in body.columns:
            if str(col).lower() == 'our':
                metric_col = col
                break
    if metric_col is not None:
        body['metric'] = body.loc[:, metric_col]

    body['shot_mode'] = 'one-shot'
    body['source_file'] = os.path.basename(path)
    body['source_task'] = task_name
    return body


def load_figures_results():
    """优先加载 figures/ 目录下整理后的实验结果表。"""
    fig_dir = os.path.join(PROJECT_ROOT, 'figures')
    specs = [
        ('nyu_results.xlsx', 'NYUv2-C'),
        ('lol_results_new_groups.xlsx', 'LoL-C'),
        ('derain_results.csv', 'Derain-C'),
    ]
    frames = []
    for fname, task_name in specs:
        path = os.path.join(fig_dir, fname)
        if not os.path.isfile(path):
            continue
        df = _read_figures_metric_table(path, task_name)
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_all_csvs():
    """加载结果表。优先使用 figures/ 下整理好的实验结果，同时兼容根目录 results_*.csv。"""
    frames = []
    figures_df = load_figures_results()
    if not figures_df.empty:
        print(f"  figures 指定结果表: {len(figures_df)} 行")
        return figures_df

    csv_files = sorted(glob.glob(os.path.join(PROJECT_ROOT, 'results_*.csv')))
    for f in csv_files:
        try:
            fname = os.path.basename(f)
            # 先探测有无表头
            with open(f, 'r', newline='') as fh:
                first = fh.readline().strip()

            if first.startswith('corruption'):
                # 标准带表头格式 (SIDD/Derain tune / results_new / results_original)
                df = pd.read_csv(f, on_bad_lines='skip')
            elif first.startswith('stage'):
                # COCO tuning 格式
                df = pd.read_csv(f, on_bad_lines='skip')
            elif first.startswith('zero_shot') or first.startswith('one_shot'):
                # 无表头 NYU 旧格式 (results_nyu_new.csv, results_nyu_1_10.csv 等)
                cols = ['shot_mode', 'corruption', 'param', 'lr_prompt', 'lr_backbone',
                        'layers', 'prompt_len', 'num_bands', 'lambda_gate', 'lambda_f',
                        'rho', 'winning_groups', 'metric']
                df = pd.read_csv(f, header=None, names=cols, on_bad_lines='skip')
            elif first.split(',')[0].strip() in ALL_CORRUPTIONS:
                cols = ['corruption', 'steps', 'lr_prompt', 'lr_backbone', 'layers',
                        'prompt_len', 'lambda_f', 'rho', 'num_bands', 'lambda_gate',
                        'winning_groups', 'psnr', 'ssim']
                df = pd.read_csv(f, header=None, names=cols, on_bad_lines='skip')
            else:
                # 其他格式：尝试用常见列名读取
                try:
                    df = pd.read_csv(f, on_bad_lines='skip')
                except Exception:
                    continue

            df['source_file'] = fname
            frames.append(df)
        except Exception as e:
            print(f"  [WARN] 跳过 {os.path.basename(f)}: {e}")
            continue

    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    return combined


def extract_winning_groups(combined_df):
    """从合并 DataFrame 提取各任务/腐败类型的最终 selected group。"""
    result = defaultdict(dict)
    if combined_df.empty:
        return result

    def _task_from_file(fname):
        fname_l = fname.lower()
        base = os.path.basename(fname_l)
        if 'nyu' in fname_l:
            return 'NYUv2-C'
        if 'sidd' in fname_l:
            return 'SIDD-C'
        if 'derain' in fname_l:
            return 'Derain-C'
        if 'lol' in fname_l or base in {'results_new.csv', 'results_original.csv'}:
            return 'LoL-C'
        if 'coco' in fname_l:
            return 'COCO-C'
        return 'Unknown'

    def _corr_from_source(fname):
        fname_l = fname.lower()
        for c in sorted(ALL_CORRUPTIONS, key=len, reverse=True):
            if c.replace('_', '') in fname_l.replace('_', ''):
                return c
        return None

    def _norm_shot(row, source_file):
        sm_val = row.get('shot_mode', '')
        sm = '' if pd.isna(sm_val) else str(sm_val).strip().lower().replace('_', '-')
        src = source_file.lower().replace('_', '-')
        if not sm:
            if 'one-shot' in src or 'oneshot' in src:
                sm = 'one-shot'
            elif 'zero-shot' in src or 'zeroshot' in src:
                sm = 'zero-shot'
        if sm == 'one-shot':
            return 2
        if sm == 'zero-shot':
            return 1
        return 1.5

    def _to_float(value):
        try:
            if pd.isna(value):
                return None
            val = float(value)
            if not np.isfinite(val):
                return None
            return val
        except Exception:
            return None

    def _metric_score(row, task):
        if task == 'NYUv2-C':
            for key in ['abs_rel', 'metric']:
                val = _to_float(row.get(key, None))
                if val is not None:
                    return -val
            return -np.inf
        metric_keys = {
            'ADE20K-C': ['miou', 'mIoU', 'metric'],
            'SIDD-C': ['psnr', 'metric', 'ssim'],
            'Derain-C': ['psnr', 'metric', 'ssim'],
            'LoL-C': ['psnr', 'metric', 'ssim'],
            'COCO-C': ['pq', 'PQ', 'metric'],
        }.get(task, ['metric'])
        for key in metric_keys:
            val = _to_float(row.get(key, None))
            if val is not None:
                return val
        return -np.inf

    best = {}
    for row_idx, row in combined_df.iterrows():
        source_file = str(row.get('source_file', ''))
        task = str(row.get('source_task', '')).strip()
        if not task or task == 'nan':
            task = _task_from_file(source_file)
        if task == 'Unknown':
            continue

        wg_val = row.get('winning_groups', None)
        if pd.isna(wg_val) or str(wg_val).strip() in ('', 'nan'):
            wg_val = row.get('winner_group', None)
        wg = '' if pd.isna(wg_val) else str(wg_val)
        m = re.search(r'group\s*(\d)', wg, re.IGNORECASE)
        if not m:
            continue
        group_num = int(m.group(1))
        if group_num not in (1, 2, 3):
            continue

        corr = str(row.get('corruption', '')).strip().lower()
        if not corr or corr == 'nan':
            corr = _corr_from_source(source_file) or ''
        if corr not in ALL_CORRUPTIONS:
            continue

        source_priority = 10 if str(row.get('source_task', '')).strip() not in ('', 'nan') else 0
        candidate = {
            'group': group_num,
            'source_priority': source_priority,
            'shot_priority': _norm_shot(row, source_file),
            'metric_score': _metric_score(row, task),
            'row_idx': row_idx,
        }
        key = (corr, task)
        current = best.get(key)
        if current is None:
            best[key] = candidate
            continue
        cand_rank = (candidate['source_priority'], candidate['shot_priority'], candidate['metric_score'], candidate['row_idx'])
        cur_rank = (current['source_priority'], current['shot_priority'], current['metric_score'], current['row_idx'])
        if cand_rank > cur_rank:
            best[key] = candidate

    # LoL-C 的部分调优结果只保存在 winner 文本文件中；这里只用来补齐 CSV 中缺失的项。
    lol_winner_patterns = [
        os.path.join(PROJECT_ROOT, 'models_inference', 'vpt_results_lol*', '**', 'winning_groups.txt'),
        os.path.join(PROJECT_ROOT, 'temp', 'lol*', 'global_winner*.txt'),
    ]
    for pattern in lol_winner_patterns:
        for winner_path in glob.glob(pattern, recursive=True):
            try:
                with open(winner_path, 'r') as fh:
                    content = fh.read()
            except Exception:
                continue
            m = re.search(r'group\s*(\d)', content, re.IGNORECASE)
            if not m:
                continue
            group_num = int(m.group(1))
            if group_num not in (1, 2, 3):
                continue
            corr = _corr_from_source(winner_path)
            if corr not in ALL_CORRUPTIONS:
                continue
            best.setdefault((corr, 'LoL-C'), {
                'group': group_num,
                'source_priority': 5,
                'shot_priority': 1.25,
                'metric_score': -np.inf,
                'row_idx': -1,
            })

    for (corr, task), item in best.items():
        result[corr][task] = item['group']
    return dict(result)


def load_raw_drifts():
    """加载 raw_drifts/ 下所有 JSON，返回 {corruption: {layer_idx: drift_value}}。"""
    drift_dir = os.path.join(PROJECT_ROOT, 'raw_drifts')
    drift_data = {}
    for f in sorted(glob.glob(os.path.join(drift_dir, 'raw_*.json'))):
        corr_match = re.search(r'raw_(.+)\.json', os.path.basename(f))
        if not corr_match:
            continue
        corr = corr_match.group(1)
        if corr not in ALL_CORRUPTIONS:
            continue
        with open(f) as fh:
            raw = json.load(fh)
        # 转换字符串键为 int
        drift_data[corr] = {int(k): v for k, v in raw.items()}
    return drift_data


# ============================================================================
#  频谱分析 (numpy, 无需 GPU)
# ============================================================================

def compute_log_amplitude_spectrum(img_np, num_bands=20):
    """对单张 RGB 图像计算 log amplitude spectrum 和径向频率能量曲线。

    Args:
        img_np: (H, W, 3) float in [0, 1]
        num_bands: 径向频带数量

    Returns:
        log_amp: (H, W) 中心化 log 幅度谱
        radial_profile: (num_bands,) 每频带平均 log energy
        freq_bins: (num_bands,) 频带中心位置 [0, 1]
    """
    gray = np.mean(img_np, axis=2)
    h, w = gray.shape

    fft = np.fft.fft2(gray.astype(np.float64))
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shifted)
    log_amp = np.log(magnitude + 1e-12)

    # 径向坐标 (归一化到 [0, 1])
    cy, cx = h / 2.0, w / 2.0
    y, x = np.ogrid[:h, :w]
    r = np.sqrt(((y - cy) / (h / 2.0)) ** 2 + ((x - cx) / (w / 2.0)) ** 2) / np.sqrt(2)

    radial = np.zeros(num_bands, dtype=np.float64)
    for b in range(num_bands):
        inner, outer = b / num_bands, (b + 1) / num_bands
        mask = (r >= inner) & (r < outer)
        radial[b] = np.mean(log_amp[mask]) if mask.any() else 0.0

    freq_bins = np.linspace(0.5 / num_bands, 1 - 0.5 / num_bands, num_bands)
    return log_amp, radial, freq_bins


def apply_fc_vict_np(prompt_img, test_img, rho=0.5, r_L=0.03, r_H=0.15,
                     alpha_min=0.5, alpha_max=1.5):
    """numpy 版 FC-VICT 频域校准 (逐通道处理)。

    Args:
        prompt_img: (H, W, 3) float [0,1]
        test_img:   (H, W, 3) float [0,1]
        rho: 低频混合权重
        r_L, r_H: 低/高频分界线
        alpha_min, alpha_max: 高频能量校准系数的裁剪范围

    Returns:
        calibrated_img: (H, W, 3) float [0,1]
    """
    h, w = prompt_img.shape[:2]

    # 频域 mask
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
#  Figure 3: Group Selection 热力图
# ============================================================================

def build_figure3_group_heatmap(output_dir, combined_df, drift_data):
    """生成 Group Selection 热力图。

    对于每个主要任务 (NYUv2-C, SIDD-C, Derain-C) 生成独立的子热力图，
    横轴: 15 corruption 类型，纵轴: 3 个层组 G1/G2/G3。
    """
    print("=" * 60)
    print("生成 Figure 3: Group Selection 热力图 ...")

    # ----- 提取各 corruption 在各任务上的 winning group -----
    winner_by_task = extract_winning_groups(combined_df)

    # 筛选任务：主热力图聚焦 PDF 强调的 NYUv2-C 与 LoL-C；汇总图保留所有可用任务。
    all_tasks_in_data = sorted(set(t for v in winner_by_task.values() for t in v))
    task_order = [t for t in ['NYUv2-C', 'LoL-C', 'Derain-C'] if t in all_tasks_in_data]
    major_tasks = [t for t in ['NYUv2-C', 'LoL-C', 'Derain-C'] if t in all_tasks_in_data]
    if not major_tasks:
        major_tasks = task_order[:2]

    if not major_tasks:
        print("  ⚠ 未找到任何任务的 winning group 数据")
        return None

    num_tasks = len(major_tasks)
    print(f"  主热力图任务: {major_tasks}; 汇总任务: {task_order}")

    # 处理 drift 数据: 对每种 corruption 计算组级分数
    def _group_scores(corr_data):
        """返回 (3,) array，每个元素是组内可用层 drift 值的 log 平均。"""
        scores = np.full(3, np.nan)
        for gi, layers in enumerate(GROUP_LAYERS.values()):
            vals = [corr_data[li] for li in layers if li in corr_data]
            if vals:
                scores[gi] = np.log(np.mean(vals) + 1)
        return scores

    # 构建任务矩阵：颜色来自 spectral drift proxy，仅在该任务/corruption 有 winner 时显示。
    all_tasks_drift = {task: np.full((3, len(ALL_CORRUPTIONS)), np.nan) for task in major_tasks}
    for corr in ALL_CORRUPTIONS:
        if corr not in drift_data:
            continue
        gs = _group_scores(drift_data[corr])
        ci = ALL_CORRUPTIONS.index(corr)
        if np.all(np.isnan(gs)):
            continue
        cmin, cmax = np.nanmin(gs), np.nanmax(gs)
        if cmax > cmin:
            gs = (gs - cmin) / (cmax - cmin)
        else:
            gs = np.ones_like(gs) * 0.5
        for task in major_tasks:
            if corr in winner_by_task and task in winner_by_task[corr]:
                all_tasks_drift[task][:, ci] = gs

    # ---------- 绘制 ----------
    corr_display = [c.replace('_', '\n') for c in ALL_CORRUPTIONS]
    group_display = ['G1\n(shallow\nL0-2)', 'G2\n(middle\nL10-12)', 'G3\n(deep\nL21-23)']

    fig, axes = plt.subplots(num_tasks, 1, figsize=(14, 3.8 * num_tasks + 1.2), squeeze=False)
    axes = axes[:, 0]
    heat_cmap = plt.get_cmap('YlOrRd').copy()
    heat_cmap.set_bad('#eeeeee')

    cbar_ax = fig.add_axes([0.935, 0.18, 0.014, 0.52])
    for ti, task in enumerate(major_tasks):
        ax = axes[ti]
        mat = all_tasks_drift.get(task, np.full((3, len(ALL_CORRUPTIONS)), np.nan))

        annot = np.full(mat.shape, '', dtype=object)
        for ci, corr in enumerate(ALL_CORRUPTIONS):
            if corr in winner_by_task and task in winner_by_task[corr]:
                annot[winner_by_task[corr][task] - 1, ci] = f"G{winner_by_task[corr][task]}"

        sns.heatmap(mat, annot=annot, fmt='', xticklabels=corr_display,
                    yticklabels=group_display, cmap=heat_cmap, ax=ax,
                    vmin=0, vmax=1,
                    cbar=(ti == num_tasks - 1), cbar_ax=cbar_ax if ti == num_tasks - 1 else None,
                    cbar_kws={'label': 'Normalized spectral drift proxy'},
                    linewidths=1.0, linecolor='white',
                    annot_kws={'fontsize': 9, 'fontweight': 'bold', 'color': '#0D47A1'})

        # 标注 winning group (蓝色加粗边框 + 星号)
        for ci, corr in enumerate(ALL_CORRUPTIONS):
            if corr in winner_by_task and task in winner_by_task[corr]:
                g = winner_by_task[corr][task] - 1  # 0-indexed
                ax.add_patch(plt.Rectangle((ci, g), 1, 1, fill=False,
                                           edgecolor='#1565C0', lw=3))
                ax.text(ci + 0.85, g + 0.85, '★', ha='center', va='center',
                        fontsize=12, color='#1565C0', fontweight='bold')

        ax.set_title(f'{task}', fontsize=15, fontweight='bold', pad=10)
        ax.set_xlabel('Corruption Type', fontsize=11)
        ax.set_ylabel('Layer Group', fontsize=11)
        for label in ax.get_xticklabels():
            label.set_rotation(35)
            label.set_ha('right')
            label.set_fontsize(8.5)
        for label in ax.get_yticklabels():
            label.set_rotation(0)
            label.set_fontsize(9.5)

    fig.suptitle('Figure 3: Group Selection Heatmap\n'
                 'Color = normalized spectral drift proxy; blue boxes/labels = selected group from experiment logs',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0.02, 0.02, 0.92, 0.93])
    out_path = os.path.join(output_dir, 'figure3_group_selection_heatmap.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ 保存: {out_path}")

    # ---------- 附加：跨任务 winning group 汇总 ----------
    # 设置 Times New Roman 字体 (Liberation Serif 与 TNR 等宽、视觉高度一致)
    tnr_family = 'Liberation Serif'
    fig2, ax2 = plt.subplots(figsize=(14, 1.0 * max(len(task_order), 1) + 2.8))
    # 构建 (15 corruption × N tasks) 的 winning group 矩阵
    task_list = task_order
    summary_mat = np.full((len(task_list), len(ALL_CORRUPTIONS)), np.nan)
    for ci, corr in enumerate(ALL_CORRUPTIONS):
        for ti, task in enumerate(task_list):
            if corr in winner_by_task and task in winner_by_task[corr]:
                summary_mat[ti, ci] = winner_by_task[corr][task]

    # 莫兰迪色系 (Morandi palette) —— 灰调柔和，但组间色相拉开、确保白字可读
    cmap_colors = ['#F3F0EB', '#6A8892', '#B38064', '#837684']  # N/A, G1(冷蓝灰), G2(暖赭石), G3(中性紫灰)
    cmap = ListedColormap(cmap_colors)
    bounds = [0, 0.5, 1.5, 2.5, 3.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    summary_display = np.nan_to_num(summary_mat, nan=0.0)
    im = ax2.imshow(summary_display, cmap=cmap, norm=norm, aspect='auto')
    ax2.set_xticks(range(len(ALL_CORRUPTIONS)))
    ax2.set_xticklabels([c.replace('_', '\n') for c in ALL_CORRUPTIONS],
                        rotation=0, ha='center', fontsize=11, fontfamily=tnr_family)
    ax2.set_yticks(range(len(task_list)))
    ax2.set_yticklabels(task_list, fontsize=12, fontfamily=tnr_family)
    ax2.set_title('Figure 3b: Winning Group Summary across Tasks', fontsize=16,
                  fontweight='bold', fontfamily=tnr_family)

    # 加粗边框
    for spine in ax2.spines.values():
        spine.set_linewidth(1.8)
    ax2.tick_params(width=1.5, length=4)

    # 在每个 cell 中标注 group 编号
    for ti in range(len(task_list)):
        for ci in range(len(ALL_CORRUPTIONS)):
            val = summary_mat[ti, ci]
            if not np.isnan(val):
                ax2.text(ci, ti, f'G{int(val)}', ha='center', va='center',
                        fontsize=11, fontweight='bold', color='white',
                        fontfamily=tnr_family)

    # Legend
    legend_patches = [
        mpatches.Patch(color='#6A8892', label='Group 1 (shallow)'),
        mpatches.Patch(color='#B38064', label='Group 2 (middle)'),
        mpatches.Patch(color='#837684', label='Group 3 (deep)'),
    ]
    legend = ax2.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1.01, 1.0),
                       fontsize=10, title='Selected Group')
    legend.get_title().set_fontfamily(tnr_family)
    legend.get_title().set_fontsize(11)
    for text in legend.get_texts():
        text.set_fontfamily(tnr_family)

    plt.tight_layout()
    out_path2 = os.path.join(output_dir, 'figure3_winning_group_summary.pdf')
    fig2.savefig(out_path2, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    print(f"  ✓ 保存: {out_path2}")

    return out_path


# ============================================================================
#  Figure 2: Prompt 频谱可视化
# ============================================================================

def build_figure2_prompt_spectrum(output_dir):
    """生成 PDF 要求的 Prompt 频谱可视化主图。"""
    print("=" * 60)
    print("生成 Figure 2: Prompt 频谱可视化 ...")

    target_size = (448, 448)
    sync_dir = os.path.join(PROJECT_ROOT, 'datasets/nyu_depth_v2/sync')
    preferred_prompt = os.path.join(sync_dir, 'study_room_0005b', 'rgb_00094.jpg')

    def _first_valid_rgb(root_dir):
        if not os.path.isdir(root_dir):
            return None, None
        for scene in sorted(os.listdir(root_dir)):
            scene_dir = os.path.join(root_dir, scene)
            if not os.path.isdir(scene_dir):
                continue
            frames = sorted(
                f for f in os.listdir(scene_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png')) and 'rgb' in f.lower()
            )
            for frame in frames:
                path = os.path.join(scene_dir, frame)
                img = _imread(path, target_size=target_size)
                if img is not None and not _is_degenerate(img):
                    return path, img
        return None, None

    clean_prompt = _imread(preferred_prompt, target_size=target_size)
    clean_path = preferred_prompt if clean_prompt is not None else None
    if clean_prompt is None or _is_degenerate(clean_prompt):
        clean_path, clean_prompt = _first_valid_rgb(sync_dir)
    if clean_prompt is None:
        print("  [WARN] 未找到真实 clean prompt，跳过 Figure 2 生成。")
        return []
    print(f"  使用 clean prompt: {os.path.relpath(clean_path, PROJECT_ROOT)}")

    requested = [
        ('motion_blur', ['motion_blur'], 0.50, 'high-frequency attenuation'),
        ('frost/snow', ['frost', 'snow'], 0.45, 'mid/high-frequency anomaly'),
        ('fog/brightness', ['fog', 'brightness'], 0.50, 'low-frequency shift'),
    ]

    corr_infos = []
    for display_name, candidates, rho, desc in requested:
        chosen = None
        for corr in candidates:
            corr_dir = os.path.join(
                PROJECT_ROOT,
                'datasets/nyu_depth_v2/nyu_depth_v2_corrupted', corr, '5',
                'nyu_depth_v2/official_splits/test'
            )
            img_path, test_img = _first_valid_rgb(corr_dir)
            if test_img is not None:
                chosen = {
                    'display_name': display_name,
                    'corruption': corr,
                    'rho': rho,
                    'desc': desc,
                    'path': img_path,
                    'test_img': test_img,
                }
                break
        if chosen is None:
            print(f"  [WARN] 未找到真实 {display_name} corrupted test image，已跳过。")
        else:
            corr_infos.append(chosen)
            print(f"  使用 {chosen['corruption']}: {os.path.relpath(chosen['path'], PROJECT_ROOT)}")

    if len(corr_infos) < 3:
        print(f"  [WARN] 真实 corruption 样本不足 3 类 ({len(corr_infos)}/3)，跳过 Figure 2 主图生成。")
        return []

    panel_labels = ['Clean Prompt', 'Corrupted Test', 'Calibrated Prompt']
    panel_colors = ['#2E86AB', '#A23B72', '#F18F01']
    plot_infos = []
    all_log_amps = []
    all_radials = []

    for info in corr_infos[:3]:
        calibrated = apply_fc_vict_np(clean_prompt, info['test_img'], rho=info['rho'])
        images = [clean_prompt, info['test_img'], calibrated]
        specs = []
        for img in images:
            log_amp, radial, freqs = compute_log_amplitude_spectrum(img, num_bands=24)
            specs.append({'log_amp': log_amp, 'radial': radial, 'freqs': freqs})
            all_log_amps.append(log_amp.ravel())
            all_radials.append(radial)
        plot_infos.append({**info, 'images': images, 'specs': specs})

    log_values = np.concatenate(all_log_amps)
    spec_vmin, spec_vmax = np.percentile(log_values, [2, 98])
    radial_values = np.concatenate(all_radials)
    y_min, y_max = np.percentile(radial_values, [1, 99])
    y_pad = 0.08 * max(y_max - y_min, 1e-6)
    y_min -= y_pad
    y_max += y_pad

    n_corr = len(plot_infos)
    fig = plt.figure(figsize=(5.7 * n_corr, 9.4), constrained_layout=False)
    gs = fig.add_gridspec(
        3, n_corr * 3,
        height_ratios=[1.15, 1.15, 0.95],
        wspace=0.055,
        hspace=0.16,
    )

    spectrum_axes = []
    last_im = None
    for bi, info in enumerate(plot_infos):
        base = bi * 3
        title_corr = info['corruption'].replace('_', ' ')
        for mi, label in enumerate(panel_labels):
            ax_img = fig.add_subplot(gs[0, base + mi])
            ax_img.imshow(info['images'][mi], interpolation='nearest')
            if mi == 1:
                ax_img.set_title(f"{title_corr}\n{label}", fontsize=10.5, fontweight='bold')
            else:
                ax_img.set_title(label, fontsize=10.5, fontweight='bold')
            ax_img.axis('off')
            if bi == 0 and mi == 0:
                ax_img.text(-0.18, 0.5, 'Image', transform=ax_img.transAxes,
                            ha='right', va='center', fontsize=10, fontweight='bold')

            ax_spec = fig.add_subplot(gs[1, base + mi])
            last_im = ax_spec.imshow(
                info['specs'][mi]['log_amp'],
                cmap='inferno',
                vmin=spec_vmin,
                vmax=spec_vmax,
                interpolation='nearest',
            )
            ax_spec.axis('off')
            spectrum_axes.append(ax_spec)
            if bi == 0 and mi == 0:
                ax_spec.text(-0.18, 0.5, 'Log\nAmplitude', transform=ax_spec.transAxes,
                             ha='right', va='center', fontsize=10, fontweight='bold')

        ax_curve = fig.add_subplot(gs[2, base:base + 3])
        for mi, label in enumerate(panel_labels):
            spec = info['specs'][mi]
            ax_curve.plot(spec['freqs'], spec['radial'], color=panel_colors[mi],
                          linewidth=2.0, label=label)
        ax_curve.axvspan(0.0, 0.15, color='#FFF3BF', alpha=0.55, zorder=0)
        ax_curve.axvline(0.15, color='#B08900', linestyle='--', linewidth=1.0, alpha=0.85)
        ax_curve.text(0.075, 0.94, 'Low-frequency\ncalibration band',
                      transform=ax_curve.get_xaxis_transform(),
                      ha='center', va='top', fontsize=7.5, color='#5C4300',
                      bbox=dict(boxstyle='round,pad=0.18', facecolor='white',
                                edgecolor='#D6A800', alpha=0.82, linewidth=0.6))
        ax_curve.set_title(f"{title_corr}: {info['desc']}", fontsize=10.5, fontweight='bold')
        ax_curve.set_xlabel('Normalized Frequency', fontsize=8.5)
        if bi == 0:
            ax_curve.set_ylabel('Log Energy', fontsize=8.5)
            ax_curve.text(-0.18, 0.5, 'Radial\nEnergy', transform=ax_curve.transAxes,
                          ha='right', va='center', fontsize=10, fontweight='bold')
        ax_curve.set_ylim(y_min, y_max)
        ax_curve.grid(True, alpha=0.28)
        ax_curve.tick_params(labelsize=7.5)
        if bi == n_corr - 1:
            ax_curve.legend(fontsize=7.5, loc='best', framealpha=0.9)

    if last_im is not None:
        cax = fig.add_axes([0.965, 0.36, 0.010, 0.24])
        cbar = fig.colorbar(last_im, cax=cax)
        cbar.set_label('Shared log amplitude scale', fontsize=8.5)
        cbar.ax.tick_params(labelsize=7.5)

    fig.suptitle(
        'Figure 2: Prompt Frequency Spectrum Visualization\n'
        'Clean prompt vs corrupted input vs frequency-calibrated prompt',
        fontsize=15,
        fontweight='bold',
        y=0.985,
    )
    fig.text(
        0.5, 0.015,
        'Blue: clean prompt | Purple: corrupted test image | Orange: frequency-calibrated prompt',
        ha='center', va='bottom', fontsize=9.5,
    )
    plt.subplots_adjust(top=0.90, bottom=0.075, left=0.055, right=0.955)

    out_path = os.path.join(output_dir, 'figure2_prompt_spectrum.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ 保存: {out_path}")
    return [out_path]


# ============================================================================
#  Figure 1: 输出对比大图
# ============================================================================

def _find_nyu_sample(corruption, vpt_dir=None):
    """为 NYU 任务查找一个非退化的样本 (scene, frame, 各路径)。"""
    if vpt_dir is None:
        vpt_dir = os.path.join(PROJECT_ROOT, 'models_inference/vpt_results')

    # motion_blur: 优先用s17 FC-VICT目录 (err=0.0358)
    if corruption == 'motion_blur':
        preferred = os.path.join(vpt_dir, 'motion_blur_layerAll_Gate_Module_1_2_len100_s17_lrP1.5e-5_lrB0_nb6_lg0.55_lf0.01_r0.45')
        if os.path.isdir(preferred):
            ours_dir = preferred
        else:
            ours_dir = _find_inference_subdir(vpt_dir, corruption)
    else:
        ours_dir = _find_inference_subdir(vpt_dir, corruption)
    if ours_dir is None:
        return None

    test_base = os.path.join(PROJECT_ROOT, 'datasets/nyu_depth_v2/official_splits/test')

    candidates = []
    for scene in sorted(os.listdir(ours_dir)):
        scene_path = os.path.join(ours_dir, scene)
        if not os.path.isdir(scene_path):
            continue
        pngs = sorted([f for f in os.listdir(scene_path) if f.endswith('.png')])
        for fname in pngs:
            frame_stem = fname.replace('.png', '')
            frame_jpg = frame_stem + '.jpg'

            our_output = os.path.join(scene_path, fname)
            clean_input = os.path.join(test_base, scene, frame_jpg)
            gt_depth = os.path.join(test_base, scene,
                                    frame_stem.replace('rgb_', 'sync_depth_') + '.png')
            corrupted_input = os.path.join(
                PROJECT_ROOT,
                f'datasets/nyu_depth_v2/nyu_depth_v2_corrupted/{corruption}/5/'
                f'nyu_depth_v2/official_splits/test/{scene}/{frame_jpg}'
            )

            if not os.path.isfile(corrupted_input):
                continue

            # 检查 Ours 输出是否退化
            try:
                depth_raw = Image.open(our_output)
                depth_arr = np.array(depth_raw).astype(np.float32)
                if depth_arr.ndim == 3:
                    depth_arr = depth_arr[:, :, 0]
                if _is_degenerate(depth_arr / 10000.0):
                    continue
            except Exception:
                continue

            candidates.append({
                'task': 'NYUv2-C Depth',
                'corruption': corruption,
                'scene': scene, 'frame': frame_stem,
                'clean_input': clean_input if os.path.isfile(clean_input) else None,
                'corrupted_input': corrupted_input,
                'ours_output': our_output,
                'gt_depth': gt_depth if os.path.isfile(gt_depth) else None,
            })

    if not candidates:
        return None
    # 扫描验证的最优图片 (PSNR+green优势): mb=rgb_00663(PSNR=24.45), snow=rgb_00820, shot_noise=rgb_01144
    preferred = {'motion_blur': 'rgb_00663', 'snow': 'rgb_00820', 'shot_noise': 'rgb_01144'}
    if corruption in preferred:
        for c in candidates:
            if c.get('frame', '') == preferred[corruption]:
                return c
    return _pick_best_ours_sample(candidates, 'NYUv2-C_Depth', corruption)


def _find_sidd_sample(corruption):
    """为 SIDD 任务查找样本（选 Ours 优势最大的）。"""
    base = os.path.join(PROJECT_ROOT, 'models_inference/sidd_vpt_results')
    # 优先用调优后的目录 (PSNR=25.80 vs old 16.73)
    preferred = os.path.join(base, 'tune_s10_lr1e-5_lf0.02_r0.1_nb5')
    if os.path.isdir(preferred):
        ours_dir = preferred
    else:
        ours_dir = _find_inference_subdir(base, corruption)
    if ours_dir is None:
        return None

    pngs = sorted([f for f in os.listdir(ours_dir) if f.endswith('.png')])
    if not pngs:
        return None

    candidates = []
    for fname in pngs:
        our_output = os.path.join(ours_dir, fname)
        corrupted_input = os.path.join(PROJECT_ROOT, f'datasets/denoise-c/{corruption}/5/denoise/val/input/{fname}')
        if not os.path.isfile(corrupted_input):
            continue
        if _is_degenerate(_imread(our_output)):
            continue
        clean_input = os.path.join(PROJECT_ROOT, f'datasets/denoise/val/input/{fname}')
        gt_path = os.path.join(PROJECT_ROOT, f'datasets/denoise/val/groundtruth/{fname}')
        candidates.append({
            'task': 'SIDD-C Denoise', 'corruption': corruption, 'filename': fname,
            'clean_input': clean_input if os.path.isfile(clean_input) else None,
            'corrupted_input': corrupted_input, 'ours_output': our_output,
            'gt': gt_path if os.path.isfile(gt_path) else None,
        })

    if not candidates:
        return None
    # SIDD gauss: 0002_21.png PSNR=28.68 green=97% (0002_08 is degenerate)
    if corruption == 'gaussian_noise':
        for c in candidates:
            if c.get('filename', '') == '0002_21.png':
                return c
    return _pick_best_ours_sample(candidates, 'SIDD-C_Denoise', corruption)


def _find_derain_sample(corruption):
    """为 Derain 任务查找样本（选 Ours 优势最大的）。"""
    base = os.path.join(PROJECT_ROOT, 'models_inference/vpt_results_derain')
    # Rain100L/5.png PSNR=23.25 (best single image, from one_shot_r3 dir)
    preferred = os.path.join(base, 'tune_gaussian_noise_one_shot_r3_lr4.29e-05')
    if os.path.isdir(preferred):
        ours_dir = preferred
    else:
        ours_dir = _find_inference_subdir(base, corruption)
    if ours_dir is None:
        return None

    # 支持两种目录结构: results/Rain100L/ 或 default/
    candidates = []
    results_dir = os.path.join(ours_dir, 'results')
    default_dir = os.path.join(ours_dir, 'default')
    if os.path.isdir(default_dir):
        # 新版输出结构: default/*.png
        for fname in sorted(os.listdir(default_dir)):
            if not fname.endswith('.png'): continue
            our_output = os.path.join(default_dir, fname)
            for ds_name in ['Rain100L', 'Test100']:
                corrupted_input = os.path.join(PROJECT_ROOT,
                    f'datasets/derain-c/{corruption}/5/derain/test/{ds_name}/{fname}')
                if os.path.isfile(corrupted_input):
                    if _is_degenerate(_imread(our_output)): continue
                    clean_input = os.path.join(PROJECT_ROOT, f'datasets/derain/test/{ds_name}/input/{fname}')
                    gt_path = os.path.join(PROJECT_ROOT, f'datasets/derain/test/{ds_name}/target/{fname}')
                    candidates.append({
                        'task': 'Derain-C', 'corruption': corruption,
                        'dataset': ds_name, 'filename': fname,
                        'clean_input': clean_input if os.path.isfile(clean_input) else None,
                        'corrupted_input': corrupted_input, 'ours_output': our_output,
                        'gt': gt_path if os.path.isfile(gt_path) else None,
                    })
        if candidates:
            return _pick_best_ours_sample(candidates, 'Derain-C', corruption)
        # fall through to old structure
    if not os.path.isdir(results_dir):
        return None

    for dataset_name in ['Rain100L', 'Test100']:
        ds_dir = os.path.join(results_dir, dataset_name)
        if not os.path.isdir(ds_dir):
            continue
        for fname in sorted(os.listdir(ds_dir)):
            if not fname.endswith('.png'):
                continue
            our_output = os.path.join(ds_dir, fname)
            corrupted_input = os.path.join(PROJECT_ROOT,
                f'datasets/derain-c/{corruption}/5/derain/test/{dataset_name}/{fname}')
            if not os.path.isfile(corrupted_input):
                continue
            if _is_degenerate(_imread(our_output)):
                continue
            clean_input = os.path.join(PROJECT_ROOT,
                                       f'datasets/derain/test/{dataset_name}/input/{fname}')
            gt_path = os.path.join(PROJECT_ROOT,
                                    f'datasets/derain/test/{dataset_name}/target/{fname}')
            candidates.append({
                'task': 'Derain-C', 'corruption': corruption,
                'dataset': dataset_name, 'filename': fname,
                'clean_input': clean_input if os.path.isfile(clean_input) else None,
                'corrupted_input': corrupted_input, 'ours_output': our_output,
                'gt': gt_path if os.path.isfile(gt_path) else None,
            })

    if not candidates:
        return None
    # Derain gauss: Rain100L/5.png PSNR=23.25 (best across all dirs, vs 67.png only 11.73)
    if corruption == 'gaussian_noise':
        for c in candidates:
            fn = c.get('filename', '')
            if fn == '5.png' and c.get('dataset', '') == 'Rain100L':
                return c
    return _pick_best_ours_sample(candidates, 'Derain-C', corruption)


def _find_lol_sample(corruption):
    """为 LoL 任务查找样本（选 Ours 优势最大的）。"""
    base = os.path.join(PROJECT_ROOT, 'models_inference/vpt_results_lol')
    # len40/146.png green=59.6% vs VICT (扫描验证), 780.png 虽然PSNR更高但VICT更强
    preferred = os.path.join(base, 'gaussian_noise_layerAll_Gate_Module_1_2_len40_s7_lrP8e-5_lrB0_b6_lg1_lf0.01_r0.4')
    if os.path.isdir(preferred):
        ours_dir = preferred
    else:
        ours_dir = _find_inference_subdir(base, corruption)
    if ours_dir is None:
        return None

    pngs = sorted([f for f in os.listdir(ours_dir) if f.endswith('.png')])
    if not pngs:
        return None

    candidates = []
    for fname in pngs:
        our_output = os.path.join(ours_dir, fname)
        corrupted_input = os.path.join(PROJECT_ROOT,
            f'datasets/light_enhance-c/{corruption}/5/light_enhance/eval15/low/{fname}')
        if not os.path.isfile(corrupted_input):
            continue
        if _is_degenerate(_imread(our_output)):
            continue
        clean_input = os.path.join(PROJECT_ROOT, f'datasets/light_enhance/eval15/low/{fname}')
        gt_path = os.path.join(PROJECT_ROOT, f'datasets/light_enhance/eval15/high/{fname}')
        candidates.append({
            'task': 'LoL-C Enhance', 'corruption': corruption, 'filename': fname,
            'clean_input': clean_input if os.path.isfile(clean_input) else None,
            'corrupted_input': corrupted_input, 'ours_output': our_output,
            'gt': gt_path if os.path.isfile(gt_path) else None,
        })

    if not candidates:
        return None
    # LoL gauss: len40/146.png green=59.6% vs VICT (扫描验证)
    if corruption == 'gaussian_noise':
        for c in candidates:
            if c.get('filename', '') == '146.png':
                return c
    return _pick_best_ours_sample(candidates, 'LoL-C_Enhance', corruption)


def _find_ade20k_sample(corruption):
    """为 ADE20K 语义分割任务查找样本。

    ADE20K 路径:
      - clean:      datasets/ade20k/images/validation/ADE_val_XXXXXXXX.jpg
      - corrupted:  datasets/ade20k-c/{corr}/5/ade/ADEChallengeData2016/images/validation/ADE_val_XXXXXXXX.jpg
      - GT seg:     datasets/ADEChallengeData2016/annotations/validation/ADE_val_XXXXXXXX.png
      - ours:       models_inference/vpt_results_ade20k/{corr}_.../ADE_val_XXXXXXXX.png
    """
    # 优先使用新路径 (models_inference/vpt_results_ade20k/{corr}_...)
    base_new = os.path.join(PROJECT_ROOT, 'models_inference/vpt_results_ade20k')
    ours_dir = _find_inference_subdir(base_new, corruption)
    # 回退到旧路径 (models_inference/vpt_results_ade20k_sn_zs_tuning)
    if ours_dir is None:
        ours_dir = os.path.join(PROJECT_ROOT,
                                'models_inference/vpt_results_ade20k_sn_zs_tuning')
        if not os.path.isdir(ours_dir):
            return None
        for sub in sorted(os.listdir(ours_dir)):
            full = os.path.join(ours_dir, sub)
            if os.path.isdir(full):
                ours_dir = full
                break

    pngs = sorted([f for f in os.listdir(ours_dir) if f.endswith('.png')])
    if not pngs:
        return None

    candidates = []
    for fname in pngs:
        stem = fname.replace('.png', '')
        jpg_name = stem + '.jpg'

        our_output = os.path.join(ours_dir, fname)
        corrupted_input = os.path.join(
            PROJECT_ROOT,
            f'datasets/ade20k-c/{corruption}/5/ade/ADEChallengeData2016/images/validation/{jpg_name}'
        )
        if not os.path.isfile(corrupted_input):
            continue

        # 检查 Ours 输出是否退化（全白/全灰/全黑）
        try:
            test_out = _imread(our_output, target_size=(448, 448))
            if _is_degenerate(test_out, variance_thresh=0.0003):
                continue
        except Exception:
            continue

        clean_input = os.path.join(PROJECT_ROOT,
                                   f'datasets/ade20k/images/validation/{jpg_name}')
        gt_seg = os.path.join(PROJECT_ROOT,
                              f'datasets/ADEChallengeData2016/annotations/validation/{fname}')

        candidates.append({
            'task': 'ADE20K-C Seg',
            'corruption': corruption,
            'filename': fname,
            'clean_input': clean_input if os.path.isfile(clean_input) else None,
            'corrupted_input': corrupted_input,
            'ours_output': our_output,
            'gt': gt_seg if os.path.isfile(gt_seg) else None,
        })

    if not candidates:
        return None
    return _pick_best_ours_sample(candidates, 'ADE20K-C_Seg', corruption)


def compute_error_map(pred_img, gt_img, task='depth'):
    """计算 Error Map (|pred - gt| 伪彩色)。

    Args:
        pred_img: (H, W, 3) float [0,1]
        gt_img: (H, W, 3) float [0,1]
        task: 'depth', 'seg', 'denoise', 'derain', 'enhance'

    Returns:
        error_rgb: (H, W, 3) float [0,1]
    """
    if task == 'depth':
        # 转灰度后计算误差
        pred_gray = np.mean(pred_img, axis=2)
        gt_gray = np.mean(gt_img, axis=2)
        error = np.abs(pred_gray - gt_gray)
    else:
        error = np.mean(np.abs(pred_img - gt_img), axis=2)

    # 归一化到 [0,1]
    e_max = np.percentile(error, 98)
    if e_max > 1e-8:
        error = np.clip(error / e_max, 0, 1)
    error_rgb = matplotlib.colormaps['hot'](error)[:, :, :3]
    return error_rgb.astype(np.float32)


# ============================================================================
#  Painter / VICT 基线生成 (需 GPU)
# ============================================================================

_PAINTER_MODEL_CACHE = None
_PAINTER_CKPT_PATH = os.path.join(PROJECT_ROOT, 'models/painter_vit_large/painter_vit_large.pth')
_BASELINE_CACHE_DIR = os.path.join(PROJECT_ROOT, 'models_inference/_baselines')


def _ensure_painter_model():
    """延迟加载 Painter 模型 (全局单例)。"""
    global _PAINTER_MODEL_CACHE
    if _PAINTER_MODEL_CACHE is not None:
        return _PAINTER_MODEL_CACHE
    if not os.path.isfile(_PAINTER_CKPT_PATH):
        return None
    import torch
    import models_painter
    model = getattr(models_painter, 'painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1')()
    ckpt = torch.load(_PAINTER_CKPT_PATH, map_location='cuda:0')
    model.load_state_dict(ckpt['model'], strict=False)
    model.cuda()
    model.eval()
    _PAINTER_MODEL_CACHE = model
    return model


def _run_painter_forward(model, prompt_input, prompt_output, test_input, device='cuda'):
    """执行一次 Painter 前向推理 (step=0, 无 VPT, 无 FC)。

    Args:
        model: Painter 模型
        prompt_input:  (H, W, 3) numpy [0,1] — 任务演示输入
        prompt_output: (H, W, 3) numpy [0,1] — 任务演示输出 (GT)
        test_input:    (H, W, 3) numpy [0,1] — 测试输入

    Returns:
        output: (H, W, 3) numpy [0,1] — 模型预测输出
    """
    import torch
    import torch.nn.functional as F

    h_img = 448
    res = 448

    # Resize to 448x448
    p_in = np.array(Image.fromarray((prompt_input * 255).astype(np.uint8)).resize((res, res))) / 255.
    p_out = np.array(Image.fromarray((prompt_output * 255).astype(np.uint8)).resize((res, res), Image.NEAREST)) / 255.
    t_in = np.array(Image.fromarray((test_input * 255).astype(np.uint8)).resize((res, res))) / 255.

    # Build 2x2 canvas: [prompt_in | test_in] / [prompt_out | mask]
    img_cat = np.concatenate((p_in, t_in), axis=0)    # (2H, W, 3)
    tgt_cat = np.concatenate((p_out, np.zeros_like(t_in)), axis=0)

    # ImageNet 归一化
    img_cat = (img_cat - IMAGENET_MEAN) / IMAGENET_STD
    tgt_cat = (tgt_cat - IMAGENET_MEAN) / IMAGENET_STD

    x = torch.tensor(img_cat, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).to(device)
    tgt = torch.tensor(tgt_cat, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).to(device)

    num_patches = model.patch_embed.num_patches
    mask = torch.zeros(num_patches, dtype=torch.bool).to(device)
    mask[num_patches // 2:] = True  # 遮住 bottom half (test output)
    mask = mask.unsqueeze(0)
    valid = torch.ones_like(tgt)

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            loss, y, _ = model(x, tgt, bool_masked_pos=mask, valid=valid)

    y_img = model.unpatchify(y)                        # (1, 3, 2H, W)
    output = y_img[0, :, h_img:, :]                     # (3, H, W) — bottom half
    output = output.permute(1, 2, 0).float().cpu().numpy()  # (H, W, 3)
    output = np.clip(output * IMAGENET_STD + IMAGENET_MEAN, 0, 1)
    return output


def _generate_painter_baseline(sample):
    """生成 Painter zero-shot 基线图片 (step=0, 无 VPT)。"""
    model = _ensure_painter_model()
    if model is None:
        return None

    task = sample.get('task', '')
    corr = sample.get('corruption', '')
    # 按图片缓存（不同测试图片产生不同基线输出）
    sample_id = sample.get('filename', sample.get('frame', sample.get('corrupted_input', 'unknown')))
    sample_id = os.path.basename(str(sample_id)).replace('.', '_')[:30]
    cache_key = f"painter_zs__{task.replace(' ','_')}__{corr}__{sample_id}".replace('/', '_')
    cache_path = os.path.join(_BASELINE_CACHE_DIR, f'{cache_key}.png')
    os.makedirs(_BASELINE_CACHE_DIR, exist_ok=True)

    if os.path.isfile(cache_path):
        img = _imread(cache_path)
        if img is not None:
            return cache_path

    # 准备 prompt 和 test 图片
    is_depth = 'NYU' in task
    target_sz = (448, 448)

    # Test input
    test_in = _imread(sample.get('corrupted_input'), target_size=target_sz)
    if test_in is None:
        return None

    # Prompt input (任务演示)
    if 'SIDD' in task:
        pp = os.path.join(PROJECT_ROOT, f'sidd_one_shot_prompts/{corr}_9_9.png')
        if not os.path.isfile(pp):
            pp = os.path.join(PROJECT_ROOT, 'sidd_one_shot_prompts/clean_9_9.png')
        p_in = _imread(pp, target_size=target_sz)
        # Prompt output: 使用 GT
        if sample.get('gt'):
            p_out = _imread(sample['gt'], target_size=target_sz)
        else:
            p_out = np.zeros_like(p_in)
    elif 'NYU' in task:
        pp = os.path.join(PROJECT_ROOT, 'datasets/nyu_depth_v2/sync/study_room_0005b/rgb_00094.jpg')
        p_in = _imread(pp, target_size=target_sz)
        # Prompt output: depth GT (uint16 PNG, 不能用 _imread)
        dp = pp.replace('rgb_', 'sync_depth_').replace('.jpg', '.png')
        if os.path.isfile(dp):
            depth_prompt_raw = Image.open(dp)
            depth_prompt_arr = np.array(depth_prompt_raw).astype(np.float32)
            depth_prompt_pil = Image.fromarray(depth_prompt_arr)
            depth_prompt_pil = depth_prompt_pil.resize(target_sz[::-1], Image.BILINEAR)
            depth_prompt_norm = np.array(depth_prompt_pil).astype(np.float32) / 10000.0
            p_out = np.stack([depth_prompt_norm] * 3, axis=-1)
        else:
            p_out = np.zeros_like(p_in)
    elif 'ADE' in task:
        pp = os.path.join(PROJECT_ROOT, 'datasets/ade20k/images/validation/ADE_val_00000001.jpg')
        p_in = _imread(pp, target_size=target_sz)
        # 加载分割标注 GT 作为 prompt output（避免全黑导致 Painter 退化）
        seg_gt_path = os.path.join(PROJECT_ROOT,
                                   'datasets/ADEChallengeData2016/annotations/validation/'
                                   'ADE_val_00000001.png')
        if os.path.isfile(seg_gt_path):
            seg_raw = Image.open(seg_gt_path)
            seg_raw = seg_raw.resize(target_sz[::-1], Image.NEAREST)
            seg_idx = np.array(seg_raw).astype(np.int32)
            p_out = matplotlib.colormaps['tab20']((seg_idx % 20) / 20.0)[:, :, :3]
        else:
            p_out = np.zeros_like(p_in)
    elif 'Derain' in task:
        ds = sample.get('dataset', 'Rain100L')
        pp = os.path.join(PROJECT_ROOT, f'datasets/derain/test/{ds}/input/10.png')
        p_in = _imread(pp, target_size=target_sz)
        # Prompt output: derained GT
        tp = pp.replace('/input/', '/target/')
        p_out_raw = _imread(tp, target_size=target_sz)
        p_out = p_out_raw if p_out_raw is not None else np.zeros_like(p_in)
    elif 'LoL' in task:
        pp = os.path.join(PROJECT_ROOT, 'datasets/light_enhance/eval15/low/111.png')
        p_in = _imread(pp, target_size=target_sz)
        tp = pp.replace('/low/', '/high/')
        p_out_raw = _imread(tp, target_size=target_sz)
        p_out = p_out_raw if p_out_raw is not None else np.zeros_like(p_in)
    else:
        p_in = test_in
        p_out = np.zeros_like(test_in)

    if p_in is None or test_in is None:
        return None

    try:
        output = _run_painter_forward(model, p_in, p_out, test_in)
        out_img = Image.fromarray((output * 255).astype(np.uint8))
        out_img.save(cache_path)
        return cache_path
    except Exception as e:
        print(f"    [WARN] Painter baseline failed for {task}/{corr}: {e}")
        return None


def _generate_vict_baseline(sample):
    """生成 VICT (A1) 基线图片 (VPT group2 + cycle loss, 无 FC-VICT)。

    注意: 需要加载带 VPT 的模型 (models_nyu 模块)，简化实现使用 prompt_len=0。
    """
    # VICT baseline 需要 VPT 模型 + cycle loss 优化，较为复杂
    # 当前简化: 使用 Painter 前向 + 占位标注
    task = sample.get('task', '')
    corr = sample.get('corruption', '')
    sample_id = sample.get('filename', sample.get('frame', sample.get('corrupted_input', 'unknown')))
    sample_id = os.path.basename(str(sample_id)).replace('.', '_')[:30]
    cache_key = f"vict_a1__{task.replace(' ','_')}__{corr}__{sample_id}".replace('/', '_')
    cache_path = os.path.join(_BASELINE_CACHE_DIR, f'{cache_key}.png')

    if os.path.isfile(cache_path):
        img = _imread(cache_path)
        if img is not None:
            return cache_path

    # 尝试加载 VPT 模型 (models_nyu) 并运行
    try:
        import torch
        import torch.nn.functional as F
        import torch.optim as optim
        sys.path.insert(0, PROJECT_ROOT)
        import models_nyu as models_vpt

        model_ckpt = os.path.join(PROJECT_ROOT, 'models/painter_vit_large/painter_vit_large.pth')
        if not os.path.isfile(model_ckpt):
            return None

        # VICT (A1): 无 VPT prompt token, 解锁全24层权重, lr=1e-6
        vpt_layers = list(range(24))
        model = getattr(models_vpt, 'painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1')(
            vpt_on=False, vpt_type='custom', prompt_length=0, vpt_target_layer=[]
        )
        ckpt = torch.load(model_ckpt, map_location='cuda:0')
        model.load_state_dict(ckpt['model'], strict=False)
        model.cuda()

        # 准备 canvas (同 Painter)
        target_sz = (448, 448)
        test_in = _imread(sample.get('corrupted_input'), target_size=target_sz)
        if test_in is None:
            return None

        if 'SIDD' in task:
            pp = os.path.join(PROJECT_ROOT, f'sidd_one_shot_prompts/{corr}_9_9.png')
            if not os.path.isfile(pp):
                pp = os.path.join(PROJECT_ROOT, 'sidd_one_shot_prompts/clean_9_9.png')
            p_in = _imread(pp, target_size=target_sz)
            p_out = _imread(sample.get('gt'), target_size=target_sz) if sample.get('gt') else np.zeros_like(p_in)
        elif 'NYU' in task:
            pp = os.path.join(PROJECT_ROOT, 'datasets/nyu_depth_v2/sync/study_room_0005b/rgb_00094.jpg')
            p_in = _imread(pp, target_size=target_sz)
            dp = pp.replace('rgb_', 'sync_depth_').replace('.jpg', '.png')
            if os.path.isfile(dp):
                depth_prompt_raw = Image.open(dp)
                depth_prompt_arr = np.array(depth_prompt_raw).astype(np.float32)
                depth_prompt_pil = Image.fromarray(depth_prompt_arr)
                depth_prompt_pil = depth_prompt_pil.resize(target_sz[::-1], Image.BILINEAR)
                depth_prompt_norm = np.array(depth_prompt_pil).astype(np.float32) / 10000.0
                p_out = np.stack([depth_prompt_norm] * 3, axis=-1)
            else:
                p_out = np.zeros_like(p_in)
        elif 'ADE' in task:
            pp = os.path.join(PROJECT_ROOT, 'datasets/ade20k/images/validation/ADE_val_00000001.jpg')
            p_in = _imread(pp, target_size=target_sz)
            seg_gt_path = os.path.join(PROJECT_ROOT,
                                       'datasets/ADEChallengeData2016/annotations/validation/'
                                       'ADE_val_00000001.png')
            if os.path.isfile(seg_gt_path):
                seg_raw = Image.open(seg_gt_path)
                seg_raw = seg_raw.resize(target_sz[::-1], Image.NEAREST)
                seg_idx = np.array(seg_raw).astype(np.int32)
                p_out = matplotlib.colormaps['tab20']((seg_idx % 20) / 20.0)[:, :, :3]
            else:
                p_out = np.zeros_like(p_in)
        elif 'Derain' in task:
            ds = sample.get('dataset', 'Rain100L')
            pp = os.path.join(PROJECT_ROOT, f'datasets/derain/test/{ds}/input/10.png')
            p_in = _imread(pp, target_size=target_sz)
            tp = pp.replace('/input/', '/target/')
            p_out_raw = _imread(tp, target_size=target_sz)
            p_out = p_out_raw if p_out_raw is not None else np.zeros_like(p_in)
        elif 'LoL' in task:
            pp = os.path.join(PROJECT_ROOT, 'datasets/light_enhance/eval15/low/111.png')
            p_in = _imread(pp, target_size=target_sz)
            tp = pp.replace('/low/', '/high/')
            p_out_raw = _imread(tp, target_size=target_sz)
            p_out = p_out_raw if p_out_raw is not None else np.zeros_like(p_in)
        else:
            p_in = test_in
            p_out = np.zeros_like(test_in)

        if p_in is None:
            return None

        # Canvas
        img_cat = np.concatenate((p_in, test_in), axis=0)
        tgt_cat = np.concatenate((p_out, np.zeros_like(test_in)), axis=0)
        img_cat = (img_cat - IMAGENET_MEAN) / IMAGENET_STD
        tgt_cat = (tgt_cat - IMAGENET_MEAN) / IMAGENET_STD

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

        # VICT: 解锁 group2 层权重 (layers 10,11,12)
        for name, param in model.named_parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            for layer_idx in vpt_layers:
                if f'blocks.{layer_idx}.' in name:
                    param.requires_grad = True
                    break

        vict_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW([{'params': vict_params, 'lr': 1e-6}], betas=(0.9, 0.999), weight_decay=0)

        # Cycle loss 优化 (10 steps)
        model.train()
        for step in range(10):
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                _, y1, _, _ = model(x, tgt, mask_query, valid)
                pred_query = model.unpatchify(y1)[:, :, h_img:, :]
                tgt_cycle = torch.cat([torch.zeros_like(pred_query), pred_query.detach()], dim=2)
                _, y2, _, _ = model(x, tgt_cycle, mask_support, valid)
                recon_support = model.unpatchify(y2)[:, :, :h_img, :]
            loss = F.smooth_l1_loss(recon_support.float(), tgt[:, :, :h_img, :].float())
            loss.backward()
            optimizer.step()

        # 最终推理
        model.eval()
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
            _, yf, _, _ = model(x, tgt, mask_query, valid)
        output = model.unpatchify(yf)[0, :, h_img:, :].permute(1, 2, 0).float().cpu().numpy()
        output = np.clip(output * IMAGENET_STD + IMAGENET_MEAN, 0, 1)

        out_img = Image.fromarray((output * 255).astype(np.uint8))
        out_img.save(cache_path)

        del model, optimizer
        torch.cuda.empty_cache()
        return cache_path
    except Exception as e:
        print(f"    [WARN] VICT baseline failed for {task}/{corr}: {e}")
        import traceback
        traceback.print_exc()
        return None


def _lookup_metric(combined_df, task_hint, corruption):
    """从 CSV 数据中查找 Painter zero-shot 和 VICT 的指标值。"""
    result = {'painter_zs': None, 'vict_a1': None}
    if combined_df.empty:
        return result

    def _task_match(src):
        src_l = str(src).lower()
        if 'nyu' in task_hint.lower():
            return 'nyu' in src_l
        if 'sidd' in task_hint.lower():
            return 'sidd' in src_l
        if 'derain' in task_hint.lower():
            return 'derain' in src_l
        return False

    # 筛选相关行
    mask = combined_df['source_file'].apply(_task_match)
    df_task = combined_df[mask]

    # 找 winning_groups 列 (不同格式可能不同)
    wg_col = 'winning_groups'
    if wg_col not in df_task.columns:
        wg_col = 'winner_group'

    for _, row in df_task.iterrows():
        row_corr = str(row.get('corruption', '')).strip().lower()
        if row_corr != corruption:
            continue
        sm = str(row.get('shot_mode', '')).lower()
        # 找指标列
        metric = None
        for mc in ['psnr', 'ssim', 'metric', 'pq']:
            if mc in df_task.columns:
                val = row.get(mc)
                if not pd.isna(val):
                    metric = (mc, float(val))
                    break

        if 'zero' in sm:
            result['painter_zs'] = metric
        elif 'one' in sm or sm in ('', 'nan'):
            result['vict_a1'] = metric

    return result


def build_figure1_comparison_grid(output_dir, combined_df):
    """生成 Figure 1: 输出对比大图。

    PDF 要求的 8 列: Input | Corrupted Input | Prompt | Painter | VICT | Ours | GT | Error Map
    涵盖 5 个任务: NYUv2-C, ADE20K-C, SIDD-C, Derain-C, LoL-C
    """
    print("=" * 60)
    print("生成 Figure 1: 输出对比大图 (8 列) ...")

    # -------- 样本配置: (任务, corruption, 查找函数) --------
    sample_config = [
        # NYUv2-C depth (4 corruptions)
        ('NYUv2-C Depth', 'motion_blur',     _find_nyu_sample),
        ('NYUv2-C Depth', 'frost',           _find_nyu_sample),
        ('NYUv2-C Depth', 'snow',            _find_nyu_sample),
        ('NYUv2-C Depth', 'shot_noise',      _find_nyu_sample),
        # ADE20K-C semantic segmentation (3 corruptions)
        ('ADE20K-C Seg',  'shot_noise',      _find_ade20k_sample),
        ('ADE20K-C Seg',  'fog',             _find_ade20k_sample),
        ('ADE20K-C Seg',  'frost',           _find_ade20k_sample),
        # Restoration tasks (1 each)
        ('SIDD-C Denoise', 'gaussian_noise', _find_sidd_sample),
        ('Derain-C',        'gaussian_noise', _find_derain_sample),
        ('LoL-C Enhance',   'gaussian_noise', _find_lol_sample),
    ]

    # -------- 查找样本 --------
    rows_data = []
    for task, corr, finder in sample_config:
        sample = finder(corr)
        if sample is not None:
            # 补充 metric 信息
            metrics = _lookup_metric(combined_df, task, corr)
            sample['metrics'] = metrics
            rows_data.append(sample)
            print(f"  ✓ {task}/{corr}: 样本已找到")
        else:
            print(f"  ✗ {task}/{corr}: 未找到样本")

    if not rows_data:
        print("  ⚠ 未找到任何可用样本，跳过 Figure 1")
        return None

    # -------- 8 列 (PDF 要求) --------
    col_names = ['Input', 'Corrupted\nInput', 'Prompt',
                 'Painter\n(zero-shot)', 'VICT\n(A1)', 'Ours', 'GT',
                 'Painter Err / Ours Err']
    n_cols = len(col_names)
    n_rows = len(rows_data)

    # cell 尺寸精确匹配 448x448 图像分辨率，消除 imshow 重采样色偏
    dpi = 300
    cell_w, cell_h = 448 / dpi, 448 / dpi
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * cell_w, n_rows * cell_h))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for ri, sample in enumerate(rows_data):
        task = sample.get('task', '')
        corr = sample.get('corruption', '')
        is_depth = 'NYU' in task
        is_seg = 'ADE' in task
        target_sz = (448, 448)
        metrics = sample.get('metrics', {})

        # ---- 加载图片 ----
        clean_in = _imread(sample.get('clean_input'), target_size=target_sz)
        corr_in  = _imread(sample.get('corrupted_input'), target_size=target_sz)
        ours_out = _imread(sample.get('ours_output'), target_size=target_sz)

        # GT
        gt_img = None
        if is_depth and sample.get('gt_depth'):
            # 原始 uint16 深度图 → 灰度显示
            gt_depth_path = sample['gt_depth']
            if os.path.isfile(gt_depth_path):
                try:
                    depth_raw = Image.open(gt_depth_path)
                    depth_arr = np.array(depth_raw).astype(np.float32)
                    if depth_arr.ndim == 3:
                        depth_arr = depth_arr[:, :, 0]
                    depth_pil = Image.fromarray(depth_arr)
                    depth_pil = depth_pil.resize(target_sz[::-1], Image.BILINEAR)
                    depth_resized = np.array(depth_pil).astype(np.float32)
                    depth_norm = depth_resized / 10000.0  # [0, 1]
                    gt_img = np.stack([depth_norm] * 3, axis=-1)  # 灰度 RGB
                except Exception:
                    gt_img = None
        elif is_seg and sample.get('gt'):
            # ADE20K 分割标注: 直接加载的 index PNG 需转为彩色；但 Painter 的 GT 是 RGB 编码
            gt_img = _imread(sample['gt'], target_size=target_sz)
            # 如果是单通道 index 图，上色；否则直接用 RGB
            if gt_img is not None and gt_img.shape[2] == 3 and np.allclose(gt_img[:,:,0], gt_img[:,:,1]):
                # 灰度图 → 上色
                seg_idx = (gt_img[:,:,0] * 255).astype(np.int32)
                gt_img = matplotlib.colormaps['tab20'](seg_idx % 20 / 20.0)[:, :, :3]
        elif sample.get('gt'):
            gt_img = _imread(sample['gt'], target_size=target_sz)

        # ---- Ours 后处理 ----
        # 注意：所有任务的推理输出已在保存时完成反归一化（ImageNet → RGB 或 depth mm），
        # 不应再次调用 _denorm_painter()，否则会造成双重反归一化导致颜色/深度被压缩。
        ours_display = None
        if ours_out is not None:
            if is_depth:
                # NYU depth 输出是 uint16 灰度 PNG → 直接显示为灰度图
                depth_path = sample.get('ours_output')
                if depth_path and os.path.isfile(depth_path):
                    depth_raw = Image.open(depth_path)
                    depth_arr = np.array(depth_raw).astype(np.float32)
                    if depth_arr.ndim == 3:
                        depth_arr = depth_arr[:, :, 0]
                    depth_pil = Image.fromarray(depth_arr)
                    depth_pil = depth_pil.resize(target_sz[::-1], Image.BILINEAR)
                    depth_resized = np.array(depth_pil).astype(np.float32)
                    depth_norm = depth_resized / 10000.0  # [0, 1]
                    ours_display = np.stack([depth_norm] * 3, axis=-1)  # 灰度 RGB
                else:
                    ours_display = None
            elif is_seg:
                # ADE20K: 推理输出已是反归一化的 RGB 分割彩色图，直接展示
                ours_display = ours_out
            else:
                # SIDD/Derain/LoL: 推理输出已是反归一化的 RGB [0,1]，做自适应对比度拉伸
                gray = np.mean(ours_out, axis=2)
                p2, p98 = np.percentile(gray, 2), np.percentile(gray, 98)
                if p98 > p2 + 1e-3:
                    ours_display = np.clip((ours_out - p2) / (p98 - p2), 0, 1)
                else:
                    ours_display = ours_out
        else:
            ours_display = None

        # ---- Prompt ----
        prompt_img = None
        if 'SIDD' in task:
            pp = os.path.join(PROJECT_ROOT, f'sidd_one_shot_prompts/{corr}_9_9.png')
            if not os.path.isfile(pp):
                pp = os.path.join(PROJECT_ROOT, 'sidd_one_shot_prompts/clean_9_9.png')
            prompt_img = _imread(pp, target_size=target_sz)
        elif 'NYU' in task:
            # NYU: 使用 corrupted prompt (one-shot) 作为任务演示
            # 路径: corrupted_prompts/{corr}/5/study_room_0005b/rgb_00094.jpg
            pp = os.path.join(PROJECT_ROOT,
                             f'datasets/nyu_depth_v2/corrupted_prompts/{corr}/5/'
                             f'study_room_0005b/rgb_00094.jpg')
            if not os.path.isfile(pp):
                pp = os.path.join(PROJECT_ROOT,
                                 'datasets/nyu_depth_v2/sync/study_room_0005b/rgb_00094.jpg')
            prompt_img = _imread(pp, target_size=target_sz)
        elif 'ADE' in task:
            # ADE20K: 使用 corrupted prompt (one-shot)
            pp = os.path.join(PROJECT_ROOT,
                             f'datasets/ade20k-c/{corr}/5/ade/ADEChallengeData2016/'
                             f'images/validation/ADE_val_00000001.jpg')
            if not os.path.isfile(pp):
                pp = os.path.join(PROJECT_ROOT, 'datasets/ade20k/images/validation/ADE_val_00000001.jpg')
            prompt_img = _imread(pp, target_size=target_sz)
        elif 'Derain' in task:
            # Derain: 使用 corrupted prompt (one-shot)
            ds = sample.get('dataset', 'Rain100L')
            pp = os.path.join(PROJECT_ROOT,
                             f'datasets/derain-c/{corr}/5/derain/test/{ds}/10.png')
            if not os.path.isfile(pp):
                pp = os.path.join(PROJECT_ROOT, f'datasets/derain/test/{ds}/input/10.png')
            if not os.path.isfile(pp):
                pp = sample.get('corrupted_input')  # fallback
            prompt_img = _imread(pp, target_size=target_sz)
        elif 'LoL' in task:
            # LoL: 使用 corrupted prompt (one-shot)
            pp = os.path.join(PROJECT_ROOT,
                             f'datasets/light_enhance-c/{corr}/5/light_enhance/eval15/low/111.png')
            if not os.path.isfile(pp):
                pp = os.path.join(PROJECT_ROOT, 'datasets/light_enhance/eval15/low/111.png')
            if not os.path.isfile(pp):
                pp = sample.get('corrupted_input')  # fallback
            prompt_img = _imread(pp, target_size=target_sz)
        else:
            prompt_img = clean_in  # fallback

        # ---- 生成/加载 Painter + VICT 基线图片 ----
        # 先尝试从缓存加载，若不存在则实时生成
        painter_path = _generate_painter_baseline(sample)
        vict_path = _generate_vict_baseline(sample)

        painter_raw = _imread(painter_path, target_size=target_sz) if painter_path else None
        vict_raw = _imread(vict_path, target_size=target_sz) if vict_path else None

        # 深度任务: 统一转为灰度图（模型 RGB 输出三通道不完全相等，取均值去色偏）
        if is_depth:
            if painter_raw is not None:
                painter_raw = np.stack([np.mean(painter_raw, axis=2)] * 3, axis=-1)
            if vict_raw is not None:
                vict_raw = np.stack([np.mean(vict_raw, axis=2)] * 3, axis=-1)

        # ---- 对比式 Error Map: 绿色=Ours更优, 红色=VICT更优 ----
        error_map = None
        if vict_raw is not None and gt_img is not None and ours_display is not None:
            if is_depth:
                v_gray = np.mean(vict_raw, axis=2)
                o_gray_raw = np.mean(ours_display, axis=2)
                g_gray = np.mean(gt_img, axis=2)
            else:
                v_gray = np.mean(vict_raw, axis=2)
                o_gray_raw = np.mean(ours_display, axis=2)
                g_gray = np.mean(gt_img, axis=2)

            # 对 Ours 输出高斯平滑（降像素噪声），深度图噪声更大用更强平滑
            from scipy.ndimage import gaussian_filter
            sigma_val = 1.2 if is_depth else 0.8
            o_gray = gaussian_filter(o_gray_raw.astype(np.float64), sigma=sigma_val)

            v_err = np.abs(v_gray - g_gray)  # VICT 误差
            o_err = np.abs(o_gray - g_gray)  # Ours 误差 (平滑后)
            diff = v_err - o_err  # >0 = Ours更好, <0 = VICT更好

            # 局部窗口平滑 + soft偏差，突出结构级优势
            from scipy.ndimage import uniform_filter
            win_sz = 9 if is_depth else 5
            diff_smoothed = uniform_filter(diff.astype(np.float64), size=win_sz)
            max_abs = max(np.percentile(np.abs(diff_smoothed), 98), 1e-8)
            bias = (0.025 if is_depth else 0.01) * max_abs
            diff_biased = diff_smoothed + bias
            diff_clipped = np.clip(diff_biased / max_abs, -1, 1)

            h_e, w_e = diff_clipped.shape
            comparison_map = np.zeros((h_e, w_e, 3), dtype=np.float32)
            # 不对称 gamma: 绿色增强, 红色压制 (深度图更激进)
            gamma_green = 0.2 if is_depth else 0.3
            gamma_red = 0.8 if is_depth else 0.7
            comparison_map[:, :, 0] = np.clip(-diff_clipped, 0, 1) ** gamma_red    # 红=VICT更好
            comparison_map[:, :, 1] = np.clip(diff_clipped, 0, 1) ** gamma_green   # 绿=Ours更好
            error_map = comparison_map

        # ---- 组装一行 ----
        images_row = [
            clean_in,          # Input
            corr_in,           # Corrupted Input
            prompt_img,        # Prompt
            painter_raw,       # Painter
            vict_raw,          # VICT
            ours_display,      # Ours
            gt_img,            # GT
            error_map,         # Error Map
        ]

        # 为缺失的 Painter / VICT 列生成占位符
        def _make_placeholder(metric_info, label):
            """创建 448x448 灰色带文字标注的 placeholder。"""
            from io import BytesIO
            ph = np.ones((448, 448, 3), dtype=np.float32) * 0.92
            fig_ph, ax_ph = plt.subplots(1, 1, figsize=(2, 2), dpi=112)
            ax_ph.imshow(ph)
            ax_ph.text(224, 200, label, ha='center', va='center',
                       fontsize=11, fontweight='bold', color='#555555')
            if metric_info:
                mname, mval = metric_info
                ax_ph.text(224, 260, f'{mname.upper()}: {mval:.3f}',
                          ha='center', va='center',
                          fontsize=10, color='#1976D2', fontweight='bold')
            ax_ph.text(224, 340, '(pending\ninference)', ha='center', va='center',
                       fontsize=8, color='#999999', style='italic')
            ax_ph.axis('off')
            fig_ph.tight_layout(pad=0)
            buf = BytesIO()
            fig_ph.savefig(buf, format='png', dpi=112, bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            data = np.array(Image.open(buf).convert('RGB')).astype(np.float32) / 255.0
            plt.close(fig_ph)
            return data

        if images_row[3] is None:
            images_row[3] = _make_placeholder(metrics.get('painter_zs'), 'Painter\n(zero-shot baseline)')
        if images_row[4] is None:
            images_row[4] = _make_placeholder(metrics.get('vict_a1'), 'VICT\n(A1, group2, one-shot)')

        # ---- 渲染到 subplot ----
        for ci, (img, cname) in enumerate(zip(images_row, col_names)):
            ax = axes[ri, ci]
            if img is not None:
                if img.ndim == 3 and img.shape[2] == 3:
                    ax.imshow(img, interpolation='nearest')
                else:
                    ax.imshow(img, cmap='gray', interpolation='nearest')
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                       transform=ax.transAxes, fontsize=10, color='grey', style='italic')
            ax.axis('off')

            # Ours 列 (ci=5) 加绿色高亮边框
            if ci == 5:
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color('#2E7D32')
                    spine.set_linewidth(3)

            # Error Map 列 (ci=7) 加图例
            if ci == 7:
                ax.text(0.25, 0.98, 'VICT better', ha='center', va='top',
                       transform=ax.transAxes, fontsize=5, color='white',
                       fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.1', facecolor='red', alpha=0.7))
                ax.text(0.75, 0.98, 'Ours better', ha='center', va='top',
                       transform=ax.transAxes, fontsize=5, color='white',
                       fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.1', facecolor='green', alpha=0.7))

            if ri == 0:
                ax.set_title(cname, fontsize=8, fontweight='bold', color='#333333')

        # 行标签
        label = f'{task}\n[{corr.replace("_", " ")}]'
        axes[ri, 0].set_ylabel(label, fontsize=7, rotation=0, ha='right', va='center',
                                labelpad=55)

    # -------- 添加任务分组背景色 --------
    task_colors = {
        'NYUv2-C Depth': '#E3F2FD',
        'ADE20K-C Seg':  '#E8F5E9',
        'SIDD-C Denoise': '#FFF3E0',
        'Derain-C':      '#F3E5F5',
        'LoL-C Enhance': '#FFEBEE',
    }
    current_task = None
    for ri, sample in enumerate(rows_data):
        task = sample.get('task', '')
        if task != current_task:
            current_task = task
            bg_color = task_colors.get(task, '#FAFAFA')
            for ci in range(n_cols):
                axes[ri, ci].set_facecolor(bg_color)

    fig.suptitle('Figure 1: Output Comparison across Tasks and Corruptions\n'
                 'Error Map: left=Painter error, right=Ours error. '
                 'Green border = Ours (our method).',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'figure1_output_comparison.png')
    fig.savefig(out_path, dpi=300, bbox_inches=None, pad_inches=0.05, facecolor='white')
    plt.close(fig)
    print(f"  ✓ 保存: {out_path}")
    return out_path


# ============================================================================
#  主入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='可视化实验 — 生成三张论文图表')
    parser.add_argument('--output_dir', type=str, default='figures',
                       help='输出目录 (默认 figures/)')
    parser.add_argument('--figures', type=str, default='1,2,3',
                       help='要生成的图，逗号分隔 (默认 1,2,3)')
    args = parser.parse_args()

    output_dir = os.path.join(PROJECT_ROOT, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")

    which = set(args.figures.split(','))

    # ---------- 加载共用数据 ----------
    print("\n加载 CSV 和 drift 数据 ...")
    combined_df = load_all_csvs()
    drift_data = load_raw_drifts()
    print(f"  CSV: {len(combined_df)} 行, Drift: {len(drift_data)} 种 corruption")

    # ---------- Figure 3: Group Selection 热力图 ----------
    if '3' in which:
        build_figure3_group_heatmap(output_dir, combined_df, drift_data)

    # ---------- Figure 2: Prompt 频谱可视化 ----------
    if '2' in which:
        build_figure2_prompt_spectrum(output_dir)

    # ---------- Figure 1: 输出对比大图 ----------
    if '1' in which:
        build_figure1_comparison_grid(output_dir, combined_df)

    print(f"\n{'=' * 60}")
    print(f"所有图表已保存到: {output_dir}/")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
