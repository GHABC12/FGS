#!/usr/bin/env python3
import argparse
import csv
import math
import os
import random
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

CORRUPTION_GROUPS = {
    'Noise': ['gaussian_noise', 'shot_noise', 'impulse_noise'],
    'Blur': ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur'],
    'Weather': ['snow', 'frost', 'fog'],
    'Digital': ['brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'],
}

GROUP_COLORS = {
    'Noise': '#D55E00',
    'Blur': '#0072B2',
    'Weather': '#009E73',
    'Digital': '#CC79A7',
}

ALL_CORRUPTIONS = [c for items in CORRUPTION_GROUPS.values() for c in items]
CORRUPTION_TO_GROUP = {c: g for g, items in CORRUPTION_GROUPS.items() for c in items}


def read_rgb(path, image_size):
    img = Image.open(path).convert('RGB')
    if image_size is not None:
        img = img.resize((image_size, image_size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def normalized_radial_log_energy(img, num_bands):
    gray = img.mean(axis=2).astype(np.float64)
    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    log_amp = np.log1p(np.abs(spectrum))

    h, w = gray.shape
    yy, xx = np.ogrid[:h, :w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    rr = np.sqrt(((yy - cy) / max(cy, 1.0)) ** 2 + ((xx - cx) / max(cx, 1.0)) ** 2) / np.sqrt(2.0)

    radial = np.zeros(num_bands, dtype=np.float64)
    for band in range(num_bands):
        inner = band / num_bands
        outer = (band + 1) / num_bands
        mask = (rr >= inner) & (rr < outer)
        radial[band] = float(log_amp[mask].mean()) if mask.any() else 0.0

    radial = radial - radial.min()
    radial = radial + 1e-12
    return radial / radial.sum()


def l1_distance(a, b):
    return float(np.abs(a - b).sum())


def rgb_test_files(clean_root):
    files = []
    for dirpath, _, filenames in os.walk(clean_root):
        for name in filenames:
            lname = name.lower()
            if lname.startswith('rgb_') and lname.endswith(('.jpg', '.jpeg', '.png')):
                files.append(os.path.join(dirpath, name))
    return sorted(files)


def mean_ci(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(1.96 * arr.std(ddof=1) / math.sqrt(arr.size))


def compute_mismatch_rows(args):
    prompt_path = os.path.join(PROJECT_ROOT, args.prompt_path)
    clean_root = os.path.join(PROJECT_ROOT, args.clean_root)
    corrupted_root = os.path.join(PROJECT_ROOT, args.corrupted_root)

    if not os.path.isfile(prompt_path):
        raise FileNotFoundError(f'Prompt image not found: {prompt_path}')
    if not os.path.isdir(clean_root):
        raise FileNotFoundError(f'Clean test root not found: {clean_root}')
    if not os.path.isdir(corrupted_root):
        raise FileNotFoundError(f'Corrupted root not found: {corrupted_root}')

    clean_files = rgb_test_files(clean_root)
    if not clean_files:
        raise RuntimeError(f'No RGB test images found under {clean_root}')

    rng = random.Random(args.seed)
    if args.sample_size > 0 and len(clean_files) > args.sample_size:
        clean_files = sorted(rng.sample(clean_files, args.sample_size))

    prompt_vec = normalized_radial_log_energy(read_rgb(prompt_path, args.image_size), args.num_bands)
    clean_vecs = {}
    clean_distances = {}
    for path in clean_files:
        rel = os.path.relpath(path, clean_root)
        vec = normalized_radial_log_energy(read_rgb(path, args.image_size), args.num_bands)
        clean_vecs[rel] = vec
        clean_distances[rel] = l1_distance(prompt_vec, vec)

    rows = []
    for corr in ALL_CORRUPTIONS:
        category = CORRUPTION_TO_GROUP[corr]
        for rel in clean_vecs:
            rows.append({
                'category': category,
                'corruption': corr,
                'severity': 0,
                'image': rel,
                'mismatch': 0.0,
            })

        for severity in range(1, 6):
            corr_dir = os.path.join(corrupted_root, corr, str(severity), 'nyu_depth_v2', 'official_splits', 'test')
            if not os.path.isdir(corr_dir):
                print(f'[warning] missing corrupted directory: {corr_dir}')
                continue
            for rel in clean_vecs:
                corr_path = os.path.join(corr_dir, rel)
                if not os.path.isfile(corr_path):
                    continue
                corr_vec = normalized_radial_log_energy(read_rgb(corr_path, args.image_size), args.num_bands)
                mismatch = l1_distance(prompt_vec, corr_vec) - clean_distances[rel]
                rows.append({
                    'category': category,
                    'corruption': corr,
                    'severity': severity,
                    'image': rel,
                    'mismatch': mismatch,
                })
    return rows


def write_mismatch_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['category', 'corruption', 'severity', 'image', 'mismatch'])
        writer.writeheader()
        writer.writerows(rows)


def load_mismatch_rows(path):
    rows = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            row['severity'] = int(row['severity'])
            row['mismatch'] = float(row['mismatch'])
            rows.append(row)
    return rows


def summarize_curves(rows):
    by_corr_sev = defaultdict(list)
    images = set()
    for row in rows:
        by_corr_sev[(row['corruption'], row['severity'])].append(row['mismatch'])
        images.add(row['image'])

    corr_mismatch = {}
    corr_curves = {}
    for corr in ALL_CORRUPTIONS:
        severities, means = [], []
        for sev in range(6):
            mean, _ = mean_ci(by_corr_sev[(corr, sev)])
            if np.isfinite(mean):
                corr_mismatch[(corr, sev)] = mean
            severities.append(sev)
            means.append(mean)
        corr_curves[corr] = (np.asarray(severities), np.asarray(means))

    curves = {}
    for group, corruptions in CORRUPTION_GROUPS.items():
        severities, means, cis = [], [], []
        for sev in range(6):
            corr_values = [corr_mismatch[(corr, sev)] for corr in corruptions if (corr, sev) in corr_mismatch]
            mean, ci = mean_ci(corr_values)
            severities.append(sev)
            means.append(mean)
            cis.append(ci)
        curves[group] = (np.asarray(severities), np.asarray(means), np.asarray(cis))

    meta = {'num_images': len(images), 'num_corruptions': len(ALL_CORRUPTIONS)}
    return curves, corr_mismatch, corr_curves, meta


def parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_performance_points(perf_path, corr_mismatch):
    if perf_path is None:
        return [], 'missing --performance-data'
    if not os.path.isfile(perf_path):
        return [], f'performance data not found: {perf_path}'

    points = []
    with open(perf_path, newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return [], 'performance CSV has no header'
        names = set(reader.fieldnames)
        required_base = {'task', 'corruption', 'severity'}
        has_drop = 'drop' in names
        has_metric = {'metric', 'clean_metric', 'higher_is_better'}.issubset(names)
        if not required_base.issubset(names) or not (has_drop or has_metric):
            return [], (
                'performance CSV must contain task,corruption,severity and either drop, '
                'or metric,clean_metric,higher_is_better columns'
            )
        for row in reader:
            corr = row.get('corruption', '').strip()
            try:
                sev = int(float(row.get('severity', '')))
            except ValueError:
                continue
            mismatch = corr_mismatch.get((corr, sev))
            if mismatch is None:
                continue
            if has_drop:
                drop = parse_float(row.get('drop'))
            else:
                metric = parse_float(row.get('metric'))
                clean = parse_float(row.get('clean_metric'))
                hib_raw = str(row.get('higher_is_better', '')).strip().lower()
                higher_is_better = hib_raw in {'1', 'true', 'yes', 'y', 'higher'}
                if metric is None or clean is None:
                    continue
                if higher_is_better:
                    drop = (clean - metric) / clean if clean != 0 else None
                else:
                    drop = metric - clean
            if drop is None or not np.isfinite(drop):
                continue
            points.append({
                'task': row.get('task', ''),
                'corruption': corr,
                'category': CORRUPTION_TO_GROUP.get(corr, 'Other'),
                'severity': sev,
                'mismatch': mismatch,
                'drop': float(drop),
            })
    if not points:
        return [], 'performance CSV yielded no usable points after matching corruption/severity'
    return points, None


def rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def spearmanr_fallback(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3:
        return np.nan, np.nan
    rx = rankdata(x)
    ry = rankdata(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    try:
        from scipy import stats
        return tuple(map(float, stats.spearmanr(x, y)))
    except Exception:
        return rho, np.nan


def severity_adjusted_spearman(x, y, severity):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    severity = np.asarray(severity)
    if len(x) < 3:
        return np.nan, np.nan
    rx = rankdata(x)
    ry = rankdata(y)
    for sev in np.unique(severity):
        mask = severity == sev
        rx[mask] -= rx[mask].mean()
        ry[mask] -= ry[mask].mean()
    rho = float(np.corrcoef(rx, ry)[0, 1])
    try:
        from scipy import stats
        return tuple(map(float, stats.pearsonr(rx, ry)))
    except Exception:
        return rho, np.nan


def format_pvalue(pval):
    if not np.isfinite(pval):
        return 'p=n/a'
    if pval < 0.001:
        return 'p<0.001'
    return f'p={pval:.3f}'


def add_regression_line(ax, x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.allclose(x, x[0]):
        return
    coef = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 100)
    ax.plot(xx, coef[0] * xx + coef[1], color='0.15', lw=1.4, alpha=0.9, zorder=4)


def tac_params():
    return {
        'font.family': 'serif',
        'font.serif': ['Liberation Serif', 'Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'legend.fontsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    }


def draw_panel_a(curves, meta, output_base):
    plt.rcParams.update(tac_params())
    fig, ax = plt.subplots(figsize=(3.5, 2.35), constrained_layout=True)
    for group, (severity, mean, ci) in curves.items():
        color = GROUP_COLORS[group]
        ax.errorbar(
            severity,
            mean,
            yerr=ci,
            marker='o',
            ms=3.2,
            lw=1.7,
            elinewidth=0.8,
            capsize=2.0,
            capthick=0.8,
            color=color,
            alpha=0.95,
            label=group,
            zorder=3,
        )
    ax.axhline(0, color='0.72', lw=0.7, ls='--')
    ax.set_xlim(-0.1, 5.1)
    ax.set_xticks(range(6))
    ax.set_xlabel('Corruption severity')
    ax.set_ylabel('Excess frequency mismatch')
    ax.set_title('Corruptions induce severity-dependent mismatch', loc='left', pad=3)
    ax.text(
        0.98,
        0.04,
        'category means; 95% CI within each category',
        transform=ax.transAxes,
        ha='right',
        va='bottom',
        fontsize=6.5,
        color='0.38',
    )
    ax.grid(True, color='0.92', lw=0.45)
    ax.legend(frameon=False, ncol=2, loc='upper left', handlelength=1.6, columnspacing=0.9)

    os.makedirs(os.path.dirname(output_base), exist_ok=True)
    png_path = output_base + '_a.png'
    pdf_path = output_base + '_a.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f'[saved] {png_path}')
    print(f'[saved] {pdf_path}')
    plt.close(fig)


def draw_panel_b(perf_points, perf_warning, output_base):
    plt.rcParams.update(tac_params())
    fig, ax = plt.subplots(figsize=(3.5, 2.35), constrained_layout=True)

    if perf_points:
        for group in CORRUPTION_GROUPS:
            group_points = [p for p in perf_points if p['category'] == group]
            if not group_points:
                continue
            ax.scatter(
                [p['mismatch'] for p in group_points],
                [p['drop'] for p in group_points],
                s=14,
                marker='o',
                color=GROUP_COLORS[group],
                alpha=0.45,
                edgecolor='none',
                linewidth=0,
                zorder=3,
            )
        x = [p['mismatch'] for p in perf_points]
        y = [p['drop'] for p in perf_points]
        s = [p['severity'] for p in perf_points]
        add_regression_line(ax, x, y)
        rho, pval = spearmanr_fallback(x, y)
        adj_rho, adj_pval = severity_adjusted_spearman(x, y, s)
        ax.text(
            0.04,
            0.96,
            f'n={len(perf_points)} task-corruption-severity points' + '\n' +
            rf'raw Spearman $\rho={rho:.2f}$ ({format_pvalue(pval)})' + '\n' +
            rf'severity-adjusted $\rho={adj_rho:.2f}$ ({format_pvalue(adj_pval)})',
            transform=ax.transAxes,
            ha='left',
            va='top',
            fontsize=6.6,
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.78, 'pad': 1.5},
        )
        ax.set_xlabel('Excess frequency mismatch')
        ax.set_ylabel('Normalized performance drop')
        ax.grid(True, color='0.92', lw=0.45)
    else:
        ax.text(
            0.5,
            0.58,
            'Performance data required',
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=9,
            weight='bold',
            color='0.20',
        )
        ax.text(
            0.5,
            0.41,
            perf_warning or 'missing clean/per-severity metrics',
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=7,
            color='0.35',
            wrap=True,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color('0.75')
    ax.set_title('Larger mismatch is associated with larger drop', loc='left', pad=3)

    os.makedirs(os.path.dirname(output_base), exist_ok=True)
    png_path = output_base + '_b.png'
    pdf_path = output_base + '_b.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f'[saved] {png_path}')
    print(f'[saved] {pdf_path}')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Draw the Introduction frequency mismatch figure.')
    parser.add_argument('--prompt-path', default='datasets/nyu_depth_v2/sync/study_room_0005b/rgb_00094.jpg')
    parser.add_argument('--clean-root', default='datasets/nyu_depth_v2/official_splits/test')
    parser.add_argument('--corrupted-root', default='datasets/nyu_depth_v2/nyu_depth_v2_corrupted')
    parser.add_argument('--sample-size', type=int, default=65)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--num-bands', type=int, default=32)
    parser.add_argument('--mismatch-cache', default='figures/intro_frequency_mismatch_data.csv')
    parser.add_argument('--performance-data', default=None,
                        help='CSV with task,corruption,severity and either drop or metric,clean_metric,higher_is_better.')
    parser.add_argument('--output-base', default='figures/intro_frequency_mismatch')
    parser.add_argument('--force-recompute', action='store_true')
    args = parser.parse_args()

    cache_path = os.path.join(PROJECT_ROOT, args.mismatch_cache)
    if os.path.isfile(cache_path) and not args.force_recompute:
        rows = load_mismatch_rows(cache_path)
        print(f'[cache] loaded {len(rows)} mismatch rows from {cache_path}')
    else:
        rows = compute_mismatch_rows(args)
        write_mismatch_csv(rows, cache_path)
        print(f'[cache] wrote {len(rows)} mismatch rows to {cache_path}')

    curves, corr_mismatch, corr_curves, meta = summarize_curves(rows)
    perf_path = os.path.join(PROJECT_ROOT, args.performance_data) if args.performance_data else None
    perf_points, perf_warning = load_performance_points(perf_path, corr_mismatch)
    if perf_warning:
        print(f'[warning] right panel not fully rendered: {perf_warning}')

    output_base = os.path.join(PROJECT_ROOT, args.output_base)
    draw_panel_a(curves, meta, output_base)
    draw_panel_b(perf_points, perf_warning, output_base)


if __name__ == '__main__':
    main()
