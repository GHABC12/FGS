#!/usr/bin/env python3
"""为 figure3_winning_group_summary 生成 10 个不同莫兰迪色系配色的 PDF 版本。"""
import os, sys, warnings
warnings.filterwarnings('ignore')
os.chdir('/home/user/shiyunxiao/Painter-main/Painter')
sys.path.insert(0, '.')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import numpy as np

from visualize_experiment import (
    load_all_csvs, load_raw_drifts, extract_winning_groups, PROJECT_ROOT, ALL_CORRUPTIONS
)

# ---- 10 套莫兰迪色系配色方案 -------------------------------------------
# 格式: (方案名, [N/A, G1, G2, G3])
palettes = [
    # 1. 经典莫兰迪: 蓝灰 / 暖赭 / 紫灰
    ("v1_classic",        ['#F3F0EB', '#6A8892', '#B38064', '#837684']),

    # 2. 暖调莫兰迪: 杏色 / 驼色 / 灰棕
    ("v2_warm_earth",     ['#F4F0E8', '#A0896E', '#C49B7A', '#8A7B6E']),

    # 3. 冷调莫兰迪: 雾蓝 / 灰绿 / 鼠尾草
    ("v3_cool_mist",      ['#F1F2EE', '#7D9197', '#8FA08A', '#9A99A0']),

    # 4. 烟粉莫兰迪: 豆沙粉 / 干枯玫瑰 / 灰褐
    ("v4_dusty_rose",     ['#F3EFEC', '#AA8C8C', '#C4A0A0', '#8A7A7E']),

    # 5. 秋意莫兰迪: 枯叶 / 栗色 / 烟灰
    ("v5_autumn_haze",    ['#F3F0E9', '#9A8B78', '#B8946E', '#7E7B74']),

    # 6. 雾凇莫兰迪: 冰蓝 / 灰绿 / 石板灰
    ("v6_frosted",        ['#F0F1F0', '#829AA3', '#9BA898', '#8A8D8E']),

    # 7. 暮色莫兰迪: 苍蓝 / 枯橙 / 暮紫
    ("v7_twilight",       ['#F2F0ED', '#6F8691', '#C49B7A', '#82778C']),

    # 8. 陶土莫兰迪: 陶土 / 砖红 / 岩灰
    ("v8_terracotta",     ['#F3F0EA', '#A48E7C', '#C49B7A', '#877D78']),

    # 9. 湖光莫兰迪: 湖水蓝 / 沙色 / 雾紫
    ("v9_lake_haze",      ['#F1F1EE', '#72909C', '#BFA88B', '#8B8894']),

    # 10. 极简莫兰迪: 铅灰 / 米驼 / 炭灰
    ("v10_graphite",      ['#F2F0EC', '#8A9298', '#B8A088', '#787A80']),
]


def generate_figure_for_palette(winner_by_task, task_order, palette_colors, output_path):
    """使用指定配色生成 winning group 汇总图。"""
    tnr_family = 'Liberation Serif'
    fig2, ax2 = plt.subplots(figsize=(14, 1.0 * max(len(task_order), 1) + 2.8))

    task_list = task_order
    summary_mat = np.full((len(task_list), len(ALL_CORRUPTIONS)), np.nan)
    for ci, corr in enumerate(ALL_CORRUPTIONS):
        for ti, task in enumerate(task_list):
            if corr in winner_by_task and task in winner_by_task[corr]:
                summary_mat[ti, ci] = winner_by_task[corr][task]

    cmap = ListedColormap(palette_colors)
    bounds = [0, 0.5, 1.5, 2.5, 3.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    summary_display = np.nan_to_num(summary_mat, nan=0.0)
    im = ax2.imshow(summary_display, cmap=cmap, norm=norm, aspect='auto')
    ax2.set_xticks(range(len(ALL_CORRUPTIONS)))
    ax2.set_xticklabels([c.replace('_', '\n') for c in ALL_CORRUPTIONS],
                        rotation=0, ha='center', fontsize=11, fontfamily=tnr_family)
    ax2.set_yticks(range(len(task_list)))
    ax2.set_yticklabels(task_list, fontsize=12, fontfamily=tnr_family)
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

    legend_patches = [
        mpatches.Patch(color=palette_colors[1], label='Group 1 (shallow)'),
        mpatches.Patch(color=palette_colors[2], label='Group 2 (middle)'),
        mpatches.Patch(color=palette_colors[3], label='Group 3 (deep)'),
    ]
    legend = ax2.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1.01, 1.0),
                       fontsize=10, title='Selected Group')
    legend.get_title().set_fontfamily(tnr_family)
    legend.get_title().set_fontsize(11)
    for text in legend.get_texts():
        text.set_fontfamily(tnr_family)

    plt.tight_layout()
    fig2.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    return output_path


def main():
    output_dir = os.path.join(PROJECT_ROOT, 'figures')
    print("加载数据...")
    combined_df = load_all_csvs()
    winner_by_task = extract_winning_groups(combined_df)

    # 确定 task order（与主脚本一致）
    all_tasks_in_data = sorted(set(t for v in winner_by_task.values() for t in v))
    task_order = [t for t in ['NYUv2-C', 'LoL-C', 'Derain-C'] if t in all_tasks_in_data]

    print(f"任务: {task_order}")
    print(f"生成 {len(palettes)} 个配色版本...")

    for name, colors in palettes:
        fname = f'figure3_winning_group_summary_{name}.pdf'
        out_path = os.path.join(output_dir, fname)
        generate_figure_for_palette(winner_by_task, task_order, colors, out_path)
        print(f"  ✓ {fname}  N/A={colors[0]} G1={colors[1]} G2={colors[2]} G3={colors[3]}")

    print(f"\n全部保存至: {output_dir}/")


if __name__ == '__main__':
    main()
