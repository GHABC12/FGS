#!/bin/bash
# =========================================================
# COCO Panoptic Zero-Shot 全 Corruption 串行调参
# Steps 固定为 10，调参其他超参数
# 用法: bash eval/coco_panoptic/tune_coco_zeroshot_all.sh
# =========================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

mkdir -p logs

CORRUPTIONS=(
    "gaussian_noise"
    "shot_noise"
    "impulse_noise"
    "defocus_blur"
    "glass_blur"
    "motion_blur"
    "zoom_blur"
    "snow"
    "frost"
    "fog"
    "brightness"
    "contrast"
    "elastic_transform"
    "pixelate"
    "jpeg_compression"
)

echo "========================================================="
echo ">>> COCO Panoptic Zero-Shot 全 Corruption 串行调参"
echo ">>> Steps 固定为 10，调参其他超参数"
echo ">>> Corruption 列表 (${#CORRUPTIONS[@]} 个): ${CORRUPTIONS[*]}"
echo ">>> 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================="

TOTAL=${#CORRUPTIONS[@]}
CURRENT=0
FAILED=()

for corr in "${CORRUPTIONS[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo ""
    echo "========================================================="
    echo ">>> [${CURRENT}/${TOTAL}] [$(date '+%H:%M:%S')] 开始调参: ${corr}"
    echo "========================================================="

    if bash "${SCRIPT_DIR}/tune_coco_panoptic.sh" "${corr}" 2>&1 | tee "logs/tune_coco_${corr}_zeroshot.log"; then
        echo ">>> [${CURRENT}/${TOTAL}] [$(date '+%H:%M:%S')] 完成调参: ${corr}"
    else
        echo ">>> [${CURRENT}/${TOTAL}] [$(date '+%H:%M:%S')] 调参失败: ${corr}"
        FAILED+=("${corr}")
    fi
    echo "========================================================="
done

echo ""
echo "========================================================="
echo ">>> 全部调参完成！"
echo ">>> 结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ">>> 结果文件:"
for corr in "${CORRUPTIONS[@]}"; do
    echo "    results_coco_${corr}_tuning.csv"
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo ">>> 失败的 Corruption (${#FAILED[@]} 个): ${FAILED[*]}"
fi
echo "========================================================="
