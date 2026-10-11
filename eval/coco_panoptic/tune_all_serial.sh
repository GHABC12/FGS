#!/bin/bash
# =========================================================
# COCO Panoptic 串行调参 - 依次执行 4 个 corruption
# =========================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

mkdir -p logs

CORRUPTIONS=(
    "gaussian_noise"
    "shot_noise"
    "defocus_blur"
    "jpeg_compression"
)

echo "========================================================="
echo ">>> COCO Panoptic 串行调参"
echo ">>> Corruption 列表: ${CORRUPTIONS[*]}"
echo ">>> 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================="

for corr in "${CORRUPTIONS[@]}"; do
    echo ""
    echo "========================================================="
    echo ">>> [$(date '+%H:%M:%S')] 开始调参: ${corr}"
    echo "========================================================="

    bash "${SCRIPT_DIR}/tune_coco_panoptic.sh" "${corr}" 2>&1 | tee "logs/tune_coco_${corr}.log"

    echo ""
    echo ">>> [$(date '+%H:%M:%S')] 完成调参: ${corr}"
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
echo "========================================================="
