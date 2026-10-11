#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ID="${RUN_ID:-20260717}"
SOURCE_RUN="${SCRIPT_DIR}/gaussian_zero_shot_group1_wd0_val2000_codex_val2000_20260716"
INPUT_DIR="${SOURCE_RUN}/inputs"
SEM_SOURCE="${SOURCE_RUN}/pred_sem"
SAMPLE_FILE="${SOURCE_RUN}/validation_sample.txt"
RUN_DIR="${SCRIPT_DIR}/gaussian_zero_shot_group1_correct_inst_prompt_val2000_${RUN_ID}"
INST_DIR="${RUN_DIR}/pred_inst"
INST_WINNER="${RUN_DIR}/winner_inst.txt"
LOG_FILE="${RUN_DIR}/run.log"

mkdir -p "${INST_DIR}"
printf '%s\n' group1 > "${INST_WINNER}"

INPUT_COUNT="$(find "${INPUT_DIR}" -maxdepth 1 \( -type f -o -type l \) | wc -l)"
SEM_COUNT="$(find "${SEM_SOURCE}" -maxdepth 1 -type f -name '*.png' | wc -l)"
SAMPLE_SHA256="$(sha256sum "${SAMPLE_FILE}" | awk '{print $1}')"
if [[ "${INPUT_COUNT}" -ne 2000 || "${SEM_COUNT}" -ne 2000 ]]; then
    echo "expected 2000 inputs and semantic predictions, got inputs=${INPUT_COUNT} sem=${SEM_COUNT}" >&2
    exit 1
fi

{
    echo "protocol=FC-VICT group1 wd0; corrected instance prompt; held-out val2000"
    echo "reuse_sem_source=${SEM_SOURCE}"
    echo "input_count=${INPUT_COUNT} sem_count=${SEM_COUNT} sample_sha256=${SAMPLE_SHA256}"
    echo "steps=10 lr=2e-5 prompt_len=30 num_bands=2 lambda_gate=0.375 lambda_f=0.05 rho=0.05 weight_decay=0"
} | tee "${LOG_FILE}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/painter_inference_coco_zero_shot_correct_prompt.py" \
    --mode inference --task inst \
    --ckpt_path models/painter_vit_large/painter_vit_large.pth \
    --model painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1 \
    --prompt 000000391460 --input_dir "${INPUT_DIR}" --save_dir "${INST_DIR}" \
    --prompt_image_path data/mmdet_custom/datasets/coco/pano_ca_inst/train_org/000000391460_image_train_org.png \
    --prompt_target_path data/mmdet_custom/datasets/coco/pano_ca_inst/train_org/000000391460_label_train_org.png \
    --steps 10 --lr 2e-5 --lr_backbone 0 --weight_decay 0 \
    --vpt_type custom --prompt_length 30 --vpt_target_layer 0 1 2 10 11 12 21 22 23 \
    --num_bands 2 --lambda_gate 0.375 --lambda_f 0.05 --rho 0.05 \
    --seed 20260426 --use_fc_vict --global_winner_path "${INST_WINNER}" \
    2>&1 | tee -a "${LOG_FILE}"

"${PYTHON_BIN}" eval/coco_panoptic/eval_with_pngs.py \
    --pred_path_sem "${SEM_SOURCE}" --pred_path_inst "${INST_DIR}" \
    --gt_path datasets/coco/annotations/instances_val2017.json \
    --dist_thr 19 --prompt 000000391460 --ckpt_file painter_vit_large.pth \
    --input_size 448 2>&1 | tee -a "${LOG_FILE}"

grep -E "^(PQ|SQ|RQ|PQ_th|PQ_st)[[:space:]]*\|" "${LOG_FILE}" > "${RUN_DIR}/result.txt"
echo "result_file=${RUN_DIR}/result.txt" | tee -a "${LOG_FILE}"
