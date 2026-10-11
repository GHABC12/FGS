#!/bin/bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 RUN_DIR STEPS MAX_IMAGES" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

RUN_DIR="$1"
STEPS="$2"
MAX_IMAGES="$3"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/iCompute/bin/python}"
INPUT_DIR="datasets/light_enhance-c/pixelate/5/light_enhance/eval15/low"
EXPECTED_IMAGES=15
if [ "${MAX_IMAGES}" -gt 0 ]; then
  EXPECTED_IMAGES="${MAX_IMAGES}"
fi

mkdir -p "${RUN_DIR}/scratch"
printf 'group2\n' > "${RUN_DIR}/fixed_winner.txt"
find "${INPUT_DIR}" -maxdepth 1 -type f -name '*.png' | sort > "${RUN_DIR}/input_images.txt"

cat > "${RUN_DIR}/parameters.json" <<EOF
{
  "task": "LOL-C pixelate PSNR versus continuous TTA steps",
  "corruption": "pixelate",
  "severity": 5,
  "prompt_mode": "one-shot",
  "seed": 2026,
  "prompt_id": "100",
  "prompt_length": 30,
  "fixed_winner": "group2",
  "active_layers": [10, 11, 12],
  "steps": ${STEPS},
  "max_images": ${MAX_IMAGES},
  "use_fc_vict": true,
  "lambda_f": 0.005,
  "rho": 0.3,
  "num_bands": 6,
  "lambda_gate": 0.3,
  "lr_prompt": 0.00004,
  "lr_backbone": 0
}
EOF

sha256sum \
  eval/lol/eval_new.sh \
  eval/lol/painter_inference_new.py \
  eval/lol/painter_steps_curve.py \
  eval/lol/summarize_steps_curve.py \
  eval/lol/run_steps_curve.sh > "${RUN_DIR}/sha256_before.txt"

"${PYTHON_BIN}" "${SCRIPT_DIR}/painter_steps_curve.py" \
  --mode inference \
  --use_fc_vict \
  --lambda_f 0.005 \
  --rho 0.3 \
  --seed 2026 \
  --global_winner_path "${RUN_DIR}/fixed_winner.txt" \
  --ckpt_path models/painter_vit_large/painter_vit_large.pth \
  --model painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1 \
  --prompt 100 \
  --split train-c \
  --perturbation pixelate \
  --level 5 \
  --input_dir "${INPUT_DIR}" \
  --save_dir "${RUN_DIR}/scratch" \
  --steps "${STEPS}" \
  --lr 4e-5 \
  --lr_backbone 0 \
  --vpt_type custom \
  --prompt_length 30 \
  --vpt_target_layer 0 1 2 10 11 12 21 22 23 \
  --num_bands 6 \
  --lambda_gate 0.3 \
  --prompt_mode one-shot \
  --curve_csv "${RUN_DIR}/steps_psnr_raw.csv" \
  --failures_file "${RUN_DIR}/run_status.json" \
  --max_images "${MAX_IMAGES}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_steps_curve.py" \
  --raw-csv "${RUN_DIR}/steps_psnr_raw.csv" \
  --summary-csv "${RUN_DIR}/steps_psnr_summary.csv" \
  --plot-png "${RUN_DIR}/steps_psnr_curve.png" \
  --result-json "${RUN_DIR}/criterion_result.json" \
  --expected-images "${EXPECTED_IMAGES}" \
  --expected-steps "${STEPS}"

sha256sum eval/lol/eval_new.sh > "${RUN_DIR}/sha256_after.txt"
echo "[CURVE] All artifacts written to ${RUN_DIR}"
