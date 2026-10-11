#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ -n "${PYTHON_BIN:-}" ]; then
    if command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v "${PYTHON_BIN}")"
    elif [ -x "${PYTHON_BIN}" ]; then
        PYTHON_BIN="$(readlink -f "${PYTHON_BIN}")"
    else
        echo "[Error] Invalid PYTHON_BIN: ${PYTHON_BIN}" >&2
        exit 1
    fi
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
elif [ -x "/usr/local/iCompute/bin/python" ]; then
    PYTHON_BIN="/usr/local/iCompute/bin/python"
else
    echo "[Error] Python interpreter not found. Set PYTHON_BIN explicitly." >&2
    exit 1
fi
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"

PROMPT_MODE="${PROMPT_MODE:-}"
CKPT_FILE="painter_vit_large/painter_vit_large.pth"
PROMPT="000000391460"
MODEL="painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1"
CKPT_PATH="models/${CKPT_FILE}"
VPT_TYPE="custom"
TARGET_LAYERS="0 1 2 10 11 12 21 22 23"
SEVERITY=5
FC_VICT_FLAG="--use_fc_vict"
LR_BACKBONE=0
INFER_SAMPLE_COUNT="${INFER_SAMPLE_COUNT:-50}"
DRIFT_SAMPLE_COUNT="${DRIFT_SAMPLE_COUNT:-50}"
RANDOM_SEED=20260426
STEPS=10
WEIGHT_DECAY=0
RUN_ROOT="${RUN_ROOT:-eval/coco_panoptic/auto_tune_15x2_50_20260719}"
ALLOW_NON_100="${ALLOW_NON_100:-0}"
SMOKE_TEST_ONLY="${SMOKE_TEST_ONLY:-0}"

CLEAN_DATA_DIR="datasets/coco/val2017/"
GLOBAL_ANCHOR_PREFIX="anchor_stats_global_clean_coco_correct_prompt"
CALIBRATION_FILE="calibration_baseline_coco.pt"
GT_ANNOTATIONS="datasets/coco/annotations/instances_val2017.json"
INFERENCE_SCRIPT="${SCRIPT_DIR}/painter_inference_coco_zero_shot_correct_prompt.py"
EVALUATOR_SCRIPT="${SCRIPT_DIR}/eval_with_pngs.py"

declare -a PROMPT_PATH_ARGS=()

set_prompt_path_args() {
    local task=$1
    local prompt_mode=$2
    local corruption=$3
    local prompt_image
    local prompt_target

    case "${task}" in
        semseg)
            prompt_image="datasets/coco/train2017/${PROMPT}.jpg"
            prompt_target="datasets/coco/pano_sem_seg/panoptic_segm_train2017_with_color/${PROMPT}.png"
            ;;
        inst)
            prompt_image="data/mmdet_custom/datasets/coco/pano_ca_inst/train_org/${PROMPT}_image_train_org.png"
            prompt_target="data/mmdet_custom/datasets/coco/pano_ca_inst/train_org/${PROMPT}_label_train_org.png"
            ;;
        *)
            echo "[Error] Unsupported task: ${task}" >&2
            return 1
            ;;
    esac

    if [ "${prompt_mode}" = "one_shot" ]; then
        prompt_image="datasets/coco/corrupted_prompts/${corruption}/${SEVERITY}/${PROMPT}.jpg"
    elif [ "${prompt_mode}" != "zero_shot" ]; then
        echo "[Error] Unsupported prompt mode: ${prompt_mode}" >&2
        return 1
    fi

    if [ ! -f "${prompt_image}" ] || [ ! -f "${prompt_target}" ]; then
        echo "[Error] Missing prompt asset: image=${prompt_image}, target=${prompt_target}" >&2
        return 1
    fi

    PROMPT_PATH_ARGS=(
        --prompt_image_path "${prompt_image}"
        --prompt_target_path "${prompt_target}"
    )
}

normalize_corruption() {
    if [ "$1" = "guassion_noise" ]; then
        echo "gaussian_noise"
    else
        echo "$1"
    fi
}

is_number() {
    "${PYTHON_BIN}" - "$1" <<'PY'
import math
import sys
try:
    v = float(sys.argv[1])
    ok = math.isfinite(v)
except Exception:
    ok = False
sys.exit(0 if ok else 1)
PY
}

float_gt() {
    "${PYTHON_BIN}" - "$1" "$2" <<'PY'
import sys
print(1 if float(sys.argv[1]) > float(sys.argv[2]) else 0)
PY
}

shrink_float_range() {
    "${PYTHON_BIN}" - "$1" "$2" "$3" "$4" <<'PY'
import sys
best = float(sys.argv[1])
step = float(sys.argv[2])
mn = float(sys.argv[3])
mx = float(sys.argv[4])
vals = []
for distance in range(0, 8):
    offsets = [0] if distance == 0 else [-distance, distance]
    for offset in offsets:
        value = best + offset * step
        if mn <= value <= mx and all(abs(value - existing) > 1e-12 for existing in vals):
            vals.append(value)
        if len(vals) == 3:
            break
    if len(vals) == 3:
        break
vals = sorted(vals)
if len(vals) != 3:
    raise SystemExit(f"cannot build 3 unique values around {best} within [{mn}, {mx}]")
def fmt(v):
    if abs(v) < 1e-4:
        return f"{v:.6g}"
    return f"{v:.6g}"
print(" ".join(fmt(v) for v in vals))
PY
}

next_half_step() {
    "${PYTHON_BIN}" - "$@" <<'PY'
import sys
best = float(sys.argv[1])
values = [float(value) for value in sys.argv[2:]]
distances = [abs(value - best) for value in values if abs(value - best) > 1e-12]
if not distances:
    raise SystemExit("no distinct neighbor for range shrink")
print(f"{min(distances) / 2:.12g}")
PY
}

shrink_int_range() {
    "${PYTHON_BIN}" - "$1" "$2" "$3" "$4" <<'PY'
import sys
best = int(round(float(sys.argv[1])))
step = float(sys.argv[2])
mn = int(sys.argv[3])
mx = int(sys.argv[4])
vals = []
for raw in (best - step, best, best + step):
    v = max(mn, min(mx, int(round(raw))))
    if v not in vals:
        vals.append(v)
for dist in range(1, mx - mn + 2):
    for v in (best - dist, best + dist):
        if mn <= v <= mx and v not in vals:
            vals.append(v)
        if len(vals) == 3:
            print(" ".join(str(x) for x in sorted(vals)))
            sys.exit(0)
print(" ".join(str(x) for x in sorted(vals[:3])))
PY
}

sample_file_list() {
    local src_list=$1
    local dst_list=$2
    local sample_count=$3
    local seed=$4
    "${PYTHON_BIN}" - "$src_list" "$dst_list" "$sample_count" "$seed" <<'PY'
import random
import sys
src, dst, k_str, seed_str = sys.argv[1:5]
k = int(k_str)
seed = int(seed_str)
with open(src, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]
if not lines:
    selected = []
else:
    rnd = random.Random(seed)
    k = max(1, min(k, len(lines)))
    selected = rnd.sample(lines, k)
with open(dst, "w", encoding="utf-8") as f:
    for item in selected:
        f.write(item + "\n")
PY
}

validate_prediction_ids() {
    local prediction_dir=$1
    "${PYTHON_BIN}" - "${FIXED_SAMPLE_LIST}" "${prediction_dir}" "${INFER_SAMPLE_COUNT}" <<'PY'
import pathlib
import sys

sample_list, prediction_dir, expected_count = sys.argv[1:]
expected_count = int(expected_count)
with open(sample_list, "r", encoding="utf-8") as handle:
    expected = {pathlib.Path(line.strip()).stem for line in handle if line.strip()}
actual = {path.stem for path in pathlib.Path(prediction_dir).glob("*.png")}
if len(expected) != expected_count or len(actual) != expected_count or actual != expected:
    print(
        f"prediction validation failed: expected={len(expected)}, actual={len(actual)}, "
        f"missing={sorted(expected - actual)[:5]}, extra={sorted(actual - expected)[:5]}",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
}

validate_anchor_file() {
    local anchor_file=$1
    "${PYTHON_BIN}" - "${anchor_file}" <<'PY'
import sys
import torch

value = torch.load(sys.argv[1], map_location="cpu")
if not isinstance(value, dict) or not value:
    raise SystemExit("invalid anchor structure")
PY
}

validate_winner_file() {
    local winner_file=$1
    local winner
    [ -s "${winner_file}" ] || return 1
    winner=$(tr -d '[:space:]' < "${winner_file}")
    [ "${winner}" = "group1" ] || [ "${winner}" = "group2" ] || [ "${winner}" = "group3" ]
}

validate_winner_scores_file() {
    local scores_file=$1
    [ -s "${scores_file}" ] &&
        grep -q '^group1_score=' "${scores_file}" &&
        grep -q '^group2_score=' "${scores_file}" &&
        grep -q '^group3_score=' "${scores_file}" &&
        grep -Eq '^winner=group[123]$' "${scores_file}"
}

get_cached_pq() {
    local stage=$1
    local round=$2
    local param1=$3
    local param1_val=$4
    local param2=$5
    local param2_val=$6
    local steps=$7
    local lr_prompt=$8
    local prompt_len=$9
    local num_bands=${10}
    local lambda_gate=${11}
    local lambda_f=${12}
    local rho=${13}

    [ -f "${RESULT_LOG}" ] || return 0
    "${PYTHON_BIN}" - "${RESULT_LOG}" "$PROMPT_MODE" "$CORRUPTION" "$stage" "$round" "$param1" "$param1_val" "$param2" "$param2_val" "$steps" "$lr_prompt" "$prompt_len" "$num_bands" "$lambda_gate" "$lambda_f" "$rho" "$PROTOCOL_HASH" "$INFER_SAMPLE_COUNT" "$SAMPLE_LIST_SHA256" <<'PY'
import csv
import math
import sys
path, prompt_mode, corruption, stage, rnd, param1, param1_val, param2, param2_val, steps, lr, plen, nb, lg, lf, rho, protocol_hash, sample_count, sample_sha256 = sys.argv[1:]
try:
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row:
                continue
            checks = {
                "prompt_mode": prompt_mode,
                "corruption": corruption,
                "stage": stage,
                "round": rnd,
                "param1": param1,
                "param1_val": param1_val,
                "param2": param2,
                "param2_val": param2_val,
                "steps": steps,
                "lr_prompt": lr,
                "prompt_len": plen,
                "num_bands": nb,
                "lambda_gate": lg,
                "lambda_f": lf,
                "rho": rho,
                "protocol_hash": protocol_hash,
            }
            if (
                all(row.get(k) == v for k, v in checks.items())
                and row.get("status") == "OK"
                and row.get("sample_count") == sample_count
                and row.get("sample_list_sha256") == sample_sha256
            ):
                pq = row.get("pq", "")
                try:
                    v = float(pq)
                    if math.isfinite(v):
                        print(pq)
                        sys.exit(0)
                except Exception:
                    pass
except FileNotFoundError:
    pass
PY
}

append_result() {
    local stage=$1
    local round=$2
    local param1=$3
    local param1_val=$4
    local param2=$5
    local param2_val=$6
    local steps=$7
    local lr_prompt=$8
    local prompt_len=$9
    local num_bands=${10}
    local lambda_gate=${11}
    local lambda_f=${12}
    local rho=${13}
    local winning_groups_sem=${14}
    local winning_groups_inst=${15}
    local status=${16}
    local pq=${17}
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    winning_groups_sem=$(printf '%s' "${winning_groups_sem}" | tr '\n,' '  ')
    winning_groups_inst=$(printf '%s' "${winning_groups_inst}" | tr '\n,' '  ')
    echo "${PROMPT_MODE},${CORRUPTION},${stage},${round},${param1},${param1_val},${param2},${param2_val},${steps},${lr_prompt},${prompt_len},${num_bands},${lambda_gate},${lambda_f},${rho},${WEIGHT_DECAY},${INFER_SAMPLE_COUNT},${SAMPLE_LIST_SHA256},${PROTOCOL_HASH},${winning_groups_sem},${winning_groups_inst},${status},${pq},${timestamp}" >> "${RESULT_LOG}"
}

ensure_anchor() {
    local task=$1
    local current_num_bands=$2
    local current_rho=$3
    local current_prompt_len=$4
    local rho_safe
    rho_safe=$(echo "${current_rho}" | tr '.' '_')
    local anchor_file="${TEMP_DIR}/${GLOBAL_ANCHOR_PREFIX}_${PROTOCOL_HASH:0:16}_${task}_clean_prompt_plen${current_prompt_len}_nb${current_num_bands}_r${rho_safe}.pt"

    if [ -s "${anchor_file}" ]; then
        if validate_anchor_file "${anchor_file}"; then
            echo "${anchor_file}"
            return 0
        fi
        echo "[Warning] Removing invalid cached anchor: ${anchor_file}" >&2
        rm -f "${anchor_file}"
    fi
    if [ ! -d "${CLEAN_DATA_DIR}" ]; then
        echo "[Error] Clean data dir not found: ${CLEAN_DATA_DIR}" >&2
        return 1
    fi

    if ! set_prompt_path_args "${task}" "zero_shot" "${CORRUPTION}"; then
        return 1
    fi

    echo ">>> [Anchor] task=${task}, prompt=clean, plen=${current_prompt_len}, nb=${current_num_bands}, rho=${current_rho}" >&2
    local anchor_log="${TEMP_DIR}/anchor_${task}_plen${current_prompt_len}_nb${current_num_bands}_r${rho_safe}.log"
    local anchor_tmp="${anchor_file}.tmp.$$"
    rm -f "${anchor_tmp}"
    if ! "${PYTHON_BIN}" "${INFERENCE_SCRIPT}" \
      --mode anchor --task "${task}" \
      ${FC_VICT_FLAG} \
      --rho "${current_rho}" \
      --sample_list "${CLEAN_SAMPLE_LIST}" \
      --anchor_stats_path "${anchor_tmp}" \
      --ckpt_path "${CKPT_PATH}" \
      --model "${MODEL}" \
      --prompt "${PROMPT}" \
      --input_dir "dummy" \
      --save_dir "dummy" \
      --vpt_type "${VPT_TYPE}" \
      --prompt_length "${current_prompt_len}" \
      --vpt_target_layer ${TARGET_LAYERS} \
      --num_bands "${current_num_bands}" \
      --prompt_level "${SEVERITY}" \
      --weight_decay "${WEIGHT_DECAY}" \
      --seed "${RANDOM_SEED}" \
      "${PROMPT_PATH_ARGS[@]}" > "${anchor_log}" 2>&1; then
        echo "[Error] Anchor generation failed. Log: ${anchor_log}" >&2
        cat "${anchor_log}" >&2
        rm -f "${anchor_tmp}"
        return 1
    fi
    if ! validate_anchor_file "${anchor_tmp}"; then
        echo "[Error] Generated anchor failed validation: ${anchor_tmp}" >&2
        rm -f "${anchor_tmp}"
        return 1
    fi
    if ! mv "${anchor_tmp}" "${anchor_file}"; then
        echo "[Error] Failed to commit anchor: ${anchor_file}" >&2
        rm -f "${anchor_tmp}"
        return 1
    fi

    echo "${anchor_file}"
}

compute_winner() {
    local task=$1
    local current_num_bands=$2
    local current_lambda_gate=$3
    local current_lambda_f=$4
    local current_rho=$5
    local current_prompt_len=$6
    local anchor_file=$7

    local rho_safe
    local lf_safe
    rho_safe=$(echo "${current_rho}" | tr '.' '_')
    lf_safe=$(echo "${current_lambda_f}" | tr '.' '_')
    local winner_file="${TEMP_DIR}/global_winner_${PROTOCOL_HASH:0:16}_${task}_${CORRUPTION}_${PROMPT_MODE}_plen${current_prompt_len}_nb${current_num_bands}_r${rho_safe}_lg${current_lambda_gate}_lf${lf_safe}.txt"
    local winner_scores_file="${winner_file%.txt}_scores.txt"

    if validate_winner_file "${winner_file}" && validate_winner_scores_file "${winner_scores_file}"; then
        echo "${winner_file}"
        return 0
    elif [ -e "${winner_file}" ] || [ -e "${winner_scores_file}" ]; then
        echo "[Warning] Removing invalid cached winner artifacts: ${winner_file}" >&2
        rm -f "${winner_file}" "${winner_scores_file}"
    fi

    local input_dir="datasets/coco-c/${CORRUPTION}/${SEVERITY}/val2017/"
    if [ ! -d "${input_dir}" ]; then
        echo "[Error] Corruption data dir not found: ${input_dir}" >&2
        return 1
    fi

    if [ ! -s "${FIXED_SAMPLE_LIST}" ]; then
        echo "[Error] Fixed sample list is empty: ${FIXED_SAMPLE_LIST}" >&2
        return 1
    fi

    if ! set_prompt_path_args "${task}" "${PROMPT_MODE}" "${CORRUPTION}"; then
        return 1
    fi

    echo ">>> [Winner] task=${task}, mode=${PROMPT_MODE}, plen=${current_prompt_len}, nb=${current_num_bands}, lg=${current_lambda_gate}, lf=${current_lambda_f}, rho=${current_rho}" >&2
    local drift_log="${TEMP_DIR}/eval_drift_${task}_plen${current_prompt_len}_nb${current_num_bands}_lg${current_lambda_gate}_lf${lf_safe}_r${rho_safe}.log"
    local winner_tmp="${winner_file}.tmp.$$"
    local winner_scores_tmp="${winner_tmp/.txt/_scores.txt}"
    rm -f "${winner_tmp}" "${winner_scores_tmp}"
    if ! "${PYTHON_BIN}" "${INFERENCE_SCRIPT}" \
      --mode eval_drift --task "${task}" \
      ${FC_VICT_FLAG} \
      --lambda_f "${current_lambda_f}" \
      --rho "${current_rho}" \
      --sample_list "${FIXED_SAMPLE_LIST}" \
      --anchor_stats_path "${anchor_file}" \
      --calibration_path "${CALIBRATION_FILE}" \
      --global_winner_path "${winner_tmp}" \
      --ckpt_path "${CKPT_PATH}" \
      --model "${MODEL}" \
      --prompt "${PROMPT}" \
      --input_dir "${input_dir}" \
      --save_dir "dummy" \
      --vpt_type "${VPT_TYPE}" \
      --prompt_length "${current_prompt_len}" \
      --vpt_target_layer ${TARGET_LAYERS} \
      --num_bands "${current_num_bands}" \
      --lambda_gate "${current_lambda_gate}" \
      --prompt_level "${SEVERITY}" \
      --weight_decay "${WEIGHT_DECAY}" \
      --seed "${RANDOM_SEED}" \
      "${PROMPT_PATH_ARGS[@]}" > "${drift_log}" 2>&1; then
        echo "[Error] Winner voting failed. Log: ${drift_log}" >&2
        cat "${drift_log}" >&2
        rm -f "${winner_tmp}" "${winner_scores_tmp}"
        return 1
    fi
    if ! validate_winner_file "${winner_tmp}" || ! validate_winner_scores_file "${winner_scores_tmp}"; then
        echo "[Error] Generated winner failed validation: ${winner_tmp}" >&2
        rm -f "${winner_tmp}" "${winner_scores_tmp}"
        return 1
    fi
    if ! mv "${winner_scores_tmp}" "${winner_scores_file}"; then
        echo "[Error] Failed to commit winner scores: ${winner_scores_file}" >&2
        rm -f "${winner_tmp}" "${winner_scores_tmp}"
        return 1
    fi
    if ! mv "${winner_tmp}" "${winner_file}"; then
        echo "[Error] Failed to commit winner: ${winner_file}" >&2
        rm -f "${winner_tmp}" "${winner_scores_file}"
        return 1
    fi

    echo "${winner_file}"
}

run_eval() {
    local stage=$1
    local round=$2
    local param1=$3
    local param1_val=$4
    local param2=$5
    local param2_val=$6
    local current_steps=$7
    local current_lr=$8
    local current_p_len=$9
    local current_nb=${10}
    local current_lg=${11}
    local current_lf=${12}
    local current_rho=${13}

    local cached_pq
    cached_pq=$(get_cached_pq "${stage}" "${round}" "${param1}" "${param1_val}" "${param2}" "${param2_val}" "${current_steps}" "${current_lr}" "${current_p_len}" "${current_nb}" "${current_lg}" "${current_lf}" "${current_rho}")
    if [ -n "${cached_pq}" ]; then
        echo ">>> [Cache] ${CORRUPTION} ${stage}.${round} ${param1}=${param1_val} ${param2}=${param2_val} PQ=${cached_pq}" >&2
        echo "${cached_pq}"
        return 0
    fi

    echo ">>> [${CORRUPTION}] ${stage}.${round} | ${param1}=${param1_val}, ${param2}=${param2_val}, steps=${current_steps}, lr=${current_lr}, plen=${current_p_len}, nb=${current_nb}, lg=${current_lg}, lf=${current_lf}, rho=${current_rho}" >&2

    local anchor_sem
    local anchor_inst
    if ! anchor_sem=$(ensure_anchor "semseg" "${current_nb}" "${current_rho}" "${current_p_len}"); then
        append_result "${stage}" "${round}" "${param1}" "${param1_val}" "${param2}" "${param2_val}" "${current_steps}" "${current_lr}" "${current_p_len}" "${current_nb}" "${current_lg}" "${current_lf}" "${current_rho}" "N/A" "N/A" "ERROR_ANCHOR_SEM" "N/A"
        echo "N/A"
        return 0
    fi
    if ! anchor_inst=$(ensure_anchor "inst" "${current_nb}" "${current_rho}" "${current_p_len}"); then
        append_result "${stage}" "${round}" "${param1}" "${param1_val}" "${param2}" "${param2_val}" "${current_steps}" "${current_lr}" "${current_p_len}" "${current_nb}" "${current_lg}" "${current_lf}" "${current_rho}" "N/A" "N/A" "ERROR_ANCHOR_INST" "N/A"
        echo "N/A"
        return 0
    fi

    local winner_sem
    local winner_inst
    if ! winner_sem=$(compute_winner "semseg" "${current_nb}" "${current_lg}" "${current_lf}" "${current_rho}" "${current_p_len}" "${anchor_sem}"); then
        append_result "${stage}" "${round}" "${param1}" "${param1_val}" "${param2}" "${param2_val}" "${current_steps}" "${current_lr}" "${current_p_len}" "${current_nb}" "${current_lg}" "${current_lf}" "${current_rho}" "N/A" "N/A" "ERROR_WINNER_SEM" "N/A"
        echo "N/A"
        return 0
    fi
    if ! winner_inst=$(compute_winner "inst" "${current_nb}" "${current_lg}" "${current_lf}" "${current_rho}" "${current_p_len}" "${anchor_inst}"); then
        append_result "${stage}" "${round}" "${param1}" "${param1_val}" "${param2}" "${param2_val}" "${current_steps}" "${current_lr}" "${current_p_len}" "${current_nb}" "${current_lg}" "${current_lf}" "${current_rho}" "$(<"${winner_sem}")" "N/A" "ERROR_WINNER_INST" "N/A"
        echo "N/A"
        return 0
    fi
    local winner_group_sem
    local winner_group_inst
    winner_group_sem=$(tr -d '[:space:]' < "${winner_sem}")
    winner_group_inst=$(tr -d '[:space:]' < "${winner_inst}")

    local input_dir="datasets/coco-c/${CORRUPTION}/${SEVERITY}/val2017/"
    local infer_input_dir="${TEMP_DIR}/infer_inputs_${stage}_r${round}_${param1_val}_${param2_val}"
    local tag="${PROMPT_MODE}_${CORRUPTION}_${stage}_r${round}_s${current_steps}_lr${current_lr}_len${current_p_len}_nb${current_nb}_lg${current_lg}_lf${current_lf}_rho${current_rho}"
    local dst_dir_sem="${PRED_ROOT_DIR}/semseg_${tag}"
    local dst_dir_ins="${PRED_ROOT_DIR}/inst_${tag}"

    if [ ! -s "${FIXED_SAMPLE_LIST}" ]; then
        echo "[Error] Fixed inference sample list is empty: ${FIXED_SAMPLE_LIST}" >&2
        append_result "${stage}" "${round}" "${param1}" "${param1_val}" "${param2}" "${param2_val}" "${current_steps}" "${current_lr}" "${current_p_len}" "${current_nb}" "${current_lg}" "${current_lf}" "${current_rho}" "${winner_group_sem}" "${winner_group_inst}" "ERROR_SAMPLE_LIST" "N/A"
        echo "N/A"
        return 0
    fi

    rm -rf "${infer_input_dir}" "${dst_dir_sem}" "${dst_dir_ins}"
    mkdir -p "${infer_input_dir}"
    while IFS= read -r img_path; do
        ln -sf "$(realpath "${img_path}")" "${infer_input_dir}/$(basename "${img_path}")"
    done < "${FIXED_SAMPLE_LIST}"

    local sem_log="${TEMP_DIR}/semseg_${stage}_r${round}_${param1_val}_${param2_val}.log"
    if ! set_prompt_path_args "semseg" "${PROMPT_MODE}" "${CORRUPTION}"; then
        append_result "${stage}" "${round}" "${param1}" "${param1_val}" "${param2}" "${param2_val}" "${current_steps}" "${current_lr}" "${current_p_len}" "${current_nb}" "${current_lg}" "${current_lf}" "${current_rho}" "${winner_group_sem}" "${winner_group_inst}" "ERROR_PROMPT_SEM" "N/A"
        echo "N/A"
        return 0
    fi
    if ! "${PYTHON_BIN}" "${INFERENCE_SCRIPT}" \
      --mode inference --task semseg \
      ${FC_VICT_FLAG} --lambda_f "${current_lf}" --rho "${current_rho}" \
      --global_winner_path "${winner_sem}" \
      --ckpt_path "${CKPT_PATH}" \
      --model "${MODEL}" \
      --prompt "${PROMPT}" \
      --input_dir "${infer_input_dir}" \
      --save_dir "${dst_dir_sem}" \
      --steps "${current_steps}" \
      --lr "${current_lr}" \
      --lr_backbone "${LR_BACKBONE}" \
      --vpt_type "${VPT_TYPE}" \
      --prompt_length "${current_p_len}" \
      --vpt_target_layer ${TARGET_LAYERS} \
      --num_bands "${current_nb}" \
      --prompt_level "${SEVERITY}" \
      --weight_decay "${WEIGHT_DECAY}" \
      --seed "${RANDOM_SEED}" \
      "${PROMPT_PATH_ARGS[@]}" > "${sem_log}" 2>&1; then
        echo "[Error] SemSeg inference failed. Log: ${sem_log}" >&2
        cat "${sem_log}" >&2
        append_result "${stage}" "${round}" "${param1}" "${param1_val}" "${param2}" "${param2_val}" "${current_steps}" "${current_lr}" "${current_p_len}" "${current_nb}" "${current_lg}" "${current_lf}" "${current_rho}" "${winner_group_sem}" "${winner_group_inst}" "ERROR_INFERENCE_SEM" "N/A"
        rm -rf "${infer_input_dir}" "${dst_dir_sem}" "${dst_dir_ins}"
        echo "N/A"
        return 0
    fi

    local ins_log="${TEMP_DIR}/inst_${stage}_r${round}_${param1_val}_${param2_val}.log"
    if ! set_prompt_path_args "inst" "${PROMPT_MODE}" "${CORRUPTION}"; then
        append_result "${stage}" "${round}" "${param1}" "${param1_val}" "${param2}" "${param2_val}" "${current_steps}" "${current_lr}" "${current_p_len}" "${current_nb}" "${current_lg}" "${current_lf}" "${current_rho}" "${winner_group_sem}" "${winner_group_inst}" "ERROR_PROMPT_INST" "N/A"
        echo "N/A"
        return 0
    fi
    if ! "${PYTHON_BIN}" "${INFERENCE_SCRIPT}" \
      --mode inference --task inst \
      ${FC_VICT_FLAG} --lambda_f "${current_lf}" --rho "${current_rho}" \
      --global_winner_path "${winner_inst}" \
      --ckpt_path "${CKPT_PATH}" \
      --model "${MODEL}" \
      --prompt "${PROMPT}" \
      --input_dir "${infer_input_dir}" \
      --save_dir "${dst_dir_ins}" \
      --steps "${current_steps}" \
      --lr "${current_lr}" \
      --lr_backbone "${LR_BACKBONE}" \
      --vpt_type "${VPT_TYPE}" \
      --prompt_length "${current_p_len}" \
      --vpt_target_layer ${TARGET_LAYERS} \
      --num_bands "${current_nb}" \
      --prompt_level "${SEVERITY}" \
      --weight_decay "${WEIGHT_DECAY}" \
      --seed "${RANDOM_SEED}" \
      "${PROMPT_PATH_ARGS[@]}" > "${ins_log}" 2>&1; then
        echo "[Error] Inst inference failed. Log: ${ins_log}" >&2
        cat "${ins_log}" >&2
        append_result "${stage}" "${round}" "${param1}" "${param1_val}" "${param2}" "${param2_val}" "${current_steps}" "${current_lr}" "${current_p_len}" "${current_nb}" "${current_lg}" "${current_lf}" "${current_rho}" "${winner_group_sem}" "${winner_group_inst}" "ERROR_INFERENCE_INST" "N/A"
        rm -rf "${infer_input_dir}" "${dst_dir_sem}" "${dst_dir_ins}"
        echo "N/A"
        return 0
    fi

    local prediction_validation_log="${TEMP_DIR}/prediction_validation_${stage}_r${round}_${param1_val}_${param2_val}.log"
    if ! {
        validate_prediction_ids "${dst_dir_sem}"
        validate_prediction_ids "${dst_dir_ins}"
    } > "${prediction_validation_log}" 2>&1; then
        echo "[Error] Prediction count/ID validation failed. Log: ${prediction_validation_log}" >&2
        cat "${prediction_validation_log}" >&2
        append_result "${stage}" "${round}" "${param1}" "${param1_val}" "${param2}" "${param2_val}" "${current_steps}" "${current_lr}" "${current_p_len}" "${current_nb}" "${current_lg}" "${current_lf}" "${current_rho}" "${winner_group_sem}" "${winner_group_inst}" "ERROR_OUTPUT_VALIDATION" "N/A"
        rm -rf "${infer_input_dir}" "${dst_dir_sem}" "${dst_dir_ins}"
        echo "N/A"
        return 0
    fi

    local eval_stdout_log="${TEMP_DIR}/eval_stdout_${stage}_r${round}_${param1_val}_${param2_val}.log"
    local eval_stderr_log="${TEMP_DIR}/eval_stderr_${stage}_r${round}_${param1_val}_${param2_val}.log"
    local eval_status=0
    "${PYTHON_BIN}" "${EVALUATOR_SCRIPT}" \
      --pred_path_sem "${dst_dir_sem}" \
      --pred_path_inst "${dst_dir_ins}" \
      --gt_path "${GT_ANNOTATIONS}" \
      --dist_thr 19 \
      --prompt "${PROMPT}" \
      --ckpt_file "${CKPT_FILE##*/}" \
      --input_size 448 > "${eval_stdout_log}" 2> "${eval_stderr_log}" || eval_status=$?

    local pq="N/A"
    if [ "${eval_status}" -eq 0 ]; then
        local output
        output=$(<"${eval_stdout_log}")
        pq=$(echo "$output" | awk -F'|' '/^PQ[[:space:]]*\|/ {gsub(/[[:space:]]/, "", $2); print $2; exit}')
        [ -z "${pq}" ] && pq="N/A"
    else
        echo "[Error] Evaluation failed. Logs: ${eval_stdout_log}, ${eval_stderr_log}" >&2
        cat "${eval_stdout_log}" >&2
        cat "${eval_stderr_log}" >&2
    fi

    local status="ERROR_EVALUATION"
    if is_number "${pq}"; then
        status="OK"
    else
        pq="N/A"
    fi
    append_result "${stage}" "${round}" "${param1}" "${param1_val}" "${param2}" "${param2_val}" "${current_steps}" "${current_lr}" "${current_p_len}" "${current_nb}" "${current_lg}" "${current_lf}" "${current_rho}" "${winner_group_sem}" "${winner_group_inst}" "${status}" "${pq}"
    echo ">>> [${CORRUPTION}] ${stage}.${round} PQ=${pq}" >&2

    rm -rf "${infer_input_dir}" "${dst_dir_sem}" "${dst_dir_ins}"
    echo "${pq}"
}

require_round_best() {
    local stage=$1
    local round=$2
    local best_pq=$3
    if ! is_number "${best_pq}"; then
        echo "[Error] ${CORRUPTION} ${stage} Round ${round}: no valid PQ result; stop without fallback." >&2
        return 1
    fi
    return 0
}

tune_corruption() {
    CORRUPTION=$(normalize_corruption "$1")
    if [ "${PROMPT_MODE}" != "zero_shot" ] && [ "${PROMPT_MODE}" != "one_shot" ]; then
        echo "[Error] PROMPT_MODE must be zero_shot or one_shot: ${PROMPT_MODE}" >&2
        return 1
    fi
    if [ "${INFER_SAMPLE_COUNT}" -ne 50 ] && [ "${ALLOW_NON_100}" != "1" ]; then
        echo "[Error] Formal tuning requires exactly 50 images. INFER_SAMPLE_COUNT=${INFER_SAMPLE_COUNT}" >&2
        return 1
    fi

    TEMP_DIR="${RUN_ROOT}/work/${CORRUPTION}/${PROMPT_MODE}"
    PRED_ROOT_DIR="${RUN_ROOT}/predictions/${CORRUPTION}/${PROMPT_MODE}"
    RESULT_LOG="${RUN_ROOT}/results/results_coco_${CORRUPTION}_${PROMPT_MODE}_tuning.csv"
    FIXED_SAMPLE_LIST="${RUN_ROOT}/sample_lists/${CORRUPTION}_seed${RANDOM_SEED}_${INFER_SAMPLE_COUNT}.txt"
    CLEAN_SAMPLE_LIST="${RUN_ROOT}/sample_lists/${CORRUPTION}_clean_ids_seed${RANDOM_SEED}_${INFER_SAMPLE_COUNT}.txt"
    export TMPDIR="${PROJECT_ROOT}/${TEMP_DIR}/tmpdir"
    mkdir -p "${TEMP_DIR}" "${PRED_ROOT_DIR}" "${TMPDIR}" "${RUN_ROOT}/results" "${RUN_ROOT}/sample_lists" "${RUN_ROOT}/logs"
    exec 8>"${TEMP_DIR}/job.lock"
    if ! flock -n 8; then
        echo "[Error] Another tuner is already running for ${CORRUPTION}/${PROMPT_MODE}." >&2
        return 1
    fi

    local required
    for required in "${INFERENCE_SCRIPT}" "${EVALUATOR_SCRIPT}" "${CKPT_PATH}" \
        "${GT_ANNOTATIONS}" "models_nyu.py" "models_painter.py"; do
        if [ ! -f "${required}" ]; then
            echo "[Error] Missing required file: ${required}" >&2
            return 1
        fi
    done

    local input_dir="datasets/coco-c/${CORRUPTION}/${SEVERITY}/val2017"
    local all_input_list="${TEMP_DIR}/all_input_images.txt"
    find "${input_dir}" -type f \( -name "*.jpg" -o -name "*.png" \) | sort > "${all_input_list}"
    if [ ! -s "${all_input_list}" ] || [ ! -d "${CLEAN_DATA_DIR}" ]; then
        echo "[Error] Missing COCO input data for ${CORRUPTION}" >&2
        return 1
    fi
    if [ ! -s "${FIXED_SAMPLE_LIST}" ]; then
        local sample_tmp="${FIXED_SAMPLE_LIST}.$$"
        sample_file_list "${all_input_list}" "${sample_tmp}" "${INFER_SAMPLE_COUNT}" "${RANDOM_SEED}"
        mv "${sample_tmp}" "${FIXED_SAMPLE_LIST}"
    fi
    if [ ! -s "${CLEAN_SAMPLE_LIST}" ]; then
        local clean_tmp="${CLEAN_SAMPLE_LIST}.$$"
        "${PYTHON_BIN}" - "${FIXED_SAMPLE_LIST}" "${CLEAN_DATA_DIR}" "${clean_tmp}" <<'PY'
import pathlib
import sys

sample_list, clean_dir, output = sys.argv[1:]
clean_dir = pathlib.Path(clean_dir)
with open(sample_list, "r", encoding="utf-8") as source, open(output, "w", encoding="utf-8") as target:
    for line in source:
        if not line.strip():
            continue
        clean_path = clean_dir / pathlib.Path(line.strip()).name
        if not clean_path.is_file():
            raise SystemExit(f"missing clean counterpart: {clean_path}")
        target.write(str(clean_path) + "\n")
PY
        mv "${clean_tmp}" "${CLEAN_SAMPLE_LIST}"
    fi
    if [ "$(wc -l < "${FIXED_SAMPLE_LIST}" | tr -d ' ')" -ne "${INFER_SAMPLE_COUNT}" ] || \
       [ "$(wc -l < "${CLEAN_SAMPLE_LIST}" | tr -d ' ')" -ne "${INFER_SAMPLE_COUNT}" ]; then
        echo "[Error] Fixed/clean sample list count mismatch." >&2
        return 1
    fi
    "${PYTHON_BIN}" - "${FIXED_SAMPLE_LIST}" "${CLEAN_SAMPLE_LIST}" "${INFER_SAMPLE_COUNT}" <<'PY'
import pathlib
import sys

corrupt_list, clean_list, expected_count = sys.argv[1:]
expected_count = int(expected_count)
def ids(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [pathlib.Path(line.strip()).stem for line in handle if line.strip()]
corrupt_ids = ids(corrupt_list)
clean_ids = ids(clean_list)
if (
    len(corrupt_ids) != expected_count
    or len(set(corrupt_ids)) != expected_count
    or corrupt_ids != clean_ids
):
    raise SystemExit("fixed corrupted and clean sample IDs do not match exactly")
PY
    SAMPLE_LIST_SHA256=$(sha256sum "${FIXED_SAMPLE_LIST}" | awk '{print $1}')
    CLEAN_SAMPLE_LIST_SHA256=$(sha256sum "${CLEAN_SAMPLE_LIST}" | awk '{print $1}')

    set_prompt_path_args "semseg" "${PROMPT_MODE}" "${CORRUPTION}" || return 1
    local sem_prompt_image="${PROMPT_PATH_ARGS[1]}"
    local sem_prompt_target="${PROMPT_PATH_ARGS[3]}"
    set_prompt_path_args "inst" "${PROMPT_MODE}" "${CORRUPTION}" || return 1
    local inst_prompt_image="${PROMPT_PATH_ARGS[1]}"
    local inst_prompt_target="${PROMPT_PATH_ARGS[3]}"

    CKPT_SHA256="${CKPT_SHA256:-$(sha256sum "${CKPT_PATH}" | awk '{print $1}')}"
    PROTOCOL_HASH=$(
        {
            sha256sum "${BASH_SOURCE[0]}" "${INFERENCE_SCRIPT}" "${EVALUATOR_SCRIPT}" \
                "${FIXED_SAMPLE_LIST}" "${CLEAN_SAMPLE_LIST}" "${GT_ANNOTATIONS}" \
                "${sem_prompt_image}" "${sem_prompt_target}" \
                "${inst_prompt_image}" "${inst_prompt_target}" "models_nyu.py" "models_painter.py"
            printf 'checkpoint_sha256=%s\n' "${CKPT_SHA256}"
            printf 'mode=%s\ncorruption=%s\nseverity=%s\nseed=%s\nsample_count=%s\ndrift_count=%s\nsteps=%s\nweight_decay=%s\n' \
                "${PROMPT_MODE}" "${CORRUPTION}" "${SEVERITY}" "${RANDOM_SEED}" \
                "${INFER_SAMPLE_COUNT}" "${DRIFT_SAMPLE_COUNT}" "${STEPS}" "${WEIGHT_DECAY}"
        } | sha256sum | awk '{print $1}'
    )
    {
        echo "protocol_hash=${PROTOCOL_HASH}"
        echo "prompt_mode=${PROMPT_MODE}"
        echo "corruption=${CORRUPTION}"
        echo "sample_list=${FIXED_SAMPLE_LIST}"
        echo "sample_list_sha256=${SAMPLE_LIST_SHA256}"
        echo "clean_sample_list=${CLEAN_SAMPLE_LIST}"
        echo "clean_sample_list_sha256=${CLEAN_SAMPLE_LIST_SHA256}"
        echo "sample_count=${INFER_SAMPLE_COUNT}"
        echo "steps=${STEPS}"
        echo "weight_decay=${WEIGHT_DECAY}"
        echo "checkpoint_sha256=${CKPT_SHA256}"
        echo "calibration_path=${CALIBRATION_FILE} (argument currently unused by inference code)"
        sha256sum "${BASH_SOURCE[0]}" "${INFERENCE_SCRIPT}" "${EVALUATOR_SCRIPT}" \
            "${FIXED_SAMPLE_LIST}" "${CLEAN_SAMPLE_LIST}" "${GT_ANNOTATIONS}" \
            "${sem_prompt_image}" "${sem_prompt_target}" \
            "${inst_prompt_image}" "${inst_prompt_target}" "models_nyu.py" "models_painter.py"
    } > "${TEMP_DIR}/protocol.txt"

    local expected_header="prompt_mode,corruption,stage,round,param1,param1_val,param2,param2_val,steps,lr_prompt,prompt_len,num_bands,lambda_gate,lambda_f,rho,weight_decay,sample_count,sample_list_sha256,protocol_hash,winning_group_sem,winning_group_inst,status,pq,timestamp"
    if [ ! -f "${RESULT_LOG}" ]; then
        echo "${expected_header}" > "${RESULT_LOG}"
    elif [ "$(head -n 1 "${RESULT_LOG}")" != "${expected_header}" ]; then
        echo "[Error] Refusing incompatible result CSV: ${RESULT_LOG}" >&2
        return 1
    fi

    local best_lambda_gate=0.3
    local best_num_bands=6
    local best_rho=0.10
    local best_lambda_f=0.01
    local best_lr=2e-5
    local best_prompt_len=30

    echo "========================================================="
    echo ">>> COCO ${CORRUPTION} ${PROMPT_MODE} 4-stage auto-tune"
    echo ">>> Result CSV: ${RESULT_LOG}"
    echo ">>> STEPS=${STEPS} fixed"
    echo ">>> Fixed sample list: ${FIXED_SAMPLE_LIST}"
    echo ">>> Sample SHA256: ${SAMPLE_LIST_SHA256}"
    echo ">>> Protocol SHA256: ${PROTOCOL_HASH}"
    echo "========================================================="

    if [ "${SMOKE_TEST_ONLY}" = "1" ]; then
        local smoke_pq
        smoke_pq=$(run_eval "SMOKE" "1" "lambda_gate" "0.3" "num_bands" "6" \
            "${STEPS}" "2e-5" "30" "6" "0.3" "0.01" "0.10")
        if ! is_number "${smoke_pq}"; then
            echo "[Error] Smoke candidate did not produce a finite PQ." >&2
            return 1
        fi
        echo ">>> Smoke candidate completed: PQ=${smoke_pq}"
        return 0
    fi

    local lg_values=(0.2 0.3 0.4)
    local nb_values=(4 6 8)
    local lg_step=0.1
    local nb_step=2
    local stage_round pq round_best_pq round_best_lg round_best_nb

    for stage_round in 1 2 3; do
        echo ">>> Stage 1 Round ${stage_round}: LAMBDA_GATE × NUM_BANDS"
        echo ">>> Current range: LAMBDA_GATE=[${lg_values[*]}], NUM_BANDS=[${nb_values[*]}]"
        round_best_pq="N/A"
        round_best_lg="${best_lambda_gate}"
        round_best_nb="${best_num_bands}"
        for lg in "${lg_values[@]}"; do
            for nb in "${nb_values[@]}"; do
                pq=$(run_eval "S1" "${stage_round}" "lambda_gate" "${lg}" "num_bands" "${nb}" "${STEPS}" "${best_lr}" "${best_prompt_len}" "${nb}" "${lg}" "${best_lambda_f}" "${best_rho}")
                if is_number "${pq}"; then
                    if ! is_number "${round_best_pq}" || [ "$(float_gt "${pq}" "${round_best_pq}")" = "1" ]; then
                        round_best_pq="${pq}"
                        round_best_lg="${lg}"
                        round_best_nb="${nb}"
                    fi
                fi
            done
        done
        require_round_best "Stage 1" "${stage_round}" "${round_best_pq}" || return 1
        best_lambda_gate="${round_best_lg}"
        best_num_bands="${round_best_nb}"
        echo ">>> Stage 1 Round ${stage_round} best: LAMBDA_GATE=${best_lambda_gate}, NUM_BANDS=${best_num_bands}, PQ=${round_best_pq}"
        lg_step=$(next_half_step "${best_lambda_gate}" "${lg_values[@]}")
        nb_step=$(next_half_step "${best_num_bands}" "${nb_values[@]}")
        read -ra lg_values <<< "$(shrink_float_range "${best_lambda_gate}" "${lg_step}" 0.05 0.95)"
        read -ra nb_values <<< "$(shrink_int_range "${best_num_bands}" "${nb_step}" 1 20)"
        echo ">>> Next range: LAMBDA_GATE=[${lg_values[*]}], NUM_BANDS=[${nb_values[*]}]"
    done

    local rho_values=(0.05 0.10 0.15)
    local lf_values=(0.005 0.01 0.05)
    local rho_step=0.05
    local lf_step=0.005
    local round_best_rho round_best_lf

    for stage_round in 1 2 3; do
        echo ">>> Stage 2 Round ${stage_round}: RHO × LAMBDA_F"
        echo ">>> Current range: RHO=[${rho_values[*]}], LAMBDA_F=[${lf_values[*]}]"
        round_best_pq="N/A"
        round_best_rho="${best_rho}"
        round_best_lf="${best_lambda_f}"
        for rho in "${rho_values[@]}"; do
            for lf in "${lf_values[@]}"; do
                pq=$(run_eval "S2" "${stage_round}" "rho" "${rho}" "lambda_f" "${lf}" "${STEPS}" "${best_lr}" "${best_prompt_len}" "${best_num_bands}" "${best_lambda_gate}" "${lf}" "${rho}")
                if is_number "${pq}"; then
                    if ! is_number "${round_best_pq}" || [ "$(float_gt "${pq}" "${round_best_pq}")" = "1" ]; then
                        round_best_pq="${pq}"
                        round_best_rho="${rho}"
                        round_best_lf="${lf}"
                    fi
                fi
            done
        done
        require_round_best "Stage 2" "${stage_round}" "${round_best_pq}" || return 1
        best_rho="${round_best_rho}"
        best_lambda_f="${round_best_lf}"
        echo ">>> Stage 2 Round ${stage_round} best: RHO=${best_rho}, LAMBDA_F=${best_lambda_f}, PQ=${round_best_pq}"
        rho_step=$(next_half_step "${best_rho}" "${rho_values[@]}")
        lf_step=$(next_half_step "${best_lambda_f}" "${lf_values[@]}")
        read -ra rho_values <<< "$(shrink_float_range "${best_rho}" "${rho_step}" 0.005 0.50)"
        read -ra lf_values <<< "$(shrink_float_range "${best_lambda_f}" "${lf_step}" 0.0005 0.20)"
        echo ">>> Next range: RHO=[${rho_values[*]}], LAMBDA_F=[${lf_values[*]}]"
    done

    local lr_values=(1e-5 2e-5 4e-5)
    local lr_step=1e-5
    local round_best_lr

    for stage_round in 1 2 3; do
        echo ">>> Stage 3 Round ${stage_round}: LR_PROMPT"
        echo ">>> Current range: LR_PROMPT=[${lr_values[*]}]"
        round_best_pq="N/A"
        round_best_lr="${best_lr}"
        for lr in "${lr_values[@]}"; do
            pq=$(run_eval "S3" "${stage_round}" "lr_prompt" "${lr}" "-" "-" "${STEPS}" "${lr}" "${best_prompt_len}" "${best_num_bands}" "${best_lambda_gate}" "${best_lambda_f}" "${best_rho}")
            if is_number "${pq}"; then
                if ! is_number "${round_best_pq}" || [ "$(float_gt "${pq}" "${round_best_pq}")" = "1" ]; then
                    round_best_pq="${pq}"
                    round_best_lr="${lr}"
                fi
            fi
        done
        require_round_best "Stage 3" "${stage_round}" "${round_best_pq}" || return 1
        best_lr="${round_best_lr}"
        echo ">>> Stage 3 Round ${stage_round} best: LR_PROMPT=${best_lr}, PQ=${round_best_pq}"
        lr_step=$(next_half_step "${best_lr}" "${lr_values[@]}")
        read -ra lr_values <<< "$(shrink_float_range "${best_lr}" "${lr_step}" 1e-6 1e-4)"
        echo ">>> Next range: LR_PROMPT=[${lr_values[*]}]"
    done

    local plen_values=(20 30 40)
    local plen_step=10
    local round_best_plen

    for stage_round in 1 2 3; do
        echo ">>> Stage 4 Round ${stage_round}: PROMPT_LEN"
        echo ">>> Current range: PROMPT_LEN=[${plen_values[*]}]"
        round_best_pq="N/A"
        round_best_plen="${best_prompt_len}"
        for plen in "${plen_values[@]}"; do
            pq=$(run_eval "S4" "${stage_round}" "prompt_len" "${plen}" "-" "-" "${STEPS}" "${best_lr}" "${plen}" "${best_num_bands}" "${best_lambda_gate}" "${best_lambda_f}" "${best_rho}")
            if is_number "${pq}"; then
                if ! is_number "${round_best_pq}" || [ "$(float_gt "${pq}" "${round_best_pq}")" = "1" ]; then
                    round_best_pq="${pq}"
                    round_best_plen="${plen}"
                fi
            fi
        done
        require_round_best "Stage 4" "${stage_round}" "${round_best_pq}" || return 1
        best_prompt_len="${round_best_plen}"
        echo ">>> Stage 4 Round ${stage_round} best: PROMPT_LEN=${best_prompt_len}, PQ=${round_best_pq}"
        plen_step=$(next_half_step "${best_prompt_len}" "${plen_values[@]}")
        read -ra plen_values <<< "$(shrink_int_range "${best_prompt_len}" "${plen_step}" 10 200)"
        echo ">>> Next range: PROMPT_LEN=[${plen_values[*]}]"
    done

    if ! "${PYTHON_BIN}" - "${RESULT_LOG}" "${PROTOCOL_HASH}" <<'PY'
import csv
import math
import sys

result_path, protocol_hash = sys.argv[1:]
expected_per_round = {"S1": 9, "S2": 9, "S3": 3, "S4": 3}
completed = {f"{stage}:{round_number}": set() for stage in expected_per_round for round_number in range(1, 4)}
with open(result_path, newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        if row.get("protocol_hash") != protocol_hash or row.get("status") != "OK":
            continue
        try:
            pq = float(row.get("pq", ""))
        except Exception:
            continue
        if not math.isfinite(pq):
            continue
        key = f"{row.get('stage')}:{row.get('round')}"
        if key not in completed:
            continue
        candidate_key = tuple(
            row.get(field, "")
            for field in (
                "param1", "param1_val", "param2", "param2_val", "steps",
                "lr_prompt", "prompt_len", "num_bands", "lambda_gate",
                "lambda_f", "rho", "sample_list_sha256",
            )
        )
        completed[key].add(candidate_key)
problems = {
    key: len(values)
    for key, values in completed.items()
    if len(values) != expected_per_round[key.split(":")[0]]
}
if problems:
    raise SystemExit(f"incomplete 72-candidate protocol: {problems}")
print("validated_completed_candidates=72")
PY
    then
        echo "[Error] Refusing to complete an incomplete 72-candidate protocol." >&2
        return 1
    fi

    echo "========================================================="
    echo ">>> Complete: ${CORRUPTION} ${PROMPT_MODE}"
    echo ">>> Best LAMBDA_GATE=${best_lambda_gate}"
    echo ">>> Best NUM_BANDS=${best_num_bands}"
    echo ">>> Best RHO=${best_rho}"
    echo ">>> Best LAMBDA_F=${best_lambda_f}"
    echo ">>> Best STEPS=${STEPS}"
    echo ">>> Best LR_PROMPT=${best_lr}"
    echo ">>> Best PROMPT_LEN=${best_prompt_len}"
    echo ">>> Best PQ=${round_best_pq}"
    echo ">>> Results: ${RESULT_LOG}"
    echo "========================================================="
    {
        echo "corruption=${CORRUPTION}"
        echo "prompt_mode=${PROMPT_MODE}"
        echo "sample_count=${INFER_SAMPLE_COUNT}"
        echo "sample_list_sha256=${SAMPLE_LIST_SHA256}"
        echo "clean_sample_list_sha256=${CLEAN_SAMPLE_LIST_SHA256}"
        echo "protocol_hash=${PROTOCOL_HASH}"
        echo "lambda_gate=${best_lambda_gate}"
        echo "num_bands=${best_num_bands}"
        echo "rho=${best_rho}"
        echo "lambda_f=${best_lambda_f}"
        echo "steps=${STEPS}"
        echo "lr_prompt=${best_lr}"
        echo "prompt_len=${best_prompt_len}"
        echo "weight_decay=${WEIGHT_DECAY}"
        echo "pq=${round_best_pq}"
    } > "${RUN_ROOT}/results/best_${CORRUPTION}_${PROMPT_MODE}.txt"
    return 0
}

if [ "$#" -eq 0 ]; then
    echo "Usage: PROMPT_MODE=zero_shot|one_shot bash ${BASH_SOURCE[0]} <corruption> [...]" >&2
    exit 1
fi
CORRUPTIONS=("$@")

mkdir -p "${RUN_ROOT}/logs"
for raw_corruption in "${CORRUPTIONS[@]}"; do
    tune_corruption "${raw_corruption}"
    echo ">>> Finished: $(normalize_corruption "${raw_corruption}")"
done
