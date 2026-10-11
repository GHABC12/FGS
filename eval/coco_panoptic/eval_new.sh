#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

# =========================================================
# 环境配置
# =========================================================
export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL=3
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
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
TEMP_DIR="${TEMP_DIR:-temp/coco_panoptic/${RUN_ID}}"
export TMPDIR="${ROOT_DIR}/${TEMP_DIR}/tmpdir"
mkdir -p "${TMPDIR}"

# =========================================================
# 配置区
# =========================================================
CKPT_FILE="painter_vit_large/painter_vit_large.pth"
PROMPT="000000391460"
MODEL="painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1"
CKPT_PATH="models/${CKPT_FILE}"
INFERENCE_SCRIPT="eval/coco_panoptic/painter_inference_coco_zero_shot_correct_prompt.py"
VPT_TYPE="custom"
PROMPT_LEN_LIST=(30)
TARGET_LAYERS="0 1 2 10 11 12 21 22 23"
SEVERITY=5
RESULT_LOG="${RESULT_LOG:-results_coco_new_${RUN_ID}.csv}"
PRED_ROOT_DIR="${PRED_ROOT_DIR:-models_inference/vpt_results_coco_new/${RUN_ID}}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
KEEP_OUTPUTS="${KEEP_OUTPUTS:-1}"
KEEP_TEMP="${KEEP_TEMP:-1}"
CONFIG_CHECK_ONLY="${CONFIG_CHECK_ONLY:-0}"
VALIDATED_GAUSSIAN_WINNER_GROUP="${VALIDATED_GAUSSIAN_WINNER_GROUP:-group1}"

# =========================================================
# 提示图片模式
# zero_shot: 提示图片使用 clean 版本
# one_shot:  提示图片使用对应的损坏版本
# =========================================================
PROMPT_MODE_LIST=("zero_shot")

# =========================================================
# FC-VICT 核心超参数配置
# =========================================================
FC_VICT_FLAG="--use_fc_vict"
# 频域正则化 Loss 权重，低频混合权重
#0.01 0.5
LAMBDA_F_LIST=(0.05)
RHO_LIST=(0.05)

# 5 0.3 再试几组
NUM_BANDS_LIST=(2)
LAMBDA_GATE_LIST=(0.375)
#3 3e-5
STEPS_LIST=(10)
LR_PROMPT_LIST=(2e-5)
LR_BACKBONE=(0)
INFER_SAMPLE_COUNT="${SAMPLE_COUNT:-100}"
DRIFT_SAMPLE_COUNT="${DRIFT_SAMPLE_COUNT:-50}"
RANDOM_SEED=20260426

CORRUPTIONS=(
"gaussian_noise"
)
if [ -n "${CORRUPTIONS_OVERRIDE:-}" ]; then
    read -r -a CORRUPTIONS <<< "${CORRUPTIONS_OVERRIDE}"
fi

CLEAN_DATA_DIR="datasets/coco/val2017/"
GLOBAL_ANCHOR_PREFIX="anchor_stats_global_clean_coco_correct_prompt_v2"
CALIBRATION_FILE="calibration_baseline_coco.pt"

mkdir -p "${TEMP_DIR}"

declare -a PROMPT_PATH_ARGS=()

set_prompt_path_args() {
    local task=$1
    local prompt_mode=$2
    local corruption=${3:-}
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
            echo "[Error] Unsupported task for prompt resolution: ${task}" >&2
            return 1
            ;;
    esac

    if [ "${prompt_mode}" = "one_shot" ]; then
        if [ -z "${corruption}" ]; then
            echo "[Error] one_shot requires a corruption name." >&2
            return 1
        fi
        prompt_image="datasets/coco/corrupted_prompts/${corruption}/${SEVERITY}/${PROMPT}.jpg"
    elif [ "${prompt_mode}" != "zero_shot" ]; then
        echo "[Error] Unsupported prompt mode: ${prompt_mode}" >&2
        return 1
    fi

    if [ ! -f "${prompt_image}" ]; then
        echo "[Error] Missing prompt image: ${prompt_image}" >&2
        return 1
    fi
    if [ ! -f "${prompt_target}" ]; then
        echo "[Error] Missing prompt target: ${prompt_target}" >&2
        return 1
    fi

    PROMPT_PATH_ARGS=(
        --prompt_image_path "${prompt_image}"
        --prompt_target_path "${prompt_target}"
    )
}

validate_entry_config() {
    if [ ! -f "${INFERENCE_SCRIPT}" ]; then
        echo "[Error] Missing inference script: ${INFERENCE_SCRIPT}" >&2
        return 1
    fi
    if [ ! -f "${CKPT_PATH}" ]; then
        echo "[Error] Missing checkpoint: ${CKPT_PATH}" >&2
        return 1
    fi

    local prompt_mode
    local corruption
    local task
    for prompt_mode in "${PROMPT_MODE_LIST[@]}"; do
        for corruption in "${CORRUPTIONS[@]}"; do
            for task in semseg inst; do
                set_prompt_path_args "${task}" "${prompt_mode}" "${corruption}"
                echo "[Config] ${task}/${prompt_mode}/${corruption}: ${PROMPT_PATH_ARGS[*]}"
            done
        done
    done
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

run_anchor_if_needed() {
    local task=$1
    local anchor_path=$2
    local num_bands=$3
    local current_rho=$4
    local current_p_len=$5
    local clean_sample_list=$6

    if [ ! -f "${anchor_path}" ]; then
        echo ">>> [Stage 0] Generating ${task} Global Clean Anchor Stats (len=${current_p_len}, num_bands=${num_bands}, rho=${current_rho})..."
        set_prompt_path_args "${task}" "zero_shot" ""
        "${PYTHON_BIN}" "${INFERENCE_SCRIPT}" \
          --mode anchor --task "${task}" \
          ${FC_VICT_FLAG} \
          --rho "${current_rho}" \
          --sample_list "${clean_sample_list}" \
          --anchor_stats_path "${anchor_path}" \
          --ckpt_path "${CKPT_PATH}" --model "${MODEL}" --prompt "${PROMPT}" \
          --input_dir "dummy" --save_dir "dummy" \
          --vpt_type "${VPT_TYPE}" --prompt_length "${current_p_len}" --vpt_target_layer ${TARGET_LAYERS} \
          --num_bands "${num_bands}" \
          --prompt_level "${SEVERITY}" \
          --weight_decay "${WEIGHT_DECAY}" \
          --seed "${RANDOM_SEED}" \
          "${PROMPT_PATH_ARGS[@]}"
    else
        echo ">>> [Stage 0] ${task} Global Clean Anchor Stats already exists: ${anchor_path}"
    fi
}

run_eval_drift_for_task() {
    local task=$1
    local anchor_path=$2
    local winner_path=$3
    local corrupt_sample_list=$4
    local input_dir=$5
    local current_p_len=$6
    local num_bands=$7
    local lambda_gate=$8
    local current_lambda_f=$9
    local current_rho=${10}
    local prompt_mode=${11}
    local corruption=${12}

    echo ">>> [Stage 2 & 3] Voting Winner Group for ${task}..."
    set_prompt_path_args "${task}" "${prompt_mode}" "${corruption}"
    "${PYTHON_BIN}" "${INFERENCE_SCRIPT}" \
      --mode eval_drift --task "${task}" \
      ${FC_VICT_FLAG} \
      --lambda_f "${current_lambda_f}" \
      --rho "${current_rho}" \
      --sample_list "${corrupt_sample_list}" \
      --anchor_stats_path "${anchor_path}" \
      --calibration_path "${CALIBRATION_FILE}" \
      --global_winner_path "${winner_path}" \
      --ckpt_path "${CKPT_PATH}" \
      --model "${MODEL}" \
      --prompt "${PROMPT}" \
      --input_dir "${input_dir}" \
      --save_dir "dummy" \
      --vpt_type "${VPT_TYPE}" \
      --prompt_length "${current_p_len}" \
      --vpt_target_layer ${TARGET_LAYERS} \
      --num_bands "${num_bands}" \
      --lambda_gate "${lambda_gate}" \
      --prompt_level "${SEVERITY}" \
      --weight_decay "${WEIGHT_DECAY}" \
      --seed "${RANDOM_SEED}" \
      "${PROMPT_PATH_ARGS[@]}"
}

# =========================================================
# 推理执行函数
# =========================================================
do_inference() {
    local corruption=$1
    local steps=$2
    local lr_prompt=$3
    local lr_backbone=$4
    local current_layers=$5
    local current_p_len=$6
    local global_winner_sem_file=$7
    local global_winner_inst_file=$8
    local current_num_bands=$9
    local current_lambda_gate=${10}
    local current_lambda_f=${11}
    local current_rho=${12}
    local current_prompt_mode=${13}

    local INPUT_DIR="datasets/coco-c/${corruption}/${SEVERITY}/val2017/"
    local INFER_ALL_LIST="${TEMP_DIR}/all_imgs_infer_${corruption}.txt"
    local INFER_SAMPLE_LIST="${TEMP_DIR}/infer_sample_${corruption}_len${current_p_len}_nb${current_num_bands}_r${current_rho}.txt"
    local INFER_INPUT_DIR="${TEMP_DIR}/infer_inputs_${corruption}_len${current_p_len}_lf${current_lambda_f}_r${current_rho}"
    local LAYER_STR="All_Gate_Module_1_2"

    local DST_DIR_SEM="${PRED_ROOT_DIR}/semseg_${current_prompt_mode}_${corruption}_layer${LAYER_STR}_len${current_p_len}_s${steps}_lrP${lr_prompt}_lrB${lr_backbone}_nb${current_num_bands}_lg${current_lambda_gate}_lf${current_lambda_f}_r${current_rho}"
    local DST_DIR_INS="${PRED_ROOT_DIR}/inst_${current_prompt_mode}_${corruption}_layer${LAYER_STR}_len${current_p_len}_s${steps}_lrP${lr_prompt}_lrB${lr_backbone}_nb${current_num_bands}_lg${current_lambda_gate}_lf${current_lambda_f}_r${current_rho}"

    find "${INPUT_DIR}" -type f \( -name "*.jpg" -o -name "*.png" \) | sort > "${INFER_ALL_LIST}"
    local TOTAL_INFER_IMAGES
    TOTAL_INFER_IMAGES=$(wc -l < "${INFER_ALL_LIST}")
    if [ "${TOTAL_INFER_IMAGES}" -eq 0 ]; then
        echo "[Error] No input images found in ${INPUT_DIR}" >&2
        return 1
    fi
    local INFER_COUNT=${INFER_SAMPLE_COUNT}
    [ "${TOTAL_INFER_IMAGES}" -lt "${INFER_COUNT}" ] && INFER_COUNT=${TOTAL_INFER_IMAGES}
    sample_file_list "${INFER_ALL_LIST}" "${INFER_SAMPLE_LIST}" "${INFER_COUNT}" "${RANDOM_SEED}"

    rm -rf "${INFER_INPUT_DIR}"
    mkdir -p "${INFER_INPUT_DIR}"
    while IFS= read -r img_path; do
        ln -sf "$(realpath "${img_path}")" "${INFER_INPUT_DIR}/$(basename "${img_path}")"
    done < "${INFER_SAMPLE_LIST}"

    echo "Running Inference: ${corruption} | Prompt Mode: ${current_prompt_mode} | Len: ${current_p_len} | Output Dirs: ${DST_DIR_SEM} & ${DST_DIR_INS}"
    echo "[Protocol] Development subset: ${INFER_COUNT} images. Do not report as full-val COCO PQ."
    rm -rf "${DST_DIR_SEM}" "${DST_DIR_INS}"

    # 1. 执行 SemSeg 推理
    set_prompt_path_args "semseg" "${current_prompt_mode}" "${corruption}"
    "${PYTHON_BIN}" "${INFERENCE_SCRIPT}" \
      --mode inference --task semseg \
      ${FC_VICT_FLAG} --lambda_f "${current_lambda_f}" --rho "${current_rho}" \
      --global_winner_path "${global_winner_sem_file}" \
      --ckpt_path "${CKPT_PATH}" \
      --model "${MODEL}" \
      --prompt "${PROMPT}" \
      --input_dir "${INFER_INPUT_DIR}" \
      --save_dir "${DST_DIR_SEM}" \
      --steps "${steps}" \
      --lr "${lr_prompt}" \
      --lr_backbone "${lr_backbone}" \
      --vpt_type "${VPT_TYPE}" \
      --prompt_length "${current_p_len}" \
      --vpt_target_layer ${current_layers} \
      --num_bands "${current_num_bands}" \
      --prompt_level "${SEVERITY}" \
      --weight_decay "${WEIGHT_DECAY}" \
      --seed "${RANDOM_SEED}" \
      "${PROMPT_PATH_ARGS[@]}"

    # 2. 执行 Inst 推理
    set_prompt_path_args "inst" "${current_prompt_mode}" "${corruption}"
    "${PYTHON_BIN}" "${INFERENCE_SCRIPT}" \
      --mode inference --task inst \
      ${FC_VICT_FLAG} --lambda_f "${current_lambda_f}" --rho "${current_rho}" \
      --global_winner_path "${global_winner_inst_file}" \
      --ckpt_path "${CKPT_PATH}" \
      --model "${MODEL}" \
      --prompt "${PROMPT}" \
      --input_dir "${INFER_INPUT_DIR}" \
      --save_dir "${DST_DIR_INS}" \
      --steps "${steps}" \
      --lr "${lr_prompt}" \
      --lr_backbone "${lr_backbone}" \
      --vpt_type "${VPT_TYPE}" \
      --prompt_length "${current_p_len}" \
      --vpt_target_layer ${current_layers} \
      --num_bands "${current_num_bands}" \
      --prompt_level "${SEVERITY}" \
      --weight_decay "${WEIGHT_DECAY}" \
      --seed "${RANDOM_SEED}" \
      "${PROMPT_PATH_ARGS[@]}"

    if [ $? -ne 0 ]; then
        echo "[Error] Inference failed for ${corruption}."
        return
    fi

    # 3. 结果评估
    local EVAL_STDOUT_LOG="${TEMP_DIR}/eval_stdout_${corruption}.log"
    local EVAL_STDERR_LOG="${TEMP_DIR}/eval_stderr_${corruption}.log"
    set +e
    "${PYTHON_BIN}" eval/coco_panoptic/eval_with_pngs.py \
      --pred_path_sem "${DST_DIR_SEM}" \
      --pred_path_inst "${DST_DIR_INS}" \
      --gt_path "datasets/coco/annotations/instances_val2017.json" \
      --dist_thr 19 \
      --prompt "${PROMPT}" \
      --ckpt_file "${CKPT_FILE##*/}" \
      --input_size 448 \
      > "${EVAL_STDOUT_LOG}" 2> "${EVAL_STDERR_LOG}"
    local EVAL_STATUS=$?
    set -e

    local OUTPUT
    OUTPUT=$(cat "${EVAL_STDOUT_LOG}")

    if [ ${EVAL_STATUS} -ne 0 ]; then
        echo "[Error] Evaluation failed for ${corruption}. Full stdout/stderr below:"
        cat "${EVAL_STDOUT_LOG}"
        cat "${EVAL_STDERR_LOG}"
        return 1
    fi

    PQ=$(echo "$OUTPUT" | awk -F'|' '/^PQ[[:space:]]*\|/ {gsub(/[[:space:]]/, "", $2); print $2; exit}')
    if [ -z "$PQ" ]; then
        PQ="N/A"
        echo "[Warning] PQ 提取失败"
    fi

    WIN_STATS_SEM="N/A"
    if [ -f "${DST_DIR_SEM}/winning_groups.txt" ]; then
        WIN_STATS_SEM=$(cat "${DST_DIR_SEM}/winning_groups.txt")
    fi
    WIN_STATS_INST="N/A"
    if [ -f "${DST_DIR_INS}/winning_groups.txt" ]; then
        WIN_STATS_INST=$(cat "${DST_DIR_INS}/winning_groups.txt")
    fi

    echo "${current_prompt_mode},${corruption},${steps},${lr_prompt},${lr_backbone},${LAYER_STR},${current_p_len},${current_num_bands},${current_lambda_gate},${current_lambda_f},${current_rho},${WIN_STATS_SEM},${WIN_STATS_INST},${PQ}" >> "${RESULT_LOG}"
    echo "[Done] Result saved to ${RESULT_LOG} (PQ: ${PQ})"

    if [ "${KEEP_OUTPUTS}" = "1" ]; then
        echo "[Keep] Preserving predictions and sampled inputs under ${PRED_ROOT_DIR} and ${INFER_INPUT_DIR}"
    else
        echo "[Cleanup] Removing predictions and temporary inference directory..."
        rm -rf "${DST_DIR_SEM}" "${DST_DIR_INS}" "${INFER_INPUT_DIR}"
    fi
    if [ "${KEEP_OUTPUTS}" != "1" ]; then
        rm -f "${INFER_ALL_LIST}" "${INFER_SAMPLE_LIST}" "${EVAL_STDOUT_LOG}" "${EVAL_STDERR_LOG}"
    fi
}

# =========================================================
# 主流程
# =========================================================
validate_entry_config
if [ "${CONFIG_CHECK_ONLY}" = "1" ]; then
    echo "[Config] Validation passed. CONFIG_CHECK_ONLY=1, exiting without inference."
    exit 0
fi

EXPECTED_HEADER="prompt_mode,corruption,steps,lr_prompt,lr_backbone,layers,prompt_len,num_bands,lambda_gate,lambda_f,rho,winning_groups_sem,winning_groups_inst,pq"
if [ -f "${RESULT_LOG}" ] && [ "$(head -n 1 "${RESULT_LOG}")" != "${EXPECTED_HEADER}" ]; then
    RESULT_LOG="${RESULT_LOG%.csv}_fixed.csv"
    echo "[Warning] Existing result log header mismatch; writing fixed-format results to ${RESULT_LOG}"
fi
if [ ! -f "${RESULT_LOG}" ]; then
    echo "${EXPECTED_HEADER}" > "${RESULT_LOG}"
fi

find "${CLEAN_DATA_DIR}" -type f \( -name "*.jpg" -o -name "*.png" \) | sort > "${TEMP_DIR}/all_clean_imgs.txt"
if [ ! -s "${TEMP_DIR}/all_clean_imgs.txt" ]; then
    echo "[Error] No clean COCO images found in ${CLEAN_DATA_DIR}" >&2
    exit 1
fi
sample_file_list "${TEMP_DIR}/all_clean_imgs.txt" "${TEMP_DIR}/clean_sample_list.txt" "50" "${RANDOM_SEED}"

for NUM_BANDS in "${NUM_BANDS_LIST[@]}"; do
  for current_rho in "${RHO_LIST[@]}"; do
    rho_safe=$(echo "${current_rho}" | tr '.' '_')

    for corruption in "${CORRUPTIONS[@]}"; do
        echo "========================================================="
        echo ">>> [Dispatch] Task started for: ${corruption}"
        echo "========================================================="

        INPUT_DIR="datasets/coco-c/${corruption}/${SEVERITY}/val2017/"
        CORRUPT_SAMPLE_LIST="${TEMP_DIR}/sample_list_${corruption}_nb${NUM_BANDS}_r${rho_safe}.txt"
        GLOBAL_WINNER_PREFIX="${TEMP_DIR}/global_winner_${corruption}_nb${NUM_BANDS}_r${rho_safe}"

        echo ">>> [Stage 2 & 3] Evaluating ${DRIFT_SAMPLE_COUNT} samples to vote for Winner Group..."
        find "${INPUT_DIR}" -type f \( -name "*.jpg" -o -name "*.png" \) | sort > "${TEMP_DIR}/all_imgs_${corruption}.txt"
        TOTAL_IMAGES=$(wc -l < "${TEMP_DIR}/all_imgs_${corruption}.txt")
        if [ "${TOTAL_IMAGES}" -eq 0 ]; then
            echo "[Error] No corrupted COCO images found in ${INPUT_DIR}" >&2
            exit 1
        fi
        SAMPLE_COUNT=${DRIFT_SAMPLE_COUNT}
        [ ${TOTAL_IMAGES} -lt ${SAMPLE_COUNT} ] && SAMPLE_COUNT=${TOTAL_IMAGES}
        sample_file_list "${TEMP_DIR}/all_imgs_${corruption}.txt" "${CORRUPT_SAMPLE_LIST}" "${SAMPLE_COUNT}" "${RANDOM_SEED}"

        for prompt_mode in "${PROMPT_MODE_LIST[@]}"; do
          echo ">>> [Prompt Mode] Current mode: ${prompt_mode}"
        for lambda_gate in "${LAMBDA_GATE_LIST[@]}"; do
          for current_lambda_f in "${LAMBDA_F_LIST[@]}"; do
            lf_safe=$(echo "${current_lambda_f}" | tr '.' '_')

            echo ">>> [Stage 4] Starting Inference Loop for Voted Winner..."
            for steps in "${STEPS_LIST[@]}"; do
              for lr_p in "${LR_PROMPT_LIST[@]}"; do
                for lr_b in "${LR_BACKBONE[@]}"; do
                  for p_len in "${PROMPT_LEN_LIST[@]}"; do
                    GLOBAL_ANCHOR_SEM="${TEMP_DIR}/${GLOBAL_ANCHOR_PREFIX}_semseg_len${p_len}_nb${NUM_BANDS}_r${rho_safe}.pt"
                    GLOBAL_ANCHOR_INST="${TEMP_DIR}/${GLOBAL_ANCHOR_PREFIX}_inst_len${p_len}_nb${NUM_BANDS}_r${rho_safe}.pt"
                    GLOBAL_WINNER_SEM="${GLOBAL_WINNER_PREFIX}_${prompt_mode}_semseg_len${p_len}_lg${lambda_gate}_lf${lf_safe}.txt"
                    GLOBAL_WINNER_INST="${GLOBAL_WINNER_PREFIX}_${prompt_mode}_inst_len${p_len}_lg${lambda_gate}_lf${lf_safe}.txt"
                    if [ "${prompt_mode}" = "zero_shot" ] && [ "${corruption}" = "gaussian_noise" ] && [ -n "${VALIDATED_GAUSSIAN_WINNER_GROUP}" ]; then
                        printf '%s\n' "${VALIDATED_GAUSSIAN_WINNER_GROUP}" > "${GLOBAL_WINNER_SEM}"
                        printf '%s\n' "${VALIDATED_GAUSSIAN_WINNER_GROUP}" > "${GLOBAL_WINNER_INST}"
                        echo ">>> [Validated Protocol] Forced ${VALIDATED_GAUSSIAN_WINNER_GROUP} for semseg and inst."
                    else
                        run_anchor_if_needed "semseg" "${GLOBAL_ANCHOR_SEM}" "${NUM_BANDS}" "${current_rho}" "${p_len}" "${TEMP_DIR}/clean_sample_list.txt"
                        run_anchor_if_needed "inst" "${GLOBAL_ANCHOR_INST}" "${NUM_BANDS}" "${current_rho}" "${p_len}" "${TEMP_DIR}/clean_sample_list.txt"
                        run_eval_drift_for_task "semseg" "${GLOBAL_ANCHOR_SEM}" "${GLOBAL_WINNER_SEM}" "${CORRUPT_SAMPLE_LIST}" "${INPUT_DIR}" "${p_len}" "${NUM_BANDS}" "${lambda_gate}" "${current_lambda_f}" "${current_rho}" "${prompt_mode}" "${corruption}"
                        run_eval_drift_for_task "inst" "${GLOBAL_ANCHOR_INST}" "${GLOBAL_WINNER_INST}" "${CORRUPT_SAMPLE_LIST}" "${INPUT_DIR}" "${p_len}" "${NUM_BANDS}" "${lambda_gate}" "${current_lambda_f}" "${current_rho}" "${prompt_mode}" "${corruption}"
                    fi

                    do_inference "${corruption}" "${steps}" "${lr_p}" "${lr_b}" "${TARGET_LAYERS}" "${p_len}" "${GLOBAL_WINNER_SEM}" "${GLOBAL_WINNER_INST}" "${NUM_BANDS}" "${lambda_gate}" "${current_lambda_f}" "${current_rho}" "${prompt_mode}"
                  done
                done
              done
            done
          done
        done
        done
        echo ">>> [Success] Evaluation completed for: ${corruption}"
        echo ""
    done
  done
done

if [ "${KEEP_TEMP}" = "1" ]; then
    echo "[Keep] Preserving sampled lists and logs under ${TEMP_DIR}"
else
    echo "[Cleanup] Removing temporary directory: ${TEMP_DIR}"
    rm -rf "${TEMP_DIR}"
fi
echo "All evaluations finished! Results saved to ${RESULT_LOG}."
