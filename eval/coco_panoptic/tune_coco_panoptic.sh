#!/bin/bash
set -e

# =========================================================
# COCO Panoptic 5 阶段自动调参引擎
# 策略: 5 阶段 × (1+5+3+3+3) 轮网格搜索（动态范围缩小）
# 指标: PQ (Panoptic Quality，越大越好)
# 注意: STEPS 固定为 10，Stage 4 仅调 LR_PROMPT
#
# 用法: bash tune_coco_panoptic.sh <corruption_type> [zero_shot|one_shot]
# 例:   bash tune_coco_panoptic.sh gaussian_noise one_shot
# =========================================================

if [ $# -lt 1 ]; then
    echo "用法: bash tune_coco_panoptic.sh <corruption_type> [zero_shot|one_shot]"
    echo "  corruption: gaussian_noise, shot_noise, impulse_noise, defocus_blur, glass_blur,"
    echo "              motion_blur, zoom_blur, snow, frost, fog, brightness, contrast,"
    echo "              elastic_transform, pixelate, jpeg_compression"
    echo "  prompt_mode: zero_shot (默认) 或 one_shot"
    exit 1
fi

# =========================================================
# 环境设置
# =========================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON_BIN="${PYTHON_BIN:-python}"

# =========================================================
# 基础配置（源自 eval_new.sh）
# =========================================================
CORRUPTION="$1"
PROMPT_MODE="${2:-zero_shot}"

if [ "${PROMPT_MODE}" != "zero_shot" ] && [ "${PROMPT_MODE}" != "one_shot" ]; then
    echo "[Error] prompt_mode 必须是 zero_shot 或 one_shot，当前: ${PROMPT_MODE}"
    exit 1
fi

PROMPT_CORRUPTION_ARG=""
if [ "${PROMPT_MODE}" = "one_shot" ]; then
    PROMPT_CORRUPTION_ARG="--prompt_corruption ${CORRUPTION}"
fi

CKPT_FILE="painter_vit_large/painter_vit_large.pth"
PROMPT="000000391460"
MODEL="painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1"
CKPT_PATH="models/${CKPT_FILE}"
VPT_TYPE="custom"
TARGET_LAYERS="0 1 2 10 11 12 21 22 23"
SEVERITY=5
FC_VICT_FLAG="--use_fc_vict"
LR_BACKBONE=0
INFER_SAMPLE_COUNT=100
DRIFT_SAMPLE_COUNT=50
RANDOM_SEED=20260426

CLEAN_DATA_DIR="datasets/coco/val2017/"
GLOBAL_ANCHOR_PREFIX="anchor_stats_global_clean_coco"
CALIBRATION_FILE="calibration_baseline_coco.pt"
GT_ANNOTATIONS="datasets/coco/annotations/instances_val2017.json"

INFERENCE_SCRIPT="${SCRIPT_DIR}/painter_inference_new.py"
EVALUATOR_SCRIPT="${SCRIPT_DIR}/eval_with_pngs.py"

if [ "${PROMPT_MODE}" = "one_shot" ]; then
    TEMP_DIR="temp/coco_panoptic_${CORRUPTION}_oneshot_tuning"
    RESULT_LOG="results_coco_${CORRUPTION}_oneshot_tuning.csv"
    PRED_ROOT_DIR="models_inference/vpt_results_coco_${CORRUPTION}_oneshot_tuning"
else
    TEMP_DIR="temp/coco_panoptic_${CORRUPTION}_tuning"
    RESULT_LOG="results_coco_${CORRUPTION}_tuning.csv"
    PRED_ROOT_DIR="models_inference/vpt_results_coco_${CORRUPTION}_tuning"
fi

# =========================================================
# 初始参数默认值（源自 eval_new.sh 固定参数）
# =========================================================
BEST_LAMBDA_GATE=0.3
BEST_NUM_BANDS=5
BEST_RHO=0.5
BEST_LAMBDA_F=0.01
BEST_STEPS=10
BEST_LR=3e-5
BEST_P_LEN=100

BEST_PQ=0
TARGET_GROUP=""
VOTE_PASSED=1
FORCED_GROUP=""

mkdir -p "${TEMP_DIR}"

# =========================================================
# 初始化结果 CSV
# =========================================================
if [ ! -f "${RESULT_LOG}" ]; then
    echo "stage,round,param1,param1_val,param2,param2_val,steps,lr_prompt,prompt_len,num_bands,lambda_gate,lambda_f,rho,target_group,winner_group,vote_passed,pq,stage2_margin,timestamp" > ${RESULT_LOG}
fi

# =========================================================
# 工具函数: sample_file_list（从 eval_new.sh 复用）
# =========================================================
sample_file_list() {
    local src_list=$1
    local dst_list=$2
    local sample_count=$3
    local seed=$4
    python - "$src_list" "$dst_list" "$sample_count" "$seed" <<'PY'
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

# =========================================================
# 工具函数: float 运算
# =========================================================
float_eval() {
    "${PYTHON_BIN}" -c "print(round(eval('$1'), 6))"
}

float_compare() {
    # PQ 越大越好: 返回 1 if $1 > $2, else 0
    local result
    result=$("${PYTHON_BIN}" -c "print(1 if $1 > $2 else 0)")
    echo "${result}"
}

# =========================================================
# 工具函数: 范围缩小
# =========================================================
shrink_range() {
    local best_val=$1
    local step=$2
    local min_val=$3
    local max_val=$4
    local is_int=${5:-false}

    "${PYTHON_BIN}" -c "
best, step, mn, mx = float('$1'), float('$2'), float('$3'), float('$4')
is_int = '${is_int}' == 'true'
lo = best - step
hi = best + step
if lo < mn: lo = mn
if hi > mx: hi = mx
def fmt(v):
    if is_int: return str(int(round(v)))
    return f'{v:.6g}'
print(f'{fmt(lo)} {fmt(best)} {fmt(hi)}')
"
}

shrink_range_log() {
    local best_val=$1
    local factor=$2
    local min_val=$3
    local max_val=$4

    "${PYTHON_BIN}" -c "
import math
best, factor, mn, mx = float('$1'), float('$2'), float('$3'), float('$4')
lo = best / factor
hi = best * factor
if lo < mn: lo = mn
if hi > mx: hi = mx
def fmt(v):
    if v < 1e-4: return f'{v:.1e}'
    return f'{v:.6g}'
print(f'{fmt(lo)} {fmt(best)} {fmt(hi)}')
"
}

# =========================================================
# 工具函数: get_winner_group
# =========================================================
get_winner_group() {
    local winner_file=$1
    if [ -f "${winner_file}" ]; then
        cat "${winner_file}" | tr -d '[:space:]'
    else
        echo ""
    fi
}

# =========================================================
# 工具函数: get_scores_margin
# =========================================================
get_scores_margin() {
    local winner_file=$1
    local target=$2
    local scores_file="${winner_file%.txt}_scores.txt"

    if [ ! -f "${scores_file}" ]; then
        echo "-999"
        return
    fi

    "${PYTHON_BIN}" -c "
import sys
scores = {}
with open('${scores_file}') as f:
    for line in f:
        if '=' in line:
            k, v = line.strip().split('=')
            try:
                scores[k] = float(v)
            except:
                pass

target = '${target}'
if target not in ['group1', 'group2', 'group3']:
    print(-999)
    sys.exit(0)

target_score = scores.get(target + '_score', -1)
other_scores = [v for k, v in scores.items() if k != target + '_score' and 'score' in k]
max_other = max(other_scores) if other_scores else 0
margin = target_score - max_other if target_score > max_other else -1
print(round(margin, 6))
"
}

# =========================================================
# 核心函数: ensure_anchor（anchor 使用干净数据）
# =========================================================
ensure_anchor() {
    local current_num_bands=$1
    local current_rho=$2
    local rho_safe
    rho_safe=$(echo "${current_rho}" | tr '.' '_')
    local anchor_file="${TEMP_DIR}/${GLOBAL_ANCHOR_PREFIX}_nb${current_num_bands}_r${rho_safe}.pt"

    if [ ! -f "${anchor_file}" ]; then
        echo ">>> [Anchor] Generating for bands=${current_num_bands}, rho=${current_rho}..." >&2
        find "${CLEAN_DATA_DIR}" -type f \( -name "*.jpg" -o -name "*.png" \) > "${TEMP_DIR}/all_clean_imgs_coco.txt"
        sample_file_list "${TEMP_DIR}/all_clean_imgs_coco.txt" "${TEMP_DIR}/clean_sample_list_coco.txt" "50" "${RANDOM_SEED}"

        "${PYTHON_BIN}" "${INFERENCE_SCRIPT}" \
          --mode anchor --task semseg \
          ${FC_VICT_FLAG} \
          --rho ${current_rho} \
          --sample_list "${TEMP_DIR}/clean_sample_list_coco.txt" \
          --anchor_stats_path ${anchor_file} \
          --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
          --input_dir "dummy" --save_dir "dummy" \
          --vpt_type ${VPT_TYPE} --prompt_length 50 --vpt_target_layer ${TARGET_LAYERS} \
          --num_bands ${current_num_bands} > /dev/null 2>&1

        echo ">>> [Anchor] Done: ${anchor_file}" >&2
    fi
    echo "${anchor_file}"
}

# =========================================================
# 核心函数: compute_winner
# 如果设置了 FORCED_GROUP，直接写 winner 文件跳过投票
# 否则运行 eval_drift 进行投票
# =========================================================
compute_winner() {
    local current_num_bands=$1
    local current_lambda_gate=$2
    local current_lambda_f=$3
    local current_rho=$4
    local anchor_file=$5

    local rho_safe=$(echo "${current_rho}" | tr '.' '_')
    local lf_safe=$(echo "${current_lambda_f}" | tr '.' '_')
    local winner_file="${TEMP_DIR}/global_winner_${CORRUPTION}_nb${current_num_bands}_r${rho_safe}_lg${current_lambda_gate}_lf${lf_safe}.txt"

    # FORCED_GROUP: 跳过投票，直接写入指定 group
    if [ -n "${FORCED_GROUP}" ]; then
        echo ">>> [Winner] Forced group: ${FORCED_GROUP}" >&2
        echo "${FORCED_GROUP}" > "${winner_file}"
        local scores_file="${winner_file%.txt}_scores.txt"
        echo "group1_score=0" > "${scores_file}"
        echo "group2_score=0" >> "${scores_file}"
        echo "group3_score=0" >> "${scores_file}"
        echo "winner=${FORCED_GROUP}" >> "${scores_file}"
        echo "${winner_file}"
        return
    fi

    if [ ! -f "${winner_file}" ]; then
        local INPUT_DIR="datasets/coco-c/${CORRUPTION}/${SEVERITY}/val2017/"
        local ALL_IMGS_FILE="${TEMP_DIR}/all_imgs_winner_${CORRUPTION}.txt"
        find "${INPUT_DIR}" -type f \( -name "*.jpg" -o -name "*.png" \) | sort > "${ALL_IMGS_FILE}"
        local TOTAL_IMAGES
        TOTAL_IMAGES=$(wc -l < "${ALL_IMGS_FILE}" | tr -d ' ')
        local SAMPLE_COUNT=${DRIFT_SAMPLE_COUNT}
        [ "${TOTAL_IMAGES}" -lt "${SAMPLE_COUNT}" ] && SAMPLE_COUNT=${TOTAL_IMAGES}
        [ "${SAMPLE_COUNT}" -eq 0 ] && SAMPLE_COUNT=1

        local CORRUPT_SAMPLE_LIST="${TEMP_DIR}/sample_list_${CORRUPTION}_nb${current_num_bands}_r${rho_safe}.txt"
        sample_file_list "${ALL_IMGS_FILE}" "${CORRUPT_SAMPLE_LIST}" "${SAMPLE_COUNT}" "${RANDOM_SEED}"

        echo ">>> [Winner] Voting: nb=${current_num_bands}, lg=${current_lambda_gate}, lf=${current_lambda_f}, rho=${current_rho}" >&2
        "${PYTHON_BIN}" "${INFERENCE_SCRIPT}" \
          --mode eval_drift --task semseg \
          ${FC_VICT_FLAG} \
          --lambda_f ${current_lambda_f} \
          --rho ${current_rho} \
          --sample_list ${CORRUPT_SAMPLE_LIST} \
          --anchor_stats_path ${anchor_file} \
          --calibration_path ${CALIBRATION_FILE} \
          --global_winner_path ${winner_file} \
          --ckpt_path ${CKPT_PATH} \
          --model ${MODEL} \
          --prompt ${PROMPT} \
          --input_dir "${INPUT_DIR}" \
          --save_dir "dummy" \
          --vpt_type ${VPT_TYPE} \
          --prompt_length 50 \
          --vpt_target_layer ${TARGET_LAYERS} \
          --num_bands ${current_num_bands} \
          --lambda_gate ${current_lambda_gate} \
          ${PROMPT_CORRUPTION_ARG} > /dev/null 2>&1
    fi
    echo "${winner_file}"
}

# =========================================================
# 核心函数: run_eval（完整推理+评估+记录）
# 用于 Stage 1, 3, 4, 5 的完整推理评估
# =========================================================
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
    local current_vote_passed=${14:-1}

    local rho_safe=$(echo "${current_rho}" | tr '.' '_')
    local lf_safe=$(echo "${current_lf}" | tr '.' '_')

    # 确保 anchor 存在
    local anchor_file
    anchor_file=$(ensure_anchor "${current_nb}" "${current_rho}")

    # 确保 winner group 存在
    local winner_file
    winner_file=$(compute_winner "${current_nb}" "${current_lg}" "${current_lf}" "${current_rho}" "${anchor_file}")

    local INPUT_DIR="datasets/coco-c/${CORRUPTION}/${SEVERITY}/val2017/"
    local INFER_ALL_LIST="${TEMP_DIR}/all_imgs_infer_${CORRUPTION}.txt"
    local INFER_SAMPLE_LIST="${TEMP_DIR}/infer_sample_${stage}_r${round}.txt"
    local INFER_INPUT_DIR="${TEMP_DIR}/infer_inputs_${stage}_r${round}"

    local DST_DIR_SEM="${PRED_ROOT_DIR}/semseg_${stage}_r${round}_s${current_steps}_lr${current_lr}_len${current_p_len}_nb${current_nb}_lg${current_lg}_lf${current_lf}_rho${current_rho}"
    local DST_DIR_INS="${PRED_ROOT_DIR}/inst_${stage}_r${round}_s${current_steps}_lr${current_lr}_len${current_p_len}_nb${current_nb}_lg${current_lg}_lf${current_lf}_rho${current_rho}"

    echo ">>> [${stage}] Round ${round} | Steps=${current_steps}, LR=${current_lr}, Len=${current_p_len}, NB=${current_nb}, LG=${current_lg}, LF=${current_lf}, RHO=${current_rho}" >&2

    # 采样推理图片
    find "${INPUT_DIR}" -type f \( -name "*.jpg" -o -name "*.png" \) > "${INFER_ALL_LIST}"
    local TOTAL_INFER_IMAGES
    TOTAL_INFER_IMAGES=$(wc -l < "${INFER_ALL_LIST}")
    local INFER_COUNT=${INFER_SAMPLE_COUNT}
    [ "${TOTAL_INFER_IMAGES}" -lt "${INFER_COUNT}" ] && INFER_COUNT=${TOTAL_INFER_IMAGES}
    [ "${INFER_COUNT}" -eq 0 ] && INFER_COUNT=1
    sample_file_list "${INFER_ALL_LIST}" "${INFER_SAMPLE_LIST}" "${INFER_COUNT}" "${RANDOM_SEED}"

    rm -rf "${INFER_INPUT_DIR}"
    mkdir -p "${INFER_INPUT_DIR}"
    while IFS= read -r img_path; do
        ln -sf "$(realpath "${img_path}")" "${INFER_INPUT_DIR}/$(basename "${img_path}")"
    done < "${INFER_SAMPLE_LIST}"

    # 清理旧输出
    rm -rf "${DST_DIR_SEM}" "${DST_DIR_INS}"

    # 运行 semseg 推理
    if ! "${PYTHON_BIN}" "${INFERENCE_SCRIPT}" \
      --mode inference --task semseg \
      ${FC_VICT_FLAG} --lambda_f ${current_lf} --rho ${current_rho} \
      --global_winner_path ${winner_file} \
      --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
      --input_dir "${INFER_INPUT_DIR}" --save_dir "${DST_DIR_SEM}" \
      --steps ${current_steps} --lr ${current_lr} --lr_backbone ${LR_BACKBONE} \
      --vpt_type ${VPT_TYPE} --prompt_length ${current_p_len} \
      --vpt_target_layer ${TARGET_LAYERS} --num_bands ${current_nb} \
      ${PROMPT_CORRUPTION_ARG} > /dev/null 2>&1; then
        echo "[Error] SemSeg inference failed for ${stage} Round ${round}" >&2
        local TIMESTAMP
        TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
        echo "${stage},${round},${param1},${param1_val},${param2},${param2_val},${current_steps},${current_lr},${current_p_len},${current_nb},${current_lg},${current_lf},${current_rho},${TARGET_GROUP},N/A,${current_vote_passed},ERROR_INFERENCE,N/A,${TIMESTAMP}" >> ${RESULT_LOG}
        rm -rf "${DST_DIR_SEM}" "${DST_DIR_INS}" "${INFER_INPUT_DIR}" "${INFER_ALL_LIST}" "${INFER_SAMPLE_LIST}"
        echo "ERROR_INFERENCE"
        return
    fi

    # 运行 inst 推理
    if ! "${PYTHON_BIN}" "${INFERENCE_SCRIPT}" \
      --mode inference --task inst \
      ${FC_VICT_FLAG} --lambda_f ${current_lf} --rho ${current_rho} \
      --global_winner_path ${winner_file} \
      --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
      --input_dir "${INFER_INPUT_DIR}" --save_dir "${DST_DIR_INS}" \
      --steps ${current_steps} --lr ${current_lr} --lr_backbone ${LR_BACKBONE} \
      --vpt_type ${VPT_TYPE} --prompt_length ${current_p_len} \
      --vpt_target_layer ${TARGET_LAYERS} --num_bands ${current_nb} \
      ${PROMPT_CORRUPTION_ARG} > /dev/null 2>&1; then
        echo "[Error] Inst inference failed for ${stage} Round ${round}" >&2
        local TIMESTAMP
        TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
        echo "${stage},${round},${param1},${param1_val},${param2},${param2_val},${current_steps},${current_lr},${current_p_len},${current_nb},${current_lg},${current_lf},${current_rho},${TARGET_GROUP},N/A,${current_vote_passed},ERROR_INFERENCE,N/A,${TIMESTAMP}" >> ${RESULT_LOG}
        rm -rf "${DST_DIR_SEM}" "${DST_DIR_INS}" "${INFER_INPUT_DIR}" "${INFER_ALL_LIST}" "${INFER_SAMPLE_LIST}"
        echo "ERROR_INFERENCE"
        return
    fi

    # 运行评估
    local EVAL_STDOUT_LOG="${TEMP_DIR}/eval_stdout_${stage}_r${round}.log"
    local EVAL_STDERR_LOG="${TEMP_DIR}/eval_stderr_${stage}_r${round}.log"
    set +e
    "${PYTHON_BIN}" "${EVALUATOR_SCRIPT}" \
      --pred_path_sem "${DST_DIR_SEM}" \
      --pred_path_inst "${DST_DIR_INS}" \
      --gt_path "${GT_ANNOTATIONS}" \
      --dist_thr 19 \
      --prompt ${PROMPT} \
      --ckpt_file "${CKPT_FILE##*/}" \
      --input_size 448 \
      > "${EVAL_STDOUT_LOG}" 2> "${EVAL_STDERR_LOG}"
    local EVAL_STATUS=$?
    set -e

    local PQ="N/A"
    if [ ${EVAL_STATUS} -eq 0 ]; then
        local OUTPUT
        OUTPUT=$(cat "${EVAL_STDOUT_LOG}")
        PQ=$(echo "$OUTPUT" | awk -F'|' '/^PQ[[:space:]]*\|/ {gsub(/[[:space:]]/, "", $2); print $2; exit}')
        if [ -z "$PQ" ]; then
            PQ="N/A"
        fi
    else
        echo "[Error] Evaluation failed for ${stage} Round ${round}" >&2
    fi

    local WINNER_GROUP
    WINNER_GROUP=$(get_winner_group "${winner_file}")

    local TIMESTAMP
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    echo "${stage},${round},${param1},${param1_val},${param2},${param2_val},${current_steps},${current_lr},${current_p_len},${current_nb},${current_lg},${current_lf},${current_rho},${TARGET_GROUP},${WINNER_GROUP},${current_vote_passed},${PQ},N/A,${TIMESTAMP}" >> ${RESULT_LOG}

    echo ">>> [${stage}] Round ${round} Result: PQ=${PQ}" >&2

    # 清理
    rm -rf "${DST_DIR_SEM}" "${DST_DIR_INS}" "${INFER_INPUT_DIR}"
    rm -f "${INFER_ALL_LIST}" "${INFER_SAMPLE_LIST}" "${EVAL_STDOUT_LOG}" "${EVAL_STDERR_LOG}"
    echo "${PQ}"
}

# =========================================================
# Stage 2 专用函数: run_stage2_vote（仅 eval_drift 投票，无推理）
# 返回: 通过 stdout 输出 "v_winner=<group> v_margin=<val> v_target_won=<0|1>"
# =========================================================
run_stage2_vote() {
    local current_nb=$1
    local current_lg=$2
    local current_lf=$3
    local current_rho=$4
    local target=$5

    local rho_safe=$(echo "${current_rho}" | tr '.' '_')
    local lf_safe=$(echo "${current_lf}" | tr '.' '_')
    local winner_file="${TEMP_DIR}/global_winner_${CORRUPTION}_nb${current_nb}_r${rho_safe}_lg${current_lg}_lf${lf_safe}.txt"

    # 如果已缓存则直接使用
    if [ ! -f "${winner_file}" ]; then
        local anchor_file
        anchor_file=$(ensure_anchor "${current_nb}" "${current_rho}")

        local INPUT_DIR="datasets/coco-c/${CORRUPTION}/${SEVERITY}/val2017/"
        local ALL_IMGS_FILE="${TEMP_DIR}/all_imgs_s2_${CORRUPTION}.txt"
        if [ ! -f "${ALL_IMGS_FILE}" ]; then
            find "${INPUT_DIR}" -type f \( -name "*.jpg" -o -name "*.png" \) | sort > "${ALL_IMGS_FILE}"
        fi
        local TOTAL_IMAGES
        TOTAL_IMAGES=$(wc -l < "${ALL_IMGS_FILE}" | tr -d ' ')
        local SAMPLE_COUNT=${DRIFT_SAMPLE_COUNT}
        [ "${TOTAL_IMAGES}" -lt "${SAMPLE_COUNT}" ] && SAMPLE_COUNT=${TOTAL_IMAGES}
        [ "${SAMPLE_COUNT}" -eq 0 ] && SAMPLE_COUNT=1

        local CORRUPT_SAMPLE_LIST="${TEMP_DIR}/sample_list_s2_${CORRUPTION}_nb${current_nb}_r${rho_safe}_lg${current_lg}.txt"
        sample_file_list "${ALL_IMGS_FILE}" "${CORRUPT_SAMPLE_LIST}" "${SAMPLE_COUNT}" "${RANDOM_SEED}"

        local EVAL_DRIFT_LOG="${TEMP_DIR}/eval_drift_nb${current_nb}_lg${current_lg}.log"
        "${PYTHON_BIN}" "${INFERENCE_SCRIPT}" \
          --mode eval_drift --task semseg \
          ${FC_VICT_FLAG} \
          --lambda_f ${current_lf} \
          --rho ${current_rho} \
          --sample_list ${CORRUPT_SAMPLE_LIST} \
          --anchor_stats_path ${anchor_file} \
          --calibration_path ${CALIBRATION_FILE} \
          --global_winner_path ${winner_file} \
          --ckpt_path ${CKPT_PATH} \
          --model ${MODEL} \
          --prompt ${PROMPT} \
          --input_dir "${INPUT_DIR}" \
          --save_dir "dummy" \
          --vpt_type ${VPT_TYPE} \
          --prompt_length 50 \
          --vpt_target_layer ${TARGET_LAYERS} \
          --num_bands ${current_nb} \
          --lambda_gate ${current_lg} \
          ${PROMPT_CORRUPTION_ARG} > "${EVAL_DRIFT_LOG}" 2>&1
    fi

    # 解析结果
    local winner
    winner=$(get_winner_group "${winner_file}")
    local margin
    margin=$(get_scores_margin "${winner_file}" "${target}")

    local target_won=0
    if [ "${winner}" = "${target}" ]; then
        target_won=1
    fi

    echo "v_winner=${winner} v_margin=${margin} v_target_won=${target_won}"
}

# =========================================================
# 主调参流程
# =========================================================
echo "========================================================="
echo ">>> COCO Panoptic ${CORRUPTION} 5-Stage Auto-Tuning"
echo "========================================================="
echo ">>> Prompt Mode: ${PROMPT_MODE}"
echo ">>> 初始参数: lg=${BEST_LAMBDA_GATE}, nb=${BEST_NUM_BANDS}, rho=${BEST_RHO}, lf=${BEST_LAMBDA_F}, steps=${BEST_STEPS}, lr=${BEST_LR}, plen=${BEST_P_LEN}"
echo ""

# =========================================================
# Stage 1: Group 选择（3 次完整推理）
# =========================================================
echo ""
echo "========================================================="
echo ">>> Stage 1: Group 选择 (FORCED_GROUP 强制各 group 获胜)"
echo "========================================================="

BEST_PQ_S1=0
BEST_GROUP_S1=""

for grp in group1 group2 group3; do
    echo ""
    echo "--- Stage 1: Testing ${grp} ---"

    # 强制指定当前测试的 group
    FORCED_GROUP_SAVED="${FORCED_GROUP}"
    FORCED_GROUP="${grp}"

    pq=$(run_eval "S1" "1" "group" "${grp}" "-" "-" \
        "${BEST_STEPS}" "${BEST_LR}" "${BEST_P_LEN}" \
        "${BEST_NUM_BANDS}" "${BEST_LAMBDA_GATE}" "${BEST_LAMBDA_F}" "${BEST_RHO}" "1")

    FORCED_GROUP="${FORCED_GROUP_SAVED}"

    if [ "${pq}" != "N/A" ] && [ "${pq}" != "ERROR_INFERENCE" ]; then
        if [ "$(float_compare "${pq}" "${BEST_PQ_S1}")" = "1" ]; then
            BEST_PQ_S1=${pq}
            BEST_GROUP_S1="${grp}"
        fi
    fi
    echo ">>> Stage 1 ${grp}: PQ=${pq}"
done

TARGET_GROUP="${BEST_GROUP_S1}"
BEST_PQ=${BEST_PQ_S1}
echo ""
echo ">>> Stage 1 最优 Group: ${TARGET_GROUP} (PQ=${BEST_PQ_S1})"

# =========================================================
# Stage 2: LAMBDA_GATE × NUM_BANDS（仅 eval_drift 投票）
# =========================================================
echo ""
echo "========================================================="
echo ">>> Stage 2: LAMBDA_GATE × NUM_BANDS (eval_drift 仅投票)"
echo ">>> Target Group: ${TARGET_GROUP}"
echo "========================================================="

# 宽初始范围，均匀采样 3 点
LG_ARR=(0.1 0.5 0.9)
NB_ARR=(2 10 18)
LG_STEP=0.4
NB_STEP=8

VOTE_PASSED=0
BEST_LG_FOR_S2="${BEST_LAMBDA_GATE}"
BEST_NB_FOR_S2="${BEST_NUM_BANDS}"

for s2_round in 1 2 3 4 5; do
    echo ""
    echo "--- Stage 2, Round ${s2_round} ---"
    echo "    LG range: ${LG_ARR[*]}"
    echo "    NB range: ${NB_ARR[*]}"

    BEST_MARGIN_S2=-999
    BEST_LG_R2=""
    BEST_NB_R2=""
    ANY_TARGET_WON=false
    for lg in "${LG_ARR[@]}"; do
        for nb in "${NB_ARR[@]}"; do
            vote_result=$(run_stage2_vote "${nb}" "${lg}" "${BEST_LAMBDA_F}" "${BEST_RHO}" "${TARGET_GROUP}")

            # 解析投票结果
            eval "${vote_result}"

            TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
            echo "S2,${s2_round},lambda_gate,${lg},num_bands,${nb},${BEST_STEPS},${BEST_LR},${BEST_P_LEN},${nb},${lg},${BEST_LAMBDA_F},${BEST_RHO},${TARGET_GROUP},${v_winner},0,N/A,${v_margin},${TIMESTAMP}" >> ${RESULT_LOG}

            echo "    LG=${lg}, NB=${nb} => winner=${v_winner}, target_won=${v_target_won}, margin=${v_margin}"

            if [ "${v_target_won}" = "1" ]; then
                ANY_TARGET_WON=true
                if [ "$(float_compare "${v_margin}" "${BEST_MARGIN_S2}")" = "1" ]; then
                    BEST_MARGIN_S2=${v_margin}
                    BEST_LG_R2=${lg}
                    BEST_NB_R2=${nb}
                fi
            fi
        done
    done

    if [ "${ANY_TARGET_WON}" = "true" ]; then
        BEST_LAMBDA_GATE=${BEST_LG_R2}
        BEST_NUM_BANDS=${BEST_NB_R2}
        BEST_LG_FOR_S2=${BEST_LG_R2}
        BEST_NB_FOR_S2=${BEST_NB_R2}
        VOTE_PASSED=1
        # 锁定 group，后续 Stage 3-5 不再投票
        FORCED_GROUP="${TARGET_GROUP}"
        echo ">>> Stage 2 完成! Round ${s2_round}: 目标 group 获胜, vote_passed=1"
        echo "    最优: LAMBDA_GATE=${BEST_LAMBDA_GATE}, NUM_BANDS=${BEST_NUM_BANDS}, margin=${BEST_MARGIN_S2}"
        break
    else
        echo ">>> Stage 2 Round ${s2_round}: 目标 group 未获胜，缩小范围..."

        # 缩小范围
        LG_STEP=$(float_eval "${LG_STEP} / 2")
        NB_STEP=$(float_eval "${NB_STEP} / 2")
        if [ "$(float_compare "${NB_STEP}" "1")" != "1" ]; then NB_STEP=1; fi

        LG_CENTER="${LG_ARR[1]}"
        NB_CENTER="${NB_ARR[1]}"

        read -ra LG_ARR <<< "$(shrink_range "${LG_CENTER}" "${LG_STEP}" 0.05 0.95 false)"
        read -ra NB_ARR <<< "$(shrink_range "${NB_CENTER}" "${NB_STEP}" 1 20 true)"

        # 检查是否陷入死循环
        LG_SIZE=$(float_eval "(${LG_ARR[2]} - ${LG_ARR[0]})")
        if [ "$(float_compare "0.01" "${LG_SIZE}")" = "1" ] || [ "${NB_ARR[0]}" = "${NB_ARR[2]}" ]; then
            echo ">>> [Warning] Stage 2 范围已缩小到极限，终止搜索"
            break
        fi
    fi
done

if [ "${VOTE_PASSED}" = "0" ]; then
    # 5 轮后仍未获胜，降级为 FORCED_GROUP
    FORCED_GROUP="${TARGET_GROUP}"
    BEST_LAMBDA_GATE=0.3
    BEST_NUM_BANDS=5
    echo ">>> [Warning] Stage 2 兜底: 投票无法产生目标 group, 降级为 FORCED_GROUP=${FORCED_GROUP}"
    echo "    LG/NB 使用默认值: LG=0.3, NB=5"
fi

echo ""
echo ">>> Stage 2 最终: LG=${BEST_LAMBDA_GATE}, NB=${BEST_NUM_BANDS}, vote_passed=${VOTE_PASSED}, FORCED_GROUP=${FORCED_GROUP:-<none>}"

# =========================================================
# Stage 3: RHO × LAMBDA_F (3 轮，完整推理)
# =========================================================
echo ""
echo "========================================================="
echo ">>> Stage 3: RHO × LAMBDA_F (3 轮完整推理)"
echo "========================================================="

RHO_STEP=0.05
LF_FACTOR=3.16

RHO_ARR=(0.05 0.15 0.30)
LF_ARR=(0.005 0.01 0.05)

for s3_round in 1 2 3; do
    echo ""
    echo "--- Stage 3, Round ${s3_round} ---"

    BEST_PQ_S3=0
    BEST_RHO_R=${BEST_RHO}
    BEST_LF_R=${BEST_LAMBDA_F}

    for rho in "${RHO_ARR[@]}"; do
        for lf in "${LF_ARR[@]}"; do
            pq=$(run_eval "S3" "${s3_round}" "rho" "${rho}" "lambda_f" "${lf}" \
                "${BEST_STEPS}" "${BEST_LR}" "${BEST_P_LEN}" \
                "${BEST_NUM_BANDS}" "${BEST_LAMBDA_GATE}" "${lf}" "${rho}" "${VOTE_PASSED}")

            if [ "${pq}" != "N/A" ] && [ "${pq}" != "ERROR_INFERENCE" ]; then
                if [ "$(float_compare "${pq}" "${BEST_PQ_S3}")" = "1" ]; then
                    BEST_PQ_S3=${pq}
                    BEST_RHO_R=${rho}
                    BEST_LF_R=${lf}
                fi
            fi
        done
    done

    BEST_RHO=${BEST_RHO_R}
    BEST_LAMBDA_F=${BEST_LF_R}
    echo ">>> Stage 3 Round ${s3_round} 最优: RHO=${BEST_RHO}, LAMBDA_F=${BEST_LAMBDA_F}, PQ=${BEST_PQ_S3}"

    # 缩小范围
    RHO_STEP=$(float_eval "${RHO_STEP} / 2")
    LF_FACTOR=$(float_eval "(${LF_FACTOR}) ** 0.5")
    read -ra RHO_ARR <<< "$(shrink_range "${BEST_RHO}" "${RHO_STEP}" 0.005 0.50 false)"
    read -ra LF_ARR <<< "$(shrink_range_log "${BEST_LAMBDA_F}" "${LF_FACTOR}" 0.0005 0.20)"
done

echo ""
echo ">>> Stage 3 最终最优: RHO=${BEST_RHO}, LAMBDA_F=${BEST_LAMBDA_F}"

# =========================================================
# Stage 4: LR_PROMPT (3 轮，完整推理，STEPS 固定为 10)
# =========================================================
echo ""
echo "========================================================="
echo ">>> Stage 4: LR_PROMPT (3 轮完整推理, STEPS=${BEST_STEPS} 固定)"
echo "========================================================="

LR_FACTOR=2.0

LR_ARR=(1e-5 3e-5 5e-5)

for s4_round in 1 2 3; do
    echo ""
    echo "--- Stage 4, Round ${s4_round} ---"

    BEST_PQ_S4=0
    BEST_LR_R=${BEST_LR}

    for lr in "${LR_ARR[@]}"; do
        pq=$(run_eval "S4" "${s4_round}" "lr_prompt" "${lr}" "-" "-" \
            "${BEST_STEPS}" "${lr}" "${BEST_P_LEN}" \
            "${BEST_NUM_BANDS}" "${BEST_LAMBDA_GATE}" "${BEST_LAMBDA_F}" "${BEST_RHO}" "${VOTE_PASSED}")

        if [ "${pq}" != "N/A" ] && [ "${pq}" != "ERROR_INFERENCE" ]; then
            if [ "$(float_compare "${pq}" "${BEST_PQ_S4}")" = "1" ]; then
                BEST_PQ_S4=${pq}
                BEST_LR_R=${lr}
            fi
        fi
    done

    BEST_LR=${BEST_LR_R}
    echo ">>> Stage 4 Round ${s4_round} 最优: LR=${BEST_LR}, PQ=${BEST_PQ_S4}"

    # 缩小范围（对数缩放）
    LR_FACTOR=$(float_eval "(${LR_FACTOR}) ** 0.5")
    read -ra LR_ARR <<< "$(shrink_range_log "${BEST_LR}" "${LR_FACTOR}" 1e-6 1e-4)"
done

echo ""
echo ">>> Stage 4 最终最优: LR=${BEST_LR}"

# =========================================================
# Stage 5: PROMPT_LEN (3 轮，完整推理)
# =========================================================
echo ""
echo "========================================================="
echo ">>> Stage 5: PROMPT_LEN (3 轮完整推理)"
echo "========================================================="

PLEN_STEP=20

PLEN_ARR=(80 100 120)

for s5_round in 1 2 3; do
    echo ""
    echo "--- Stage 5, Round ${s5_round} ---"

    BEST_PQ_S5=0
    BEST_PLEN_R=${BEST_P_LEN}

    for plen in "${PLEN_ARR[@]}"; do
        pq=$(run_eval "S5" "${s5_round}" "prompt_len" "${plen}" "-" "-" \
            "${BEST_STEPS}" "${BEST_LR}" "${plen}" \
            "${BEST_NUM_BANDS}" "${BEST_LAMBDA_GATE}" "${BEST_LAMBDA_F}" "${BEST_RHO}" "${VOTE_PASSED}")

        if [ "${pq}" != "N/A" ] && [ "${pq}" != "ERROR_INFERENCE" ]; then
            if [ "$(float_compare "${pq}" "${BEST_PQ_S5}")" = "1" ]; then
                BEST_PQ_S5=${pq}
                BEST_PLEN_R=${plen}
            fi
        fi
    done

    BEST_P_LEN=${BEST_PLEN_R}
    echo ">>> Stage 5 Round ${s5_round} 最优: PROMPT_LEN=${BEST_P_LEN}, PQ=${BEST_PQ_S5}"

    # 缩小范围
    PLEN_STEP=$(float_eval "${PLEN_STEP} / 2")
    if [ "$(float_compare "${PLEN_STEP}" "5")" != "1" ]; then PLEN_STEP=5; fi
    read -ra PLEN_ARR <<< "$(shrink_range "${BEST_P_LEN}" "${PLEN_STEP}" 10 200 true)"
done

echo ""
echo ">>> Stage 5 最终最优: PROMPT_LEN=${BEST_P_LEN}"

# =========================================================
# 清理和总结
# =========================================================
echo ""
echo "========================================================="
echo ">>> Auto-Tuning Complete for: ${CORRUPTION}"
echo "========================================================="
echo ">>> Best Parameters:"
echo "    TARGET_GROUP   = ${TARGET_GROUP}"
echo "    LAMBDA_GATE    = ${BEST_LAMBDA_GATE}"
echo "    NUM_BANDS      = ${BEST_NUM_BANDS}"
echo "    RHO            = ${BEST_RHO}"
echo "    LAMBDA_F       = ${BEST_LAMBDA_F}"
echo "    STEPS          = ${BEST_STEPS} (固定)"
echo "    LR_PROMPT      = ${BEST_LR}"
echo "    PROMPT_LEN     = ${BEST_P_LEN}"
echo "    VOTE_PASSED    = ${VOTE_PASSED}"
if [ "${VOTE_PASSED}" = "0" ]; then
    echo "    FORCED_GROUP   = ${FORCED_GROUP} (投票兜底)"
fi
echo ""
echo ">>> 结果保存到: ${RESULT_LOG}"

# 清理临时推理输出
rm -rf "${PRED_ROOT_DIR}"
rm -rf "${TEMP_DIR}"
