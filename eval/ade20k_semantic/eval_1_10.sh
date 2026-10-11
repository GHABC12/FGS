#!/bin/bash
set -e

# =========================================================
# 环境配置
# =========================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"


PYTHON_BIN="${PYTHON_BIN:-python}"

# =========================================================
# 配置区
# =========================================================
CKPT_FILE="painter_vit_large/painter_vit_large.pth"
PROMPT="ADE_train_00009574"  # ADE20k 默认 prompt
MODEL="painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1"
CKPT_PATH="models/${CKPT_FILE}"

VPT_TYPE="custom"
PROMPT_LEN_LIST=(100)
TARGET_LAYERS="0 1 2 10 11 12 21 22 23"
SEVERITY=5
RESULT_LOG="results_ade20k_1_10.csv"
SEED=42

# Prompt 模式: zero-shot=干净提示图; one-shot=使用与当前 corruption 对应的损坏提示图
PROMPT_MODE="zero-shot"

# =========================================================
# FC-VICT 核心超参数配置
# =========================================================
FC_VICT_FLAG="--use_fc_vict"
# 频域正则化 Loss 权重 (针对分割任务，PDF建议 0 ~ 0.005)

LAMBDA_F_LIST=(0.005)
RHO_LIST=(0.05)

# 径向频带划分数 5 0.5
NUM_BANDS_LIST=(5)
LAMBDA_GATE_LIST=(0.5)
# 6 5e-5
STEPS_LIST=(6)
LR_PROMPT_LIST=(5e-5)
LR_BACKBONE=(0)

CORRUPTIONS=(
# "gaussian_noise"
# "shot_noise"
# "impulse_noise"
# "defocus_blur"
# "glass_blur"
# "motion_blur"
"zoom_blur"
# "snow"
"frost"
"fog"
"brightness"
# "contrast"
# "elastic_transform"
# "pixelate"
# "jpeg_compression"
)

# 全局文件配置
CLEAN_DATA_DIR="datasets/ade20k/images/validation/"
GLOBAL_ANCHOR_PREFIX="anchor_stats_global_clean_ade20k"
CALIBRATION_FILE="calibration_baseline_ade20k.pt"
TEMP_DIR="temp/ade20k_semantic_1_10"

mkdir -p "${TEMP_DIR}"

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
    local global_winner_file=$7
    local current_num_bands=$8
    local current_lambda_gate=$9
    local current_lambda_f=${10}
    local current_rho=${11}

    local INPUT_DIR="datasets/ade20k-c/${corruption}/${SEVERITY}/ade/ADEChallengeData2016/images/validation/"
    local LAYER_STR="All_Gate_Module_1_2"
    local DST_DIR="models_inference/vpt_results_ade20k/${corruption}_layer${LAYER_STR}_len${current_p_len}_s${steps}_lrP${lr_prompt}_lrB${lr_backbone}_nb${current_num_bands}_lg${current_lambda_gate}_lf${current_lambda_f}_r${current_rho}"

    echo "Running Inference: ${corruption} | Output Dir: ${DST_DIR}"

    if ! "${PYTHON_BIN}" eval/ade20k_semantic/painter_inference_1_10.py \
      --mode inference \
      ${FC_VICT_FLAG} \
      --lambda_f ${current_lambda_f} \
      --rho ${current_rho} \
      --global_winner_path ${global_winner_file} \
      --ckpt_path ${CKPT_PATH} \
      --model ${MODEL} \
      --prompt ${PROMPT} \
      --input_dir "${INPUT_DIR}" \
      --save_dir "${DST_DIR}" \
      --steps ${steps} \
      --lr ${lr_prompt} \
      --lr_backbone ${lr_backbone} \
      --vpt_type ${VPT_TYPE} \
      --prompt_length ${current_p_len} \
      --vpt_target_layer ${current_layers} \
      --num_bands ${current_num_bands} \
      --num_images ${NUM_IMAGES} \
      --seed ${SEED} \
      --prompt_mode ${PROMPT_MODE}; then

        echo "[Error] Inference failed for ${corruption}."
        echo "${corruption},${steps},${lr_prompt},${lr_backbone},${LAYER_STR},${current_p_len},${current_num_bands},${current_lambda_gate},${current_lambda_f},${current_rho},ERROR_INFERENCE,ERROR" >> ${RESULT_LOG}
        return
    fi

    # 运行 ADE20k 评估器
    echo "Running ADE20k Evaluator..."
    if ! "${PYTHON_BIN}" eval/ade20k_semantic/ADE20kSemSegEvaluatorCustom.py \
      --pred_dir "${DST_DIR}" \
      --suffix default; then

        echo "[Error] Evaluator failed for ${corruption}."
        echo "${corruption},${steps},${lr_prompt},${lr_backbone},${LAYER_STR},${current_p_len},${current_num_bands},${current_lambda_gate},${current_lambda_f},${current_rho},ERROR_EVALUATOR,ERROR" >> ${RESULT_LOG}
        return
    fi

    # 从生成的 results.txt 中提取 mIoU
    MIOU="N/A"
    if [ -f "${DST_DIR}/eval_ade20k_default/results.txt" ]; then
        MIOU=$(tail -n 1 "${DST_DIR}/eval_ade20k_default/results.txt" | grep -o "'mIoU': [0-9.]*" | awk '{print $2}')
    fi

    WIN_STATS="N/A"
    if [ -f "${DST_DIR}/winning_groups.txt" ]; then
        WIN_STATS=$(cat "${DST_DIR}/winning_groups.txt")
    fi

    echo "${corruption},${steps},${lr_prompt},${lr_backbone},${LAYER_STR},${current_p_len},${current_num_bands},${current_lambda_gate},${current_lambda_f},${current_rho},${WIN_STATS},${MIOU}" >> ${RESULT_LOG}

    echo "[Done] Result saved to ${RESULT_LOG} (mIoU: ${MIOU})"
    echo "[Cleanup] Removing temporary inference directory: ${DST_DIR}"
    rm -rf "${DST_DIR}"
}

# =========================================================
# 主流程
# =========================================================
if [ ! -f "${RESULT_LOG}" ]; then
    echo "corruption,steps,lr_prompt,lr_backbone,layers,prompt_len,num_bands,lambda_gate,lambda_f,rho,winning_groups,mIoU" > ${RESULT_LOG}
fi

find "${CLEAN_DATA_DIR}" -type f \( -name "*.jpg" -o -name "*.png" \) > "${TEMP_DIR}/all_clean_imgs_ade20k.txt"
TOTAL_CLEAN=$(wc -l < "${TEMP_DIR}/all_clean_imgs_ade20k.txt")
CLEAN_SAMPLE=$((TOTAL_CLEAN / 10))
[ $CLEAN_SAMPLE -eq 0 ] && CLEAN_SAMPLE=1
"${PYTHON_BIN}" -c "
import random, sys
random.seed(${SEED})
lines = open(sys.argv[1]).readlines()
random.shuffle(lines)
for l in lines[:int(sys.argv[2])]: print(l, end='')
" "${TEMP_DIR}/all_clean_imgs_ade20k.txt" ${CLEAN_SAMPLE} > "${TEMP_DIR}/clean_sample_list_ade20k.txt"

for num_bands in "${NUM_BANDS_LIST[@]}"; do
  for current_rho in "${RHO_LIST[@]}"; do
    rho_safe=$(echo "${current_rho}" | tr '.' '_')
    GLOBAL_ANCHOR_CUR="${TEMP_DIR}/${GLOBAL_ANCHOR_PREFIX}_nb${num_bands}_r${rho_safe}.pt"

    if [ ! -f "${GLOBAL_ANCHOR_CUR}" ]; then
        echo ">>> [Stage 0] Generating Global Clean Anchor Stats (num_bands=${num_bands}, rho=${current_rho}) from ${CLEAN_DATA_DIR}..."
        "${PYTHON_BIN}" eval/ade20k_semantic/painter_inference_1_10.py \
          --mode anchor \
          ${FC_VICT_FLAG} \
          --rho ${current_rho} \
          --sample_list "${TEMP_DIR}/clean_sample_list_ade20k.txt" \
          --anchor_stats_path ${GLOBAL_ANCHOR_CUR} \
          --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
          --input_dir "dummy" --save_dir "dummy" \
          --vpt_type ${VPT_TYPE} --prompt_length 50 --vpt_target_layer ${TARGET_LAYERS} \
          --num_bands ${num_bands}
    else
        echo ">>> [Stage 0] Global Clean Anchor Stats already exists: ${GLOBAL_ANCHOR_CUR}"
    fi

    for corruption in "${CORRUPTIONS[@]}"; do
        echo "========================================================="
        echo ">>> [Dispatch] Task started for: ${corruption}"
        echo "========================================================="

        INPUT_DIR="datasets/ade20k-c/${corruption}/${SEVERITY}/ade/ADEChallengeData2016/images/validation/"
        CORRUPT_SAMPLE_LIST="${TEMP_DIR}/sample_list_${corruption}_nb${num_bands}_r${rho_safe}.txt"
        GLOBAL_WINNER_PREFIX="${TEMP_DIR}/global_winner_${corruption}_nb${num_bands}_r${rho_safe}"

        echo ">>> [Stage 2 & 3] Evaluating 10% samples to vote for Winner Group..."
        find "${INPUT_DIR}" -type f \( -name "*.jpg" -o -name "*.png" \) > "${TEMP_DIR}/all_imgs_${corruption}.txt"
        TOTAL_IMAGES=$(wc -l < "${TEMP_DIR}/all_imgs_${corruption}.txt")

        SAMPLE_COUNT=$((TOTAL_IMAGES / 10))
        [ $SAMPLE_COUNT -eq 0 ] && SAMPLE_COUNT=1
        "${PYTHON_BIN}" -c "
import random, sys
random.seed(${SEED})
lines = open(sys.argv[1]).readlines()
random.shuffle(lines)
for l in lines[:int(sys.argv[2])]: print(l, end='')
" "${TEMP_DIR}/all_imgs_${corruption}.txt" ${SAMPLE_COUNT} > ${CORRUPT_SAMPLE_LIST}

        NUM_IMAGES=$((TOTAL_IMAGES / 10))
        [ $NUM_IMAGES -eq 0 ] && NUM_IMAGES=1

        for lambda_gate in "${LAMBDA_GATE_LIST[@]}"; do
          for current_lambda_f in "${LAMBDA_F_LIST[@]}"; do
            lf_safe=$(echo "${current_lambda_f}" | tr '.' '_')
            GLOBAL_WINNER_FILE="${GLOBAL_WINNER_PREFIX}_lg${lambda_gate}_lf${lf_safe}.txt"

            "${PYTHON_BIN}" eval/ade20k_semantic/painter_inference_1_10.py \
              --mode eval_drift \
              ${FC_VICT_FLAG} \
              --lambda_f ${current_lambda_f} \
              --rho ${current_rho} \
              --sample_list ${CORRUPT_SAMPLE_LIST} \
              --anchor_stats_path ${GLOBAL_ANCHOR_CUR} \
              --calibration_path ${CALIBRATION_FILE} \
              --global_winner_path ${GLOBAL_WINNER_FILE} \
              --ckpt_path ${CKPT_PATH} \
              --model ${MODEL} \
              --prompt ${PROMPT} \
              --input_dir "${INPUT_DIR}" \
              --save_dir "dummy" \
              --vpt_type ${VPT_TYPE} \
              --prompt_length 50 \
              --vpt_target_layer ${TARGET_LAYERS} \
              --num_bands ${num_bands} \
              --lambda_gate ${lambda_gate} \
              --prompt_mode ${PROMPT_MODE}

            echo ">>> [Stage 4] Starting Inference Loop for Voted Winner..."
            for steps in "${STEPS_LIST[@]}"; do
              for lr_p in "${LR_PROMPT_LIST[@]}"; do
                for lr_b in "${LR_BACKBONE[@]}"; do
                  for p_len in "${PROMPT_LEN_LIST[@]}"; do
                     do_inference "${corruption}" "${steps}" "${lr_p}" "${lr_b}" "${TARGET_LAYERS}" "${p_len}" "${GLOBAL_WINNER_FILE}" "${num_bands}" "${lambda_gate}" "${current_lambda_f}" "${current_rho}"
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

echo "[Cleanup] Removing temporary directory: ${TEMP_DIR}"
rm -rf "${TEMP_DIR}"

echo "All evaluations finished! Results saved to ${RESULT_LOG}."
