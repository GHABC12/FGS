#!/bin/bash
set -e

# =========================================================
# 环境配置
# =========================================================
# 屏蔽各种Python和底层库的警告信息，使终端输出简洁
export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL=3

# =========================================================
# 配置区
# =========================================================
CKPT_FILE="painter_vit_large/painter_vit_large.pth"
PROMPT="ADE_train_00014165"  # ADE20k 默认 prompt
MODEL="painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1"
CKPT_PATH="models/${CKPT_FILE}"

VPT_TYPE="custom"
PROMPT_LEN_LIST=(100)
TARGET_LAYERS="0 1 2 10 11 12 21 22 23"
SEVERITY=5
RESULT_LOG="results_ade20k.csv"

# [新增参数]
NUM_BANDS=10       # 径向频带划分数
LAMBDA_GATE=0.5    # 融合权重 (0.5表示频域和效用各占50%)

STEPS_LIST=(5)
LR_PROMPT_LIST=(5e-5 1e-4)
LR_BACKBONE=(0)

CORRUPTIONS=(
"gaussian_noise"
# "shot_noise"
# "impulse_noise"
# "defocus_blur"
# "glass_blur"
# "motion_blur"
# "zoom_blur"
"snow"
# "frost"
"fog"
# "brightness"
#"contrast"
# "elastic_transform"
#"pixelate"
#"jpeg_compression"
)

# 全局文件配置
CLEAN_DATA_DIR="datasets/ade20k/images/validation/"
GLOBAL_ANCHOR="anchor_stats_global_clean_ade20k.pt"
CALIBRATION_FILE="calibration_baseline_ade20k.pt"
TEMP_DIR="temp/ade20k_semantic"

# 创建临时目录
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

    local INPUT_DIR="datasets/ade20k-c/${corruption}/${SEVERITY}/ade/ADEChallengeData2016/images/validation/"
    local LAYER_STR="All_Gate_Module_1_2"
    local DST_DIR="models_inference/vpt_results_ade20k/${corruption}_layer${LAYER_STR}_len${current_p_len}_s${steps}_lrP${lr_prompt}_lrB${lr_backbone}"

    echo "Running Inference: ${corruption} | Len: ${current_p_len} | Output Dir: ${DST_DIR}"

    if ! python eval/ade20k_semantic/painter_inference.py \
      --mode inference \
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
      --num_bands ${NUM_BANDS}; then

        echo "[Error] Inference failed for ${corruption}."
        echo "${corruption},${steps},${lr_prompt},${lr_backbone},${LAYER_STR},${current_p_len},ERROR_INFERENCE,ERROR" >> ${RESULT_LOG}
        return
    fi

    # 运行 ADE20k 评估器
    echo "Running ADE20k Evaluator..."
    if ! python eval/ade20k_semantic/ADE20kSemSegEvaluatorCustom.py \
      --pred_dir "${DST_DIR}" \
      --suffix default; then

        echo "[Error] Evaluator failed for ${corruption}."
        echo "${corruption},${steps},${lr_prompt},${lr_backbone},${LAYER_STR},${current_p_len},ERROR_EVALUATOR,ERROR" >> ${RESULT_LOG}
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

    # 单线程执行，直接追加写入结果即可，无需文件锁
    echo "${corruption},${steps},${lr_prompt},${lr_backbone},${LAYER_STR},${current_p_len},${WIN_STATS},${MIOU}" >> ${RESULT_LOG}

    echo "[Done] Result saved to ${RESULT_LOG} (mIoU: ${MIOU})"
    echo "[Cleanup] Removing temporary inference directory: ${DST_DIR}"
    rm -rf "${DST_DIR}"
}

# =========================================================
# 主流程
# =========================================================
if [ ! -f "${RESULT_LOG}" ]; then
    echo "corruption,steps,lr_prompt,lr_backbone,layers,prompt_len,winning_groups,mIoU" > ${RESULT_LOG}
fi

# [阶段 0] 全局离线计算 Anchor 统计量 (使用 Clean Validation Set)
if [ ! -f "${GLOBAL_ANCHOR}" ]; then
    echo ">>> [Stage 0] Generating Global Clean Anchor Stats from ${CLEAN_DATA_DIR}..."
    find "${CLEAN_DATA_DIR}" -type f \( -name "*.jpg" -o -name "*.png" \) > "${TEMP_DIR}/all_clean_imgs_ade20k.txt"
    shuf -n 65 "${TEMP_DIR}/all_clean_imgs_ade20k.txt" > "${TEMP_DIR}/clean_sample_list_ade20k.txt"

    python eval/ade20k_semantic/painter_inference.py \
      --mode anchor \
      --sample_list "${TEMP_DIR}/clean_sample_list_ade20k.txt" \
      --anchor_stats_path ${GLOBAL_ANCHOR} \
      --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
      --input_dir "dummy" --save_dir "dummy" \
      --vpt_type ${VPT_TYPE} --prompt_length 50 --vpt_target_layer ${TARGET_LAYERS} \
      --num_bands ${NUM_BANDS}
else
    echo ">>> [Stage 0] Global Clean Anchor Stats already exists: ${GLOBAL_ANCHOR}"
fi

# ==========================================
# 遍历各损坏数据集进行 Test-Time Adaptation (单线程串行执行)
# ==========================================
for corruption in "${CORRUPTIONS[@]}"; do
    echo "========================================================="
    echo ">>> [Dispatch] Task started for: ${corruption}"
    echo "========================================================="

    INPUT_DIR="datasets/ade20k-c/${corruption}/${SEVERITY}/ade/ADEChallengeData2016/images/validation/"
    CORRUPT_SAMPLE_LIST="${TEMP_DIR}/sample_list_${corruption}_ade20k.txt"
    GLOBAL_WINNER_FILE="${TEMP_DIR}/global_winner_${corruption}_ade20k.txt"

    echo ">>> [Stage 2 & 3] Evaluating 10% samples to vote for Winner Group..."
    find "${INPUT_DIR}" -type f \( -name "*.jpg" -o -name "*.png" \) > "${TEMP_DIR}/all_imgs_${corruption}_ade20k.txt"
    TOTAL_IMAGES=$(wc -l < "${TEMP_DIR}/all_imgs_${corruption}_ade20k.txt")

    SAMPLE_COUNT=$((TOTAL_IMAGES / 10))
    [ $SAMPLE_COUNT -eq 0 ] && SAMPLE_COUNT=1
    shuf -n ${SAMPLE_COUNT} "${TEMP_DIR}/all_imgs_${corruption}_ade20k.txt" > ${CORRUPT_SAMPLE_LIST}

    VOTE_PROMPT_LEN=${PROMPT_LEN_LIST[0]}
    python eval/ade20k_semantic/painter_inference.py \
      --mode eval_drift \
      --sample_list ${CORRUPT_SAMPLE_LIST} \
      --anchor_stats_path ${GLOBAL_ANCHOR} \
      --calibration_path ${CALIBRATION_FILE} \
      --global_winner_path ${GLOBAL_WINNER_FILE} \
      --ckpt_path ${CKPT_PATH} \
      --model ${MODEL} \
      --prompt ${PROMPT} \
      --input_dir "${INPUT_DIR}" \
      --save_dir "dummy" \
      --vpt_type ${VPT_TYPE} \
      --prompt_length ${VOTE_PROMPT_LEN} \
      --vpt_target_layer ${TARGET_LAYERS} \
      --num_bands ${NUM_BANDS} \
      --lambda_gate ${LAMBDA_GATE}

    echo ">>> [Stage 4] Starting Inference Loop for Voted Winner..."
    for steps in "${STEPS_LIST[@]}"; do
      for lr_p in "${LR_PROMPT_LIST[@]}"; do
        for lr_b in "${LR_BACKBONE[@]}"; do
          for p_len in "${PROMPT_LEN_LIST[@]}"; do
             do_inference "${corruption}" "${steps}" "${lr_p}" "${lr_b}" "${TARGET_LAYERS}" "${p_len}" "${GLOBAL_WINNER_FILE}"
          done
        done
      done
    done

    echo ">>> [Success] Evaluation completed for: ${corruption}"
    echo ""
done

# 清理统一的临时目录
echo "[Cleanup] Removing temporary directory: ${TEMP_DIR}"
rm -rf "${TEMP_DIR}"

echo "All evaluations finished! Results saved to ${RESULT_LOG}."