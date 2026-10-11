#!/bin/bash
set -e

# Ensure all relative paths resolve from this script directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

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
PROMPT="100" 
MODEL="painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1"
CKPT_PATH="models/${CKPT_FILE}"

VPT_TYPE="custom"
PROMPT_LEN_LIST=(40)
TARGET_LAYERS="0 1 2 10 11 12 21 22 23"
SEVERITY="5" 
RESULT_LOG="results_lol.csv"

# 最佳参数7 1e-4
NUM_BANDS_LIST=(6)       # 径向频带划分数
LAMBDA_GATE_LIST=(0.5)   # 融合权重 (0.5表示频域和效用各占50%)
#最佳参数7 1e-4
STEPS_LIST=(7)
LR_PROMPT_LIST=(1e-4)
LR_BACKBONE=(0)

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

# 全局文件配置
CLEAN_DATA_DIR="datasets/light_enhance/eval15/low"
GLOBAL_ANCHOR_PREFIX="anchor_stats_global_clean_lol"
CALIBRATION_FILE="calibration_baseline_lol.pt"
TEMP_DIR="temp/lol"

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
    local current_num_bands=$8
    local current_lambda_gate=$9

    local INPUT_DIR="datasets/light_enhance-c/${corruption}/${SEVERITY}/light_enhance/eval15/low"
    if [ "$corruption" == "clean" ]; then
        INPUT_DIR="${CLEAN_DATA_DIR}"
        SEVERITY="0"
    fi

    local LAYER_STR="All_Gate_Module_1_2"
    local DST_DIR="models_inference/vpt_results_lol/${corruption}_layer${LAYER_STR}_len${current_p_len}_s${steps}_lrP${lr_prompt}_lrB${lr_backbone}_b${current_num_bands}_lg${current_lambda_gate}"

    echo "Running Inference: ${corruption} | Len: ${current_p_len} | Bands: ${current_num_bands} | Lambda: ${current_lambda_gate} | Output Dir: ${DST_DIR}"

    # 执行 TTA：日志实时打印到终端，同时保留输出用于解析 PSNR/SSIM
    OUTPUT=$(python "${SCRIPT_DIR}/painter_inference.py" \
      --mode inference \
      --global_winner_path ${global_winner_file} \
      --ckpt_path ${CKPT_PATH} \
      --model ${MODEL} \
      --prompt ${PROMPT} \
      --split train \
      --perturbation ${corruption} \
      --level ${SEVERITY} \
      --input_dir "${INPUT_DIR}" \
      --save_dir "${DST_DIR}" \
      --steps ${steps} \
      --lr ${lr_prompt} \
      --lr_backbone ${lr_backbone} \
      --vpt_type ${VPT_TYPE} \
      --prompt_length ${current_p_len} \
      --vpt_target_layer ${current_layers} \
      --num_bands ${current_num_bands} \
      --lambda_gate ${current_lambda_gate} | tee /dev/tty)

    if [ $? -ne 0 ]; then
        echo "[Error] Inference failed for ${corruption}."
        return
    fi

    # 从 Python 脚本的标准输出中提取 PSNR 和 SSIM
    PSNR=$(echo "$OUTPUT" | grep "FINAL_METRICS:" | awk -F'PSNR: ' '{print $2}' | awk -F',' '{print $1}')
    SSIM=$(echo "$OUTPUT" | grep "FINAL_METRICS:" | awk -F'SSIM: ' '{print $2}' | awk '{print $1}')

    if [ -z "$PSNR" ]; then
        PSNR="N/A"
        SSIM="N/A"
        echo "[Warning] 提取指标数据失败。"
    fi

    WIN_STATS="N/A"
    if [ -f "${DST_DIR}/winning_groups.txt" ]; then
        WIN_STATS=$(cat "${DST_DIR}/winning_groups.txt")
    fi

    # 单线程执行，直接追加写入结果即可，无需文件锁
    echo "${corruption},${steps},${lr_prompt},${lr_backbone},${LAYER_STR},${current_p_len},${current_num_bands},${current_lambda_gate},${WIN_STATS},${PSNR},${SSIM}" >> ${RESULT_LOG}

    echo "[Done] Result saved to ${RESULT_LOG} (PSNR: ${PSNR}, SSIM: ${SSIM})"
    echo "[Cleanup] Removing temporary inference directory: ${DST_DIR}"
    rm -rf "${DST_DIR}"
}

# =========================================================
# 主流程
# =========================================================
if [ ! -f "${RESULT_LOG}" ]; then
    echo "corruption,steps,lr_prompt,lr_backbone,layers,prompt_len,num_bands,lambda_gate,winning_groups,psnr,ssim" > ${RESULT_LOG}
fi

ensure_global_anchor() {
    local current_num_bands=$1
    CURRENT_GLOBAL_ANCHOR="${GLOBAL_ANCHOR_PREFIX}_b${current_num_bands}.pt"

    # [阶段 0] 全局离线计算 Anchor 统计量；不同 NUM_BANDS 需要独立缓存。
    if [ ! -f "${CURRENT_GLOBAL_ANCHOR}" ]; then
        echo ">>> [Stage 0] Generating Global Clean Anchor Stats (${current_num_bands} bands) from ${CLEAN_DATA_DIR}..."
    find "${CLEAN_DATA_DIR}" -type f \( -name "*.jpg" -o -name "*.png" \) > "${TEMP_DIR}/all_clean_imgs.txt"
    
    # 限制样本数量，LOL eval15 共 15 张
    shuf -n 15 "${TEMP_DIR}/all_clean_imgs.txt" > "${TEMP_DIR}/clean_sample_list.txt"

    python "${SCRIPT_DIR}/painter_inference.py" \
      --mode anchor \
      --sample_list "${TEMP_DIR}/clean_sample_list.txt" \
      --anchor_stats_path ${CURRENT_GLOBAL_ANCHOR} \
      --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
      --split train --perturbation clean --level 0 \
      --input_dir "dummy" --save_dir "dummy" \
      --vpt_type ${VPT_TYPE} --prompt_length 50 --vpt_target_layer ${TARGET_LAYERS} \
      --num_bands ${current_num_bands}
    else
        echo ">>> [Stage 0] Global Clean Anchor Stats already exists: ${CURRENT_GLOBAL_ANCHOR}"
    fi
}

# ==========================================
# 遍历各损坏数据集进行 Test-Time Adaptation
# ==========================================
for current_num_bands in "${NUM_BANDS_LIST[@]}"; do
  ensure_global_anchor "${current_num_bands}"

  for corruption in "${CORRUPTIONS[@]}"; do
    echo "========================================================="
    echo ">>> [Dispatch] Task started for: ${corruption} | Bands: ${current_num_bands}"
    echo "========================================================="

    INPUT_DIR="datasets/light_enhance-c/${corruption}/${SEVERITY}/light_enhance/eval15/low"
    if [ "$corruption" == "clean" ]; then
        INPUT_DIR="${CLEAN_DATA_DIR}"
        SEVERITY="0"
    fi
    
    CORRUPT_SAMPLE_LIST="${TEMP_DIR}/sample_list_${corruption}.txt"
    GLOBAL_WINNER_PREFIX="${TEMP_DIR}/global_winner_${corruption}_b${current_num_bands}"

    echo ">>> [Stage 2 & 3] Evaluating samples to vote for Winner Group..."
    find "${INPUT_DIR}" -type f \( -name "*.jpg" -o -name "*.png" \) > "${TEMP_DIR}/all_imgs_${corruption}.txt"
    TOTAL_IMAGES=$(wc -l < "${TEMP_DIR}/all_imgs_${corruption}.txt")

    # ----------------------------------------------------
    # [恢复全量评估] val 集较小，直接使用全部样本进行 Drift 分析
    # ----------------------------------------------------
    SAMPLE_COUNT=$TOTAL_IMAGES
    shuf -n ${SAMPLE_COUNT} "${TEMP_DIR}/all_imgs_${corruption}.txt" > ${CORRUPT_SAMPLE_LIST}

    for current_lambda_gate in "${LAMBDA_GATE_LIST[@]}"; do
      GLOBAL_WINNER_FILE="${GLOBAL_WINNER_PREFIX}_lg${current_lambda_gate}.txt"

      echo ">>> [Stage 2 & 3] Voting Winner Group | Lambda: ${current_lambda_gate}"
      python "${SCRIPT_DIR}/painter_inference.py" \
      --mode eval_drift \
      --sample_list ${CORRUPT_SAMPLE_LIST} \
      --anchor_stats_path ${CURRENT_GLOBAL_ANCHOR} \
      --calibration_path ${CALIBRATION_FILE} \
      --global_winner_path ${GLOBAL_WINNER_FILE} \
      --ckpt_path ${CKPT_PATH} \
      --model ${MODEL} \
      --prompt ${PROMPT} \
      --split train --perturbation ${corruption} --level ${SEVERITY} \
      --input_dir "${INPUT_DIR}" \
      --save_dir "dummy" \
      --vpt_type ${VPT_TYPE} \
      --prompt_length 50 \
      --vpt_target_layer ${TARGET_LAYERS} \
      --num_bands ${current_num_bands} \
      --lambda_gate ${current_lambda_gate}

      echo ">>> [Stage 4] Starting Inference Loop for Voted Winner..."
      for steps in "${STEPS_LIST[@]}"; do
        for lr_p in "${LR_PROMPT_LIST[@]}"; do
          for lr_b in "${LR_BACKBONE[@]}"; do
            for p_len in "${PROMPT_LEN_LIST[@]}"; do
               do_inference "${corruption}" "${steps}" "${lr_p}" "${lr_b}" "${TARGET_LAYERS}" "${p_len}" "${GLOBAL_WINNER_FILE}" "${current_num_bands}" "${current_lambda_gate}"
            done
          done
        done
      done
    done

    echo ">>> [Success] Evaluation completed for: ${corruption}"
    echo ""
  done
done

# 清理统一的临时目录
echo "[Cleanup] Removing temporary directory: ${TEMP_DIR}"
rm -rf "${TEMP_DIR}"

echo "All evaluations finished! Results saved to ${RESULT_LOG}."