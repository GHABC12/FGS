#!/bin/bash
set -e

# =========================================================
# 环境配置
# =========================================================
# 屏蔽各种Python和底层库的警告信息，使终端输出简洁
export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL=3
# 优先使用 conda 的 libstdc++，避免 torch 导入时命中系统旧版 CXXABI。
CONDA_LIB="/root/miniconda3/lib"
if [ -d "${CONDA_LIB}" ]; then
  case ":${LD_LIBRARY_PATH}:" in
    *":${CONDA_LIB}:"*) ;;
    *) export LD_LIBRARY_PATH="${CONDA_LIB}:${LD_LIBRARY_PATH}" ;;
  esac
fi
CONDA_NCCL="${CONDA_LIB}/libnccl.so.2"
if [ -f "${CONDA_NCCL}" ]; then
  case ":${LD_PRELOAD}:" in
    *":${CONDA_NCCL}:"*) ;;
    *) export LD_PRELOAD="${CONDA_NCCL}${LD_PRELOAD:+:${LD_PRELOAD}}" ;;
  esac
fi

# =========================================================
# 配置区
# =========================================================
CKPT_FILE="painter_vit_large/painter_vit_large.pth"
PROMPT="100"
MODEL="painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1"
CKPT_PATH="models/${CKPT_FILE}"

VPT_TYPE="custom"
PROMPT_LEN_LIST=(100)
TARGET_LAYERS="0 1 2 10 11 12 21 22 23"
SEVERITY=5
RESULT_LOG="results_derain.csv"

NUM_BANDS_LIST=(6 7)
LAMBDA_GATE_LIST=(0.35 0.4 0.45)

STEPS_LIST=(9)
LR_PROMPT_LIST=(1.2e-4)
LR_BACKBONE=(0)

CORRUPTIONS=(
# "gaussian_noise"
# "shot_noise"
"impulse_noise"
# "defocus_blur"
# "glass_blur"
# "motion_blur"
"zoom_blur"
# "snow"
# "frost"
# "fog"
# "brightness"
# "contrast"
# "elastic_transform"
"pixelate"
# "jpeg_compression"
)

# 全局文件配置
CLEAN_DATA_DIR="datasets/derain/train/input/"
CORRUPTED_ROOT="datasets/derain-c"
GT_ROOT="datasets/derain/test"
GLOBAL_ANCHOR="anchor_stats_global_clean_derain.pt"
CALIBRATION_FILE="calibration_baseline_derain.pt"
RAW_DRIFT_DIR="raw_drifts_derain"
TEMP_DIR="temp/derain"

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

    local INPUT_DIR="${CORRUPTED_ROOT}/${corruption}/${SEVERITY}/derain/test/"
    local LAYER_STR="All_Gate_Module_1_2"
    local DST_DIR="models_inference/vpt_results_derain/${corruption}_layer${LAYER_STR}_bands${current_num_bands}_gate${current_lambda_gate}_len${current_p_len}_s${steps}_lrP${lr_prompt}_lrB${lr_backbone}"
    local SAVE_DIR="${DST_DIR}/results"

    echo "Running Inference: ${corruption} | Bands: ${current_num_bands} | Lambda: ${current_lambda_gate} | Len: ${current_p_len} | Output Dir: ${DST_DIR}"

    python eval/derain/painter_inference.py \
      --mode inference \
      --global_winner_path ${global_winner_file} \
      --ckpt_path ${CKPT_PATH} \
      --model ${MODEL} \
      --prompt ${PROMPT} \
      --input_dir "${INPUT_DIR}" \
      --save_dir "${SAVE_DIR}" \
      --steps ${steps} \
      --lr ${lr_prompt} \
      --lr_backbone ${lr_backbone} \
      --vpt_type ${VPT_TYPE} \
      --prompt_length ${current_p_len} \
      --vpt_target_layer ${current_layers} \
      --num_bands ${current_num_bands}

    if [ $? -ne 0 ]; then
        echo "[Error] Inference failed for ${corruption}."
        return
    fi

    OUTPUT=$(python eval/derain/evaluate_PSNR_SSIM.py \
      --pred_path "${DST_DIR}" \
      --gt_path "${GT_ROOT}")

    STATS=$(echo "$OUTPUT" | grep -i "For all datasets" | tail -n 1)
    PSNR=$(echo "$STATS" | awk -F'PSNR:' '{print $2}' | awk '{print $1}')
    SSIM=$(echo "$STATS" | awk -F'SSIM:' '{print $2}' | awk '{print $1}')

    if [ -z "$PSNR" ]; then
        PSNR="$STATS"
        echo "[Warning] 提取 PSNR 失败，已将完整指标数据保存入 CSV 避免丢失。"
    fi

    WIN_STATS="N/A"
    if [ -f "${SAVE_DIR}/winning_groups.txt" ]; then
        WIN_STATS=$(cat "${SAVE_DIR}/winning_groups.txt")
    fi

    # 单线程执行，直接追加写入结果即可
    echo "${corruption},${steps},${lr_prompt},${lr_backbone},${LAYER_STR},${current_p_len},${current_num_bands},${current_lambda_gate},${WIN_STATS},${PSNR},${SSIM}" >> ${RESULT_LOG}

    echo "[Done] Result saved to ${RESULT_LOG} (PSNR: ${PSNR} SSIM: ${SSIM})"
    echo "[Cleanup] Removing temporary inference directory: ${DST_DIR}"
    rm -rf "${DST_DIR}"
}

# =========================================================
# 主流程
# =========================================================
if [ ! -f "${RESULT_LOG}" ]; then
    echo "corruption,steps,lr_prompt,lr_backbone,layers,prompt_len,num_bands,lambda_gate,winning_groups,psnr,ssim" > ${RESULT_LOG}
fi

for num_bands in "${NUM_BANDS_LIST[@]}"; do
  CURRENT_GLOBAL_ANCHOR="${GLOBAL_ANCHOR}"
  if [ "${num_bands}" != "10" ]; then
    CURRENT_GLOBAL_ANCHOR="${GLOBAL_ANCHOR%.pt}_bands${num_bands}.pt"
  fi

  # [阶段 0] 全局离线计算 Anchor 统计量
  if [ ! -f "${CURRENT_GLOBAL_ANCHOR}" ]; then
      echo ">>> [Stage 0] Generating Global Clean Anchor Stats from ${CLEAN_DATA_DIR}..."
      find "${CLEAN_DATA_DIR}" -type f \( -name "*.jpg" -o -name "*.png" \) > "${TEMP_DIR}/all_clean_imgs_derain.txt"
      TOTAL_CLEAN_IMAGES=$(wc -l < "${TEMP_DIR}/all_clean_imgs_derain.txt")
      CLEAN_SAMPLE_COUNT=$((TOTAL_CLEAN_IMAGES / 10))
      [ $CLEAN_SAMPLE_COUNT -gt 100 ] && CLEAN_SAMPLE_COUNT=100
      [ $CLEAN_SAMPLE_COUNT -eq 0 ] && CLEAN_SAMPLE_COUNT=1
      shuf -n ${CLEAN_SAMPLE_COUNT} "${TEMP_DIR}/all_clean_imgs_derain.txt" > "${TEMP_DIR}/clean_sample_list_derain.txt"

      python eval/derain/painter_inference.py \
        --mode anchor \
        --sample_list "${TEMP_DIR}/clean_sample_list_derain.txt" \
        --anchor_stats_path ${CURRENT_GLOBAL_ANCHOR} \
        --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
        --input_dir "dummy" --save_dir "dummy" \
        --vpt_type ${VPT_TYPE} --prompt_length 50 --vpt_target_layer ${TARGET_LAYERS} \
        --num_bands ${num_bands}
  else
      echo ">>> [Stage 0] Global Clean Anchor Stats already exists: ${CURRENT_GLOBAL_ANCHOR}"
  fi

  for lambda_gate in "${LAMBDA_GATE_LIST[@]}"; do
    # ==========================================
    # 遍历各损坏数据集进行 Test-Time Adaptation
    # ==========================================
    for corruption in "${CORRUPTIONS[@]}"; do
      echo "========================================================="
      echo ">>> [Dispatch] Task started for: ${corruption}"
      echo "========================================================="
      
      INPUT_DIR="${CORRUPTED_ROOT}/${corruption}/${SEVERITY}/derain/test/"
      CORRUPT_SAMPLE_LIST="${TEMP_DIR}/sample_list_${corruption}_derain.txt"
      GLOBAL_WINNER_FILE="${TEMP_DIR}/global_winner_${corruption}_bands${num_bands}_gate${lambda_gate}_derain.txt"

      if [ ! -d "${INPUT_DIR}" ]; then
        echo ">>> [Skip] ${INPUT_DIR} not found, skipping ${corruption}."
        continue
      fi

      echo ">>> [Stage 2 & 3] Evaluating 10% samples to vote for Winner Group..."
      find "${INPUT_DIR}" -type f \( -name "*.jpg" -o -name "*.png" \) > "${TEMP_DIR}/all_imgs_${corruption}_derain.txt"
      TOTAL_IMAGES=$(wc -l < "${TEMP_DIR}/all_imgs_${corruption}_derain.txt")

      SAMPLE_COUNT=$((TOTAL_IMAGES / 10))
      [ $SAMPLE_COUNT -eq 0 ] && SAMPLE_COUNT=1
      shuf -n ${SAMPLE_COUNT} "${TEMP_DIR}/all_imgs_${corruption}_derain.txt" > ${CORRUPT_SAMPLE_LIST}

      python eval/derain/painter_inference.py \
        --mode eval_drift \
        --sample_list ${CORRUPT_SAMPLE_LIST} \
        --anchor_stats_path ${CURRENT_GLOBAL_ANCHOR} \
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
        --lambda_gate ${lambda_gate}

      echo ">>> [Stage 4] Starting Inference Loop for Voted Winner..."
      for steps in "${STEPS_LIST[@]}"; do
        for lr_p in "${LR_PROMPT_LIST[@]}"; do
          for lr_b in "${LR_BACKBONE[@]}"; do
            for p_len in "${PROMPT_LEN_LIST[@]}"; do
               do_inference "${corruption}" "${steps}" "${lr_p}" "${lr_b}" "${TARGET_LAYERS}" "${p_len}" "${GLOBAL_WINNER_FILE}" "${num_bands}" "${lambda_gate}"
            done
          done
        done
      done

      echo ">>> [Success] Evaluation completed for: ${corruption}"
      echo ""
    done
  done
done

# 清理统一的临时目录
echo "[Cleanup] Removing temporary directory: ${TEMP_DIR}"
rm -rf "${TEMP_DIR}"

echo "All evaluations finished! Results saved to ${RESULT_LOG}."