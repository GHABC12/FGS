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
PROMPT="study_room_0005b/rgb_00094"
MODEL="painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1"
CKPT_PATH="models/${CKPT_FILE}"

VPT_TYPE="custom"
PROMPT_LEN_LIST=(100)
TARGET_LAYERS="0 1 2 10 11 12 21 22 23"
SEVERITY=5
RESULT_LOG="results_nyu.csv"

# [新增参数：支持列表]
NUM_BANDS_LIST=(5)       # 径向频带划分数
LAMBDA_GATE_LIST=(0.4)    # 融合权重 (0.5表示频域和效用各占50%)

#在所有数据集上试一下10 5e-6 和15 1e-5
STEPS_LIST=(0)
LR_PROMPT_LIST=(5e-6)
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
CLEAN_DATA_DIR="datasets/nyu_depth_v2/sync/"
GLOBAL_ANCHOR="anchor_stats_global_clean.pt"
CALIBRATION_FILE="calibration_baseline.pt"
TEMP_DIR="temp/nyuv2_depth"

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

    local INPUT_DIR="datasets/nyu_depth_v2/nyu_depth_v2_corrupted/${corruption}/${SEVERITY}/nyu_depth_v2/official_splits/test/"
    local LAYER_STR="All_Gate_Module_1_2"
    local DST_DIR="models_inference/vpt_results/${corruption}_layer${LAYER_STR}_len${current_p_len}_s${steps}_lrP${lr_prompt}_lrB${lr_backbone}_nb${current_num_bands}_lg${current_lambda_gate}"

    echo "Running Inference: ${corruption} | Len: ${current_p_len} | Output Dir: ${DST_DIR}"

    python eval/nyuv2_depth/painter_inference.py \
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
      --num_bands ${current_num_bands}

    if [ $? -ne 0 ]; then
        echo "[Error] Inference failed for ${corruption}."
        return
    fi

    OUTPUT=$(python eval/nyuv2_depth/eval_with_pngs_test.py \
      --pred_path "${DST_DIR}" \
      --gt_path "datasets/nyu_depth_v2/official_splits/test/" \
      --dataset nyu \
      --min_depth_eval 1e-3 --max_depth_eval 10 \
      --eigen_crop)

    STATS=$(echo "$OUTPUT" | awk 'NF>0' | tail -n 1)
    ABS_REL=$(echo "$STATS" | tr ',' ' ' | awk '{print $2}')

    if [ -z "$ABS_REL" ]; then
        ABS_REL="$STATS"
        echo "[Warning] 提取第 2 列失败，已将完整指标数据保存入 CSV 避免丢失。"
    fi

    WIN_STATS="N/A"
    if [ -f "${DST_DIR}/winning_groups.txt" ]; then
        WIN_STATS=$(cat "${DST_DIR}/winning_groups.txt")
    fi

    # 单线程执行，直接追加写入结果即可，无需文件锁
    echo "${corruption},${steps},${lr_prompt},${lr_backbone},${LAYER_STR},${current_p_len},${current_num_bands},${current_lambda_gate},${WIN_STATS},${ABS_REL}" >> ${RESULT_LOG}

    echo "[Done] Result saved to ${RESULT_LOG} (abs_rel: ${ABS_REL})"
    echo "[Cleanup] Removing temporary inference directory: ${DST_DIR}"
    rm -rf "${DST_DIR}"
}

# =========================================================
# 主流程
# =========================================================
if [ ! -f "${RESULT_LOG}" ]; then
    echo "corruption,steps,lr_prompt,lr_backbone,layers,prompt_len,num_bands,lambda_gate,winning_groups,abs_rel" > ${RESULT_LOG}
fi

find "${CLEAN_DATA_DIR}" -type f \( -name "*.jpg" -o -name "*.png" \) | grep -i "rgb" > "${TEMP_DIR}/all_clean_imgs.txt"
shuf -n 65 "${TEMP_DIR}/all_clean_imgs.txt" > "${TEMP_DIR}/clean_sample_list.txt"

# ==========================================
# 遍历各损坏数据集进行 Test-Time Adaptation (单线程串行执行)
# ==========================================
for num_bands in "${NUM_BANDS_LIST[@]}"; do
  GLOBAL_ANCHOR_CUR="${TEMP_DIR}/anchor_stats_global_clean_nb${num_bands}.pt"
  if [ ! -f "${GLOBAL_ANCHOR_CUR}" ]; then
      echo ">>> [Stage 0] Generating Global Clean Anchor Stats (num_bands=${num_bands}) from ${CLEAN_DATA_DIR}..."
      python eval/nyuv2_depth/painter_inference.py \
        --mode anchor \
        --sample_list "${TEMP_DIR}/clean_sample_list.txt" \
        --anchor_stats_path ${GLOBAL_ANCHOR_CUR} \
        --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
        --input_dir "dummy" --save_dir "dummy" \
        --vpt_type ${VPT_TYPE} --prompt_length 50 --vpt_target_layer ${TARGET_LAYERS} \
        --num_bands ${num_bands}
  else
      echo ">>> [Stage 0] Global Clean Anchor Stats already exists: ${GLOBAL_ANCHOR_CUR}"
  fi

  for lambda_gate in "${LAMBDA_GATE_LIST[@]}"; do
for corruption in "${CORRUPTIONS[@]}"; do
    echo "========================================================="
    echo ">>> [Dispatch] Task started for: ${corruption}"
    echo "========================================================="

    INPUT_DIR="datasets/nyu_depth_v2/nyu_depth_v2_corrupted/${corruption}/${SEVERITY}/nyu_depth_v2/official_splits/test/"
    CORRUPT_SAMPLE_LIST="${TEMP_DIR}/sample_list_${corruption}.txt"
    GLOBAL_WINNER_FILE="${TEMP_DIR}/global_winner_${corruption}_nb${num_bands}_lg${lambda_gate}.txt"

    echo ">>> [Stage 2 & 3] Evaluating 10% samples to vote for Winner Group..."
    find "${INPUT_DIR}" -type f \( -name "*.jpg" -o -name "*.png" \) > "${TEMP_DIR}/all_imgs_${corruption}.txt"
    TOTAL_IMAGES=$(wc -l < "${TEMP_DIR}/all_imgs_${corruption}.txt")

    SAMPLE_COUNT=$((TOTAL_IMAGES / 10))
    [ $SAMPLE_COUNT -eq 0 ] && SAMPLE_COUNT=1
    shuf -n ${SAMPLE_COUNT} "${TEMP_DIR}/all_imgs_${corruption}.txt" > ${CORRUPT_SAMPLE_LIST}

    python eval/nyuv2_depth/painter_inference.py \
      --mode eval_drift \
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