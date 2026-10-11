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
SEED=2026
export PYTHONHASHSEED=${SEED}

VPT_TYPE="custom"
#40
PROMPT_LEN_LIST=(20 30 40 50 60)
TARGET_LAYERS="0 1 2 10 11 12 21 22 23"
SEVERITY="5" 

RESULT_LOG="results_lol_pixelate_oneshot.csv"

# Prompt 模式: zero-shot=our485 干净 low/high; one-shot=corruption 下 our485/low + 干净 our485/high（测试图仍为 eval15）
PROMPT_MODE="one-shot"

# =========================================================
# FC-VICT 核心超参数配置
# =========================================================
FC_VICT_FLAG="--use_fc_vict"
# 频域正则化 Loss 权重 (策略三)、低频混合权重 (策略一)

LAMBDA_F_LIST=(0.005)
RHO_LIST=(0.3)

NUM_BANDS_LIST=(6)
LAMBDA_GATE_LIST=(0.3)

STEPS_LIST=(5)
LR_PROMPT_LIST=(4e-5)
LR_BACKBONE=(0)

CORRUPTIONS=(
# "gaussian_noise"
# "shot_noise"
# "impulse_noise"
# "defocus_blur"
# "glass_blur"
# "motion_blur"
# "zoom_blur"
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
CLEAN_DATA_DIR="datasets/light_enhance/eval15/low"
GLOBAL_ANCHOR_PREFIX="anchor_stats_global_clean_lol"
CALIBRATION_FILE="calibration_baseline_lol.pt"
TEMP_DIR="temp/lol"

# 创建临时目录
mkdir -p "${TEMP_DIR}"

seeded_shuf() {
    # shuf closes the random-source pipe once it has enough bytes; Python then
    # gets BrokenPipeError — harmless, but noisy unless caught.
    shuf --random-source=<(python - <<PY
import random
import sys

rng = random.Random(${SEED})
try:
    for _ in range(256):
        sys.stdout.buffer.write(bytes(rng.randrange(256) for _ in range(4096)))
except BrokenPipeError:
    pass
PY
    ) "$@"
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
    local global_winner_file=$7
    local current_num_bands=$8
    local current_lambda_gate=$9
    local current_lambda_f=${10}
    local current_rho=${11}

    local INPUT_DIR="datasets/light_enhance-c/${corruption}/${SEVERITY}/light_enhance/eval15/low"
    if [ "$corruption" == "clean" ]; then
        INPUT_DIR="${CLEAN_DATA_DIR}"
        SEVERITY="0"
    fi

    local LAYER_STR="All_Gate_Module_1_2"
    local DST_DIR="models_inference/vpt_results_lol/${corruption}_layer${LAYER_STR}_len${current_p_len}_s${steps}_lrP${lr_prompt}_lrB${lr_backbone}_b${current_num_bands}_lg${current_lambda_gate}_lf${current_lambda_f}_r${current_rho}"

    echo "Running Inference: ${corruption} | Len: ${current_p_len} | Bands: ${current_num_bands} | Lambda_gate: ${current_lambda_gate} | lambda_f: ${current_lambda_f} | rho: ${current_rho} | Output Dir: ${DST_DIR}"

    # 执行 TTA：日志实时打印到终端，同时保留输出用于解析 PSNR/SSIM
    # 显式传入 --rho 和 --lambda_f
    OUTPUT=$(python "${SCRIPT_DIR}/painter_inference_new.py" \
      --mode inference \
      ${FC_VICT_FLAG} \
      --lambda_f ${current_lambda_f} \
      --rho ${current_rho} \
      --seed ${SEED} \
      --global_winner_path ${global_winner_file} \
      --ckpt_path ${CKPT_PATH} \
      --model ${MODEL} \
      --prompt ${PROMPT} \
      --split train-c \
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
      --lambda_gate ${current_lambda_gate} \
      --prompt_mode ${PROMPT_MODE} 2>&1)
    echo "${OUTPUT}"

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
    echo "${corruption},${steps},${lr_prompt},${lr_backbone},${LAYER_STR},${current_p_len},${current_num_bands},${current_lambda_gate},${current_lambda_f},${current_rho},${WIN_STATS},${PSNR},${SSIM}" >> ${RESULT_LOG}

    echo "[Done] Result saved to ${RESULT_LOG} (PSNR: ${PSNR}, SSIM: ${SSIM})"
    echo "[Cleanup] Removing temporary inference directory: ${DST_DIR}"
    rm -rf "${DST_DIR}"
}

# =========================================================
# 主流程
# =========================================================
if [ ! -f "${RESULT_LOG}" ]; then
    echo "corruption,steps,lr_prompt,lr_backbone,layers,prompt_len,num_bands,lambda_gate,lambda_f,rho,winning_groups,psnr,ssim" > ${RESULT_LOG}
fi

ensure_global_anchor() {
    local current_num_bands=$1
    local current_rho=$2
    local rho_safe
    rho_safe=$(echo "${current_rho}" | tr '.' '_')
    CURRENT_GLOBAL_ANCHOR="${GLOBAL_ANCHOR_PREFIX}_b${current_num_bands}_r${rho_safe}.pt"

    # [阶段 0] 全局离线计算 Anchor 统计量；不同 NUM_BANDS 需要独立缓存。
    if [ ! -f "${CURRENT_GLOBAL_ANCHOR}" ]; then
        echo ">>> [Stage 0] Generating Global Clean Anchor Stats (${current_num_bands} bands) from ${CLEAN_DATA_DIR}..."
    find "${CLEAN_DATA_DIR}" -type f \( -name "*.jpg" -o -name "*.png" \) | sort > "${TEMP_DIR}/all_clean_imgs.txt"
    
    # 限制样本数量，LOL eval15 共 15 张
    seeded_shuf -n 15 "${TEMP_DIR}/all_clean_imgs.txt" > "${TEMP_DIR}/clean_sample_list.txt"

    # 生成 Anchor 时也传入 --rho 保持一致
    python "${SCRIPT_DIR}/painter_inference_new.py" \
      --mode anchor \
      ${FC_VICT_FLAG} \
      --rho ${current_rho} \
      --seed ${SEED} \
      --sample_list "${TEMP_DIR}/clean_sample_list.txt" \
      --anchor_stats_path ${CURRENT_GLOBAL_ANCHOR} \
      --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
      --split train --perturbation clean --level 0 \
      --input_dir "dummy" --save_dir "dummy" \
      --vpt_type ${VPT_TYPE} --prompt_length 50 --vpt_target_layer ${TARGET_LAYERS} \
      --num_bands ${current_num_bands} \
      --prompt_mode ${PROMPT_MODE}
    else
        echo ">>> [Stage 0] Global Clean Anchor Stats already exists: ${CURRENT_GLOBAL_ANCHOR}"
    fi
}

# ==========================================
# 遍历各损坏数据集进行 Test-Time Adaptation
# ==========================================
for current_num_bands in "${NUM_BANDS_LIST[@]}"; do
  for current_rho in "${RHO_LIST[@]}"; do
    rho_safe=$(echo "${current_rho}" | tr '.' '_')
    ensure_global_anchor "${current_num_bands}" "${current_rho}"

    for corruption in "${CORRUPTIONS[@]}"; do
      echo "========================================================="
      echo ">>> [Dispatch] Task started for: ${corruption} | Bands: ${current_num_bands} | rho: ${current_rho}"
      echo "========================================================="

      INPUT_DIR="datasets/light_enhance-c/${corruption}/${SEVERITY}/light_enhance/eval15/low"
      if [ "$corruption" == "clean" ]; then
          INPUT_DIR="${CLEAN_DATA_DIR}"
          SEVERITY="0"
      fi
      
      CORRUPT_SAMPLE_LIST="${TEMP_DIR}/sample_list_${corruption}_b${current_num_bands}_r${rho_safe}.txt"
      GLOBAL_WINNER_PREFIX="${TEMP_DIR}/global_winner_${corruption}_b${current_num_bands}_r${rho_safe}"

      echo ">>> [Stage 2 & 3] Evaluating samples to vote for Winner Group..."
      find "${INPUT_DIR}" -type f \( -name "*.jpg" -o -name "*.png" \) | sort > "${TEMP_DIR}/all_imgs_${corruption}.txt"
      TOTAL_IMAGES=$(wc -l < "${TEMP_DIR}/all_imgs_${corruption}.txt")

      # ----------------------------------------------------
      # [恢复全量评估] val 集较小，直接使用全部样本进行 Drift 分析
      # ----------------------------------------------------
      SAMPLE_COUNT=$TOTAL_IMAGES
      seeded_shuf -n ${SAMPLE_COUNT} "${TEMP_DIR}/all_imgs_${corruption}.txt" > ${CORRUPT_SAMPLE_LIST}

      for current_lambda_gate in "${LAMBDA_GATE_LIST[@]}"; do
        for current_lambda_f in "${LAMBDA_F_LIST[@]}"; do
          lf_safe=$(echo "${current_lambda_f}" | tr '.' '_')
          GLOBAL_WINNER_FILE="${GLOBAL_WINNER_PREFIX}_lg${current_lambda_gate}_lf${lf_safe}.txt"

          echo ">>> [Stage 2 & 3] Voting Winner Group | lambda_gate: ${current_lambda_gate} | lambda_f: ${current_lambda_f} | rho: ${current_rho}"
          
          # 显式传入 --rho 和 --lambda_f
          python "${SCRIPT_DIR}/painter_inference_new.py" \
          --mode eval_drift \
          ${FC_VICT_FLAG} \
          --lambda_f ${current_lambda_f} \
          --rho ${current_rho} \
          --seed ${SEED} \
          --sample_list ${CORRUPT_SAMPLE_LIST} \
          --anchor_stats_path ${CURRENT_GLOBAL_ANCHOR} \
          --calibration_path ${CALIBRATION_FILE} \
          --global_winner_path ${GLOBAL_WINNER_FILE} \
          --ckpt_path ${CKPT_PATH} \
          --model ${MODEL} \
          --prompt ${PROMPT} \
          --split train-c --perturbation ${corruption} --level ${SEVERITY} \
          --input_dir "${INPUT_DIR}" \
          --save_dir "dummy" \
          --vpt_type ${VPT_TYPE} \
          --prompt_length 50 \
          --vpt_target_layer ${TARGET_LAYERS} \
          --num_bands ${current_num_bands} \
          --lambda_gate ${current_lambda_gate} \
          --prompt_mode ${PROMPT_MODE}

          echo ">>> [Stage 4] Starting Inference Loop for Voted Winner..."
          for steps in "${STEPS_LIST[@]}"; do
            for lr_p in "${LR_PROMPT_LIST[@]}"; do
              for lr_b in "${LR_BACKBONE[@]}"; do
                for p_len in "${PROMPT_LEN_LIST[@]}"; do
                   do_inference "${corruption}" "${steps}" "${lr_p}" "${lr_b}" "${TARGET_LAYERS}" "${p_len}" "${GLOBAL_WINNER_FILE}" "${current_num_bands}" "${current_lambda_gate}" "${current_lambda_f}" "${current_rho}"
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

# 清理统一的临时目录
echo "[Cleanup] Removing temporary directory: ${TEMP_DIR}"
rm -rf "${TEMP_DIR}"

echo "All evaluations finished! Results saved to ${RESULT_LOG}."