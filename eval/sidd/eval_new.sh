#!/bin/bash
set -e

# =========================================================
# 环境配置
# =========================================================
export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL=3

# =========================================================
# 配置区
# =========================================================
CKPT_FILE="painter_vit_large/painter_vit_large.pth"
PROMPT="9_9" # SIDD 使用的 Prompt 图片
MODEL="painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1"
CKPT_PATH="models/${CKPT_FILE}"

VPT_TYPE="custom"
PROMPT_LEN_LIST=(100)
TARGET_LAYERS="0 1 2 10 11 12 21 22 23"
SEVERITY=5
RESULT_LOG="results_sidd.csv"

# =========================================================
# Shot Mode 配置
# =========================================================
# zero-shot: 提示图使用干净图片 (datasets/denoise/train/input/9_9.png)
# one-shot:  提示图使用对应 corruption 的损坏图片 (datasets/denoise-c/<corruption>/5/denoise/train/input/9_9.png)
SHOT_MODE="one-shot"
#5 0.3
NUM_BANDS_LIST=(5)            # 径向频带划分数（可填多个，与下方笛卡尔组合遍历）
LAMBDA_GATE_LIST=(0.2)         # 融合权重 (0.5表示频域和效用各占50%)

# =========================================================
# FC-VICT 核心超参数配置 (防止降速与崩溃版本)
# =========================================================
FC_VICT_FLAG="--use_fc_vict"
#0.01 0.1
LAMBDA_F_LIST=(0.005)    # 频域正则化权重
RHO_LIST=(0.05)          # 低频混合权重
#5 3e-6
STEPS_LIST=(3 5 8)
LR_PROMPT_LIST=(1e-6 5e-6 1e-5) 
LR_BACKBONE=(0)

# SIDD 的损坏类型
CORRUPTIONS=(
"gaussian_noise"
)

# 全局文件配置
CLEAN_DATA_DIR="datasets/denoise/val/input/"
GT_DATA_DIR="datasets/denoise/val/groundtruth/"
CALIBRATION_FILE="calibration_sidd_baseline.pt"
TEMP_DIR="temp/sidd"

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
    local current_lambda_f=$8
    local current_rho=$9
    local current_num_bands=${10}
    local current_lambda_gate=${11}

    local INPUT_DIR="datasets/denoise-c/${corruption}/${SEVERITY}/denoise/val/input/"
    local LAYER_STR="All_Gate_Module_1_2"
    local DST_DIR="models_inference/sidd_vpt_results/${corruption}_layer${LAYER_STR}_len${current_p_len}_s${steps}_lrP${lr_prompt}_lrB${lr_backbone}_lf${current_lambda_f}_r${current_rho}_nb${current_num_bands}"

    echo "Running Inference: ${corruption} | Len: ${current_p_len} | lambda_f: ${current_lambda_f} | rho: ${current_rho} | Output: ${DST_DIR}"

    local PROMPT_ARG=""
    if [ "${SHOT_MODE}" = "one-shot" ]; then
        local CORRUPT_PROMPT="datasets/denoise-c/${corruption}/${SEVERITY}/denoise/train/input/${PROMPT}.png"
        if [ -f "${CORRUPT_PROMPT}" ]; then
            PROMPT_ARG="--prompt_rgb_path ${CORRUPT_PROMPT}"
        else
            echo "[Warning] 损坏提示图不存在: ${CORRUPT_PROMPT}，回退到干净提示图"
        fi
    fi

    python3 eval/sidd/painter_inference_new.py \
      --mode inference \
      ${FC_VICT_FLAG} \
      ${PROMPT_ARG} \
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
      --num_bands ${current_num_bands}

    if [ $? -ne 0 ]; then
        echo "[Error] Inference failed for ${corruption}."
        return
    fi

    # 调用评估脚本
    OUTPUT=$(python3 eval/sidd/eval_sidd_test.py \
      --pred_path "${DST_DIR}" \
      --gt_path "${GT_DATA_DIR}")

    STATS=$(echo "$OUTPUT" | awk 'NF>0' | tail -n 1)
    PSNR=$(echo "$STATS" | awk -F',' '{print $1}')
    SSIM=$(echo "$STATS" | awk -F',' '{print $2}')

    if [ -z "$PSNR" ] || [ -z "$SSIM" ]; then
        PSNR="N/A"
        SSIM="N/A"
        echo "[Warning] 提取指标失败，请检查评估脚本输出。"
    fi

    WIN_STATS="N/A"
    if [ -f "${DST_DIR}/winning_groups.txt" ]; then
        WIN_STATS=$(cat "${DST_DIR}/winning_groups.txt")
    fi

    echo "${corruption},${steps},${lr_prompt},${lr_backbone},${LAYER_STR},${current_p_len},${current_lambda_f},${current_rho},${current_num_bands},${current_lambda_gate},${WIN_STATS},${PSNR},${SSIM}" >> ${RESULT_LOG}

    echo "[Done] Result saved to ${RESULT_LOG} (PSNR: ${PSNR}, SSIM: ${SSIM})"
    # echo "[Cleanup] Removing temporary inference directory: ${DST_DIR}"
    # rm -rf "${DST_DIR}"
}

# =========================================================
# 主流程
# =========================================================
if [ ! -f "${RESULT_LOG}" ]; then
    echo "corruption,steps,lr_prompt,lr_backbone,layers,prompt_len,lambda_f,rho,num_bands,lambda_gate,winning_groups,psnr,ssim" > ${RESULT_LOG}
fi

ensure_global_anchor() {
    local current_rho=$1
    local current_num_bands=$2
    local rho_safe=$(echo "${current_rho}" | tr '.' '_')
    local nb_safe=$(echo "${current_num_bands}" | tr '.' '_')
    CURRENT_GLOBAL_ANCHOR="${TEMP_DIR}/anchor_stats_sidd_global_clean_r${rho_safe}_nb${nb_safe}.pt"

    if [ ! -f "${CURRENT_GLOBAL_ANCHOR}" ]; then
        echo ">>> [Stage 0] Generating Global Anchor (rho=${current_rho}, num_bands=${current_num_bands}) from ${CLEAN_DATA_DIR}..."
        find "${CLEAN_DATA_DIR}" -type f \( -name "*.jpg" -o -name "*.png" \) > "${TEMP_DIR}/all_clean_imgs.txt"
        TOTAL_CLEAN_IMAGES=$(wc -l < "${TEMP_DIR}/all_clean_imgs.txt")
        CLEAN_SAMPLE_COUNT=$((TOTAL_CLEAN_IMAGES / 10))
        [ $CLEAN_SAMPLE_COUNT -eq 0 ] && CLEAN_SAMPLE_COUNT=1
        shuf -n ${CLEAN_SAMPLE_COUNT} "${TEMP_DIR}/all_clean_imgs.txt" > "${TEMP_DIR}/clean_sample_list.txt"

        python3 eval/sidd/painter_inference_new.py \
          --mode anchor \
          ${FC_VICT_FLAG} \
          --rho ${current_rho} \
          --sample_list "${TEMP_DIR}/clean_sample_list.txt" \
          --anchor_stats_path ${CURRENT_GLOBAL_ANCHOR} \
          --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
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
for current_rho in "${RHO_LIST[@]}"; do
  rho_safe=$(echo "${current_rho}" | tr '.' '_')

  for current_num_bands in "${NUM_BANDS_LIST[@]}"; do
    nb_safe=$(echo "${current_num_bands}" | tr '.' '_')
    ensure_global_anchor "${current_rho}" "${current_num_bands}"

    for corruption in "${CORRUPTIONS[@]}"; do
        echo "========================================================="
        echo ">>> [Dispatch] Task started for: ${corruption} | rho: ${current_rho} | num_bands: ${current_num_bands}"
        echo "========================================================="

        INPUT_DIR="datasets/denoise-c/${corruption}/${SEVERITY}/denoise/val/input/"
        CORRUPT_SAMPLE_LIST="${TEMP_DIR}/sample_list_${corruption}.txt"
        
        echo ">>> [Stage 2 & 3] Evaluating 10% samples to vote for Winner Group..."
        find "${INPUT_DIR}" -type f \( -name "*.jpg" -o -name "*.png" \) > "${TEMP_DIR}/all_imgs_${corruption}.txt"
        TOTAL_IMAGES=$(wc -l < "${TEMP_DIR}/all_imgs_${corruption}.txt")

        SAMPLE_COUNT=$((TOTAL_IMAGES / 10))
        [ $SAMPLE_COUNT -eq 0 ] && SAMPLE_COUNT=1
        shuf -n ${SAMPLE_COUNT} "${TEMP_DIR}/all_imgs_${corruption}.txt" > ${CORRUPT_SAMPLE_LIST}

        for current_lambda_f in "${LAMBDA_F_LIST[@]}"; do
            lf_safe=$(echo "${current_lambda_f}" | tr '.' '_')

            for current_lambda_gate in "${LAMBDA_GATE_LIST[@]}"; do
                lg_safe=$(echo "${current_lambda_gate}" | tr '.' '_')
                GLOBAL_WINNER_FILE="${TEMP_DIR}/global_winner_${corruption}_r${rho_safe}_lf${lf_safe}_nb${nb_safe}_lg${lg_safe}.txt"

                echo ">>> [Stage 2 & 3] Voting Winner Group | lambda_f: ${current_lambda_f} | rho: ${current_rho} | num_bands: ${current_num_bands} | lambda_gate: ${current_lambda_gate}"

                DRIFT_PROMPT_ARG=""
                if [ "${SHOT_MODE}" = "one-shot" ]; then
                    CORRUPT_PROMPT_DRIFT="datasets/denoise-c/${corruption}/${SEVERITY}/denoise/train/input/${PROMPT}.png"
                    if [ -f "${CORRUPT_PROMPT_DRIFT}" ]; then
                        DRIFT_PROMPT_ARG="--prompt_rgb_path ${CORRUPT_PROMPT_DRIFT}"
                    else
                        echo "[Warning] 损坏提示图不存在: ${CORRUPT_PROMPT_DRIFT}，回退到干净提示图"
                    fi
                fi

                python3 eval/sidd/painter_inference_new.py \
                  --mode eval_drift \
                  ${FC_VICT_FLAG} \
                  ${DRIFT_PROMPT_ARG} \
                  --lambda_f ${current_lambda_f} \
                  --rho ${current_rho} \
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
                  --num_bands ${current_num_bands} \
                  --lambda_gate ${current_lambda_gate}

                echo ">>> [Stage 4] Starting Inference Loop for Voted Winner..."
                for steps in "${STEPS_LIST[@]}"; do
                  for lr_p in "${LR_PROMPT_LIST[@]}"; do
                    for lr_b in "${LR_BACKBONE[@]}"; do
                      for p_len in "${PROMPT_LEN_LIST[@]}"; do
                         do_inference "${corruption}" "${steps}" "${lr_p}" "${lr_b}" "${TARGET_LAYERS}" "${p_len}" "${GLOBAL_WINNER_FILE}" "${current_lambda_f}" "${current_rho}" "${current_num_bands}" "${current_lambda_gate}"
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

# echo "[Cleanup] Removing temporary directory: ${TEMP_DIR}"
# rm -rf "${TEMP_DIR}"

echo "All evaluations finished! Results saved to ${RESULT_LOG}."