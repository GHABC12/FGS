#!/bin/bash
set -e

# =========================================================
# 环境配置
# =========================================================
export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# [修复1] 允许外部覆盖 Python 解释器；未设置时回退到 python
PYTHON_BIN="${PYTHON_BIN:-python}"

CONDA_LIB="/root/miniconda3/lib"
if [ -d "${CONDA_LIB}" ]; then
  case ":${LD_LIBRARY_PATH}:" in
    *":${CONDA_LIB}:"*) ;;
    *) export LD_LIBRARY_PATH="${CONDA_LIB}:${LD_LIBRARY_PATH}" ;;
  esac
fi
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
RESULT_LOG="results_nyu_new.csv"

# =========================================================
# 提示图片模式
# zero_shot: 提示图片使用 clean 版本 (对应损坏类型无关的通用提示)
# one_shot:  提示图片使用对应的损坏版本 (与测试图片同类型的损坏提示)
# =========================================================
PROMPT_MODE_LIST=("one_shot")

# =========================================================
# FC-VICT 核心超参数配置
# =========================================================
FC_VICT_FLAG=""
# 频域正则化 Loss 权重
#0.01 0.08
LAMBDA_F_LIST=(0)
RHO_LIST=(0)

# 径向频带划分数
NUM_BANDS_LIST=(4)
LAMBDA_GATE_LIST=(0.5)
#10 5e-6
STEPS_LIST=(10)
LR_PROMPT_LIST=(5e-6)
LR_BACKBONE=(0)

CORRUPTIONS=(
# "gaussian_noise"
# "shot_noise"
# "impulse_noise"
# "defocus_blur"
# "glass_blur"
# "motion_blur"
# "zoom_blur"
"snow"
# "frost"
# "fog"
# "brightness"
# "contrast"
# "elastic_transform"
# "pixelate"
# "jpeg_compression"
)

# 全局文件配置
CLEAN_DATA_DIR="datasets/nyu_depth_v2/sync/"
GLOBAL_ANCHOR_PREFIX="anchor_stats_global_clean"
CALIBRATION_FILE="calibration_baseline.pt"
TEMP_DIR="temp/nyuv2_depth"

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
    local current_prompt_mode=${12}

    local INPUT_DIR="datasets/nyu_depth_v2/nyu_depth_v2_corrupted/${corruption}/${SEVERITY}/nyu_depth_v2/official_splits/test/"
    local LAYER_STR="All_Gate_Module_1_2"
    local DST_DIR="models_inference/vpt_results/${corruption}_layer${LAYER_STR}_len${current_p_len}_s${steps}_lrP${lr_prompt}_lrB${lr_backbone}_nb${current_num_bands}_lg${current_lambda_gate}_lf${current_lambda_f}_r${current_rho}"

    local PROMPT_CORRUPTION_ARG=""
    if [ "${current_prompt_mode}" = "one_shot" ]; then
        PROMPT_CORRUPTION_ARG="--prompt_corruption ${corruption}"
    fi

    echo "Running Inference: ${corruption} | Prompt Mode: ${current_prompt_mode} | Len: ${current_p_len} | Output Dir: ${DST_DIR}"

    "${PYTHON_BIN}" eval/nyuv2_depth/painter_inference_new.py \
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
      ${PROMPT_CORRUPTION_ARG}

    if [ $? -ne 0 ]; then
        echo "[Error] Inference failed for ${corruption}."
        return
    fi

    OUTPUT=$("${PYTHON_BIN}" eval/nyuv2_depth/eval_with_pngs_test.py \
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

    echo "${current_prompt_mode},${corruption},${steps},${lr_prompt},${lr_backbone},${LAYER_STR},${current_p_len},${current_num_bands},${current_lambda_gate},${current_lambda_f},${current_rho},${WIN_STATS},${ABS_REL}" >> ${RESULT_LOG}

    echo "[Done] Result saved to ${RESULT_LOG} (abs_rel: ${ABS_REL})"
    echo "[Cleanup] Removing temporary inference directory: ${DST_DIR}"
    rm -rf "${DST_DIR}"
}

# =========================================================
# 主流程
# =========================================================
if [ ! -f "${RESULT_LOG}" ]; then
    echo "prompt_mode,corruption,steps,lr_prompt,lr_backbone,layers,prompt_len,num_bands,lambda_gate,lambda_f,rho,winning_groups,abs_rel" > ${RESULT_LOG}
fi

find "${CLEAN_DATA_DIR}" -type f \( -name "*.jpg" -o -name "*.png" \) | grep -i "rgb" > "${TEMP_DIR}/all_clean_imgs.txt"
shuf -n 65 "${TEMP_DIR}/all_clean_imgs.txt" > "${TEMP_DIR}/clean_sample_list.txt"

for num_bands in "${NUM_BANDS_LIST[@]}"; do
  for current_rho in "${RHO_LIST[@]}"; do
    rho_safe=$(echo "${current_rho}" | tr '.' '_')
    GLOBAL_ANCHOR_CUR="${TEMP_DIR}/${GLOBAL_ANCHOR_PREFIX}_nb${num_bands}_r${rho_safe}.pt"

    if [ ! -f "${GLOBAL_ANCHOR_CUR}" ]; then
        echo ">>> [Stage 0] Generating Global Clean Anchor Stats (num_bands=${num_bands}, rho=${current_rho}) from ${CLEAN_DATA_DIR}..."
        "${PYTHON_BIN}" eval/nyuv2_depth/painter_inference_new.py \
          --mode anchor \
          ${FC_VICT_FLAG} \
          --rho ${current_rho} \
          --sample_list "${TEMP_DIR}/clean_sample_list.txt" \
          --anchor_stats_path ${GLOBAL_ANCHOR_CUR} \
          --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
          --input_dir "dummy" --save_dir "dummy" \
          --vpt_type ${VPT_TYPE} --prompt_length 50 --vpt_target_layer ${TARGET_LAYERS} \
          --num_bands ${num_bands}
    else
        echo ">>> [Stage 0] Global Clean Anchor Stats already exists: ${GLOBAL_ANCHOR_CUR}"
    fi

    # 严格遵循 LOL 结构：先遍历 Corruption
    for corruption in "${CORRUPTIONS[@]}"; do
        echo "========================================================="
        echo ">>> [Dispatch] Task started for: ${corruption}"
        echo "========================================================="

        INPUT_DIR="datasets/nyu_depth_v2/nyu_depth_v2_corrupted/${corruption}/${SEVERITY}/nyu_depth_v2/official_splits/test/"
        CORRUPT_SAMPLE_LIST="${TEMP_DIR}/sample_list_${corruption}_nb${num_bands}_r${rho_safe}.txt"
        GLOBAL_WINNER_PREFIX="${TEMP_DIR}/global_winner_${corruption}_nb${num_bands}_r${rho_safe}"

        echo ">>> [Stage 2 & 3] Evaluating 10% samples to vote for Winner Group..."
        find "${INPUT_DIR}" -type f \( -name "*.jpg" -o -name "*.png" \) > "${TEMP_DIR}/all_imgs_${corruption}.txt"
        TOTAL_IMAGES=$(wc -l < "${TEMP_DIR}/all_imgs_${corruption}.txt")

        SAMPLE_COUNT=$((TOTAL_IMAGES / 10))
        [ $SAMPLE_COUNT -eq 0 ] && SAMPLE_COUNT=1
        shuf -n ${SAMPLE_COUNT} "${TEMP_DIR}/all_imgs_${corruption}.txt" > ${CORRUPT_SAMPLE_LIST}

        for prompt_mode in "${PROMPT_MODE_LIST[@]}"; do
          echo ">>> [Prompt Mode] Current mode: ${prompt_mode}"
        for lambda_gate in "${LAMBDA_GATE_LIST[@]}"; do
          for current_lambda_f in "${LAMBDA_F_LIST[@]}"; do
            lf_safe=$(echo "${current_lambda_f}" | tr '.' '_')
            GLOBAL_WINNER_FILE="${GLOBAL_WINNER_PREFIX}_lg${lambda_gate}_lf${lf_safe}.txt"

            EVAL_DRIFT_PROMPT_ARG=""
            if [ "${prompt_mode}" = "one_shot" ]; then
                EVAL_DRIFT_PROMPT_ARG="--prompt_corruption ${corruption}"
            fi

            "${PYTHON_BIN}" eval/nyuv2_depth/painter_inference_new.py \
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
              ${EVAL_DRIFT_PROMPT_ARG}

            echo ">>> [Stage 4] Starting Inference Loop for Voted Winner..."
            for steps in "${STEPS_LIST[@]}"; do
              for lr_p in "${LR_PROMPT_LIST[@]}"; do
                for lr_b in "${LR_BACKBONE[@]}"; do
                  for p_len in "${PROMPT_LEN_LIST[@]}"; do
                     do_inference "${corruption}" "${steps}" "${lr_p}" "${lr_b}" "${TARGET_LAYERS}" "${p_len}" "${GLOBAL_WINNER_FILE}" "${num_bands}" "${lambda_gate}" "${current_lambda_f}" "${current_rho}" "${prompt_mode}"
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

echo "[Cleanup] Removing temporary directory: ${TEMP_DIR}"
rm -rf "${TEMP_DIR}"

echo "All evaluations finished! Results saved to ${RESULT_LOG}."