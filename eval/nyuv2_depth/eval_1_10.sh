#!/bin/bash
set -e

# =========================================================
# 环境配置
# =========================================================
export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL=3

# 允许外部覆盖 Python 解释器；默认使用当前 shell 环境中的 python
PYTHON_BIN="${PYTHON_BIN:-python}"
# =========================================================
# 配置区
# =========================================================
CKPT_FILE="painter_vit_large/painter_vit_large.pth"
PROMPT="study_room_0005b/rgb_00094"
MODEL="painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1"
CKPT_PATH="models/${CKPT_FILE}"

# shot_noise / one-shot 调参结论（1/10 子集 + seed=42）：较优组合
#   num_bands=8, lambda_gate=0.3, rho=0.15, lambda_f=0.01, steps=12, lr=1e-5, prompt_len=150
# GPU 显存紧张时可取消下行注释或在外部 export
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

VPT_TYPE="custom"
PROMPT_LEN_LIST=(150)
TARGET_LAYERS="0 1 2 10 11 12 21 22 23"
SEVERITY=5
RESULT_LOG="results_nyu_1_10.csv"

# 仅 Stage 4 推理子集：固定种子，每次约十分之一测试图（可被环境变量覆盖）
INFERENCE_SAMPLE_SEED="${INFERENCE_SAMPLE_SEED:-42}"

# =========================================================
# 提示图片模式
# zero_shot: 提示图片使用 clean 版本 (对应损坏类型无关的通用提示)
# one_shot:  提示图片使用对应的损坏版本 (与测试图片同类型的损坏提示)
# =========================================================
PROMPT_MODE_LIST=("one_shot")

# =========================================================
# FC-VICT 核心超参数配置
# =========================================================
# 开启频域校准 (策略一)，关闭动态回退机制
FC_VICT_FLAG="--use_fc_vict"

# 适配阶段：固定学习率（无 LR scheduler / cosine / step decay）；AdamW weight_decay 默认 0（非 LR 衰减，可与 PyTorch 默认 0.01 区分）
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"

# 频域与适配（与 shot_noise one-shot 调参结论一致）
LAMBDA_F_LIST=(0.01)
RHO_LIST=(0.15)
NUM_BANDS_LIST=(8)
LAMBDA_GATE_LIST=(0.3)
STEPS_LIST=(12)
LR_PROMPT_LIST=(1e-5)
LR_BACKBONE=(0)

CORRUPTIONS=(
# "gaussian_noise" 
"shot_noise" 
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

    local ALL_SORTED="${TEMP_DIR}/all_infer_sorted_${corruption}.txt"
    local INF_LIST="${TEMP_DIR}/inference_sample_${corruption}.txt"
    find "${INPUT_DIR}" -type f \( -name "*.jpg" -o -name "*.png" \) | sort > "${ALL_SORTED}"
    "${PYTHON_BIN}" -c "
import random
random.seed(${INFERENCE_SAMPLE_SEED})
lines = [l.strip() for l in open('${ALL_SORTED}', encoding='utf-8') if l.strip()]
n = max(1, len(lines) // 10)
random.shuffle(lines)
open('${INF_LIST}', 'w', encoding='utf-8').write('\n'.join(lines[:n]))
"
    echo "[Inference 1/10] seed=${INFERENCE_SAMPLE_SEED} | wrote $(wc -l < "${INF_LIST}") paths -> ${INF_LIST}"

    "${PYTHON_BIN}" eval/nyuv2_depth/painter_inference_1_10.py \
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
      --inference_sample_list "${INF_LIST}" \
      --steps ${steps} \
      --lr ${lr_prompt} \
      --lr_backbone ${lr_backbone} \
      --weight_decay ${WEIGHT_DECAY} \
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
        "${PYTHON_BIN}" eval/nyuv2_depth/painter_inference_1_10.py \
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
        TOTAL_IMAGES=$(wc -l < "${TEMP_DIR}/all_imgs_${corruption}.txt" | tr -d ' ')

        if [ "${TOTAL_IMAGES}" -eq 0 ]; then
            echo ">>> [Skip] No jpg/png under corrupted test dir (skip eval_drift & inference for this corruption):"
            echo "    ${INPUT_DIR}"
            continue
        fi

        SAMPLE_COUNT=$((TOTAL_IMAGES / 10))
        [ "${SAMPLE_COUNT}" -eq 0 ] && SAMPLE_COUNT=1
        shuf -n ${SAMPLE_COUNT} "${TEMP_DIR}/all_imgs_${corruption}.txt" > "${CORRUPT_SAMPLE_LIST}"

        if [ ! -s "${CORRUPT_SAMPLE_LIST}" ]; then
            echo ">>> [Error] Sample list is empty after shuf: ${CORRUPT_SAMPLE_LIST}"
            echo "    (TOTAL_IMAGES=${TOTAL_IMAGES}, INPUT_DIR=${INPUT_DIR})"
            exit 1
        fi

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

            "${PYTHON_BIN}" eval/nyuv2_depth/painter_inference_1_10.py \
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