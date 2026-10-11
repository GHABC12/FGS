#!/bin/bash
set -e

# Lightweight frozen-Painter task/context experiment. Existing FC-VICT
# behavior below is intentionally unchanged when this subcommand is absent.
if [ "${1:-}" = "task_context" ]; then
  shift
  PYTHON_BIN="${PYTHON_BIN:-/usr/local/iCompute/bin/python}"
  exec "${PYTHON_BIN}" eval/ade20k_semantic/task_context_decoupling_experiment.py "$@"
fi

# =========================================================
# 环境配置
# =========================================================
export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL=3

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
RESULT_LOG="results_ade20k_new.csv"
NUM_IMAGES=500
SEED=42

# Prompt 模式: zero-shot=干净提示图; one-shot=使用与当前 corruption 对应的损坏提示图
PROMPT_MODE="one-shot"  # overridden per-corruption below

# =========================================================
# FC-VICT 核心超参数配置
# =========================================================
FC_VICT_FLAG="--use_fc_vict"
# 频域正则化 Loss 权重 (针对分割任务，PDF建议 0 ~ 0.005)
LAMBDA_F_LIST=(0.005)
RHO_LIST=(0.0625 0.01)

# 径向频带划分数
NUM_BANDS_LIST=(5 2)
LAMBDA_GATE_LIST=(0.5)

STEPS_LIST=(10)
LR_PROMPT_LIST=(1e-5)
LR_BACKBONE=(0)

CORRUPTIONS=(
"fog"
"frost"
)

# 全局文件配置
CLEAN_DATA_DIR="datasets/ade20k/images/validation/"
GLOBAL_ANCHOR_PREFIX="anchor_stats_global_clean_ade20k"
CALIBRATION_FILE="calibration_baseline_ade20k.pt"
TEMP_DIR="temp/ade20k_semantic"

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

    if ! "${PYTHON_BIN}" eval/ade20k_semantic/painter_inference_new.py \
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
    # echo "[Cleanup] Removing temporary inference directory: ${DST_DIR}"
    # rm -rf "${DST_DIR}"
}

# =========================================================
# 主流程
# =========================================================
if [ ! -f "${RESULT_LOG}" ]; then
    echo "corruption,steps,lr_prompt,lr_backbone,layers,prompt_len,num_bands,lambda_gate,lambda_f,rho,winning_groups,mIoU" > ${RESULT_LOG}
fi

find "${CLEAN_DATA_DIR}" -type f \( -name "*.jpg" -o -name "*.png" \) > "${TEMP_DIR}/all_clean_imgs_ade20k.txt"
"${PYTHON_BIN}" -c "
import random, sys
random.seed(${SEED})
lines = open(sys.argv[1]).readlines()
random.shuffle(lines)
for l in lines[:int(sys.argv[2])]: print(l, end='')
" "${TEMP_DIR}/all_clean_imgs_ade20k.txt" 65 > "${TEMP_DIR}/clean_sample_list_ade20k.txt"

for num_bands in "${NUM_BANDS_LIST[@]}"; do
  for current_rho in "${RHO_LIST[@]}"; do
    rho_safe=$(echo "${current_rho}" | tr '.' '_')
    GLOBAL_ANCHOR_CUR="${TEMP_DIR}/${GLOBAL_ANCHOR_PREFIX}_nb${num_bands}_r${rho_safe}.pt"
    
    if [ ! -f "${GLOBAL_ANCHOR_CUR}" ]; then
        echo ">>> [Stage 0] Generating Global Clean Anchor Stats (num_bands=${num_bands}, rho=${current_rho}) from ${CLEAN_DATA_DIR}..."
        "${PYTHON_BIN}" eval/ade20k_semantic/painter_inference_new.py \
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

        # Per-corruption optimal parameter overrides (from CSV tuning results)
        case $corruption in
          fog)
            if [ "$num_bands" != "5" ] || [ "$current_rho" != "0.0625" ]; then
              echo "Skipping fog (needs num_bands=5, rho=0.0625)"
              continue
            fi
            PROMPT_MODE="zero-shot"
            local_steps=10
            local_lr=1e-5
            ;;
          frost)
            if [ "$num_bands" != "2" ] || [ "$current_rho" != "0.01" ]; then
              echo "Skipping frost (needs num_bands=2, rho=0.01)"
              continue
            fi
            PROMPT_MODE="one-shot"
            local_steps=10
            local_lr=5e-5
            ;;
        esac

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

        for lambda_gate in "${LAMBDA_GATE_LIST[@]}"; do
          for current_lambda_f in "${LAMBDA_F_LIST[@]}"; do
            lf_safe=$(echo "${current_lambda_f}" | tr '.' '_')
            GLOBAL_WINNER_FILE="${GLOBAL_WINNER_PREFIX}_lg${lambda_gate}_lf${lf_safe}.txt"

            "${PYTHON_BIN}" eval/ade20k_semantic/painter_inference_new.py \
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

            echo ">>> [Stage 4] Starting Inference for Voted Winner (per-corruption optimal params)..."
            do_inference "${corruption}" "${local_steps}" "${local_lr}" "0" "${TARGET_LAYERS}" "100" "${GLOBAL_WINNER_FILE}" "${num_bands}" "${lambda_gate}" "${current_lambda_f}" "${current_rho}"
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
