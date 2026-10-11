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
TEMP_DIR="temp/coco_panoptic"
export TMPDIR="/home/user/shiyunxiao/Painter-main/Painter/${TEMP_DIR}/tmpdir"
mkdir -p "${TMPDIR}"

# =========================================================
# 配置区
# =========================================================
CKPT_FILE="painter_vit_large/painter_vit_large.pth"
PROMPT="000000391460"
MODEL="painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1"
CKPT_PATH="models/${CKPT_FILE}"

VPT_TYPE="custom"
PROMPT_LEN_LIST=(100)
TARGET_LAYERS="0 1 2 10 11 12 21 22 23"
SEVERITY=5
RESULT_LOG="results_coco.csv"
PRED_ROOT_DIR="models_inference/vpt_results_coco"

#5 0.3
NUM_BANDS_LIST=(5 6)       # 径向频带划分数列表
LAMBDA_GATE_LIST=(0.2 0.3 0.4)    # 融合权重列表
#3 3e-5
STEPS_LIST=(3)
LR_PROMPT_LIST=(3e-5)
LR_BACKBONE=(0)
INFER_SAMPLE_COUNT=500
DRIFT_SAMPLE_COUNT=50
RANDOM_SEED=20260426

CORRUPTIONS=(
"gaussian_noise"
# "shot_noise"
# "impulse_noise"
"defocus_blur"
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
"jpeg_compression"
)

# 全局文件配置
CLEAN_DATA_DIR="datasets/coco/val2017/"
GLOBAL_ANCHOR_BASE="anchor_stats_global_clean_coco.pt"
CALIBRATION_FILE="calibration_baseline_coco.pt"

# 创建临时目录
mkdir -p "${TEMP_DIR}"

# 固定随机种子采样，保证实验可复现
sample_file_list() {
    local src_list=$1
    local dst_list=$2
    local sample_count=$3
    local seed=$4
    python - "$src_list" "$dst_list" "$sample_count" "$seed" <<'PY'
import random
import sys

src, dst, k_str, seed_str = sys.argv[1:5]
k = int(k_str)
seed = int(seed_str)

with open(src, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

if not lines:
    selected = []
else:
    rnd = random.Random(seed)
    k = max(1, min(k, len(lines)))
    selected = rnd.sample(lines, k)

with open(dst, "w", encoding="utf-8") as f:
    for item in selected:
        f.write(item + "\n")
PY
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

    local INPUT_DIR="datasets/coco-c/${corruption}/${SEVERITY}/val2017/"
    local INFER_ALL_LIST="${TEMP_DIR}/all_imgs_infer_${corruption}.txt"
    local INFER_SAMPLE_LIST="${TEMP_DIR}/infer_sample_${corruption}.txt"
    local INFER_INPUT_DIR="${TEMP_DIR}/infer_inputs_${corruption}"
    local LAYER_STR="All_Gate_Module_1_2"
    
    # COCO全景分割需要分别生成 SemSeg 和 Inst 的预测
    local DST_DIR_SEM="${PRED_ROOT_DIR}/semseg_${corruption}_layer${LAYER_STR}_len${current_p_len}_s${steps}_lrP${lr_prompt}_lrB${lr_backbone}"
    local DST_DIR_INS="${PRED_ROOT_DIR}/inst_${corruption}_layer${LAYER_STR}_len${current_p_len}_s${steps}_lrP${lr_prompt}_lrB${lr_backbone}"

    find "${INPUT_DIR}" -type f \( -name "*.jpg" -o -name "*.png" \) > "${INFER_ALL_LIST}"
    local TOTAL_INFER_IMAGES
    TOTAL_INFER_IMAGES=$(wc -l < "${INFER_ALL_LIST}")
    local INFER_COUNT=${INFER_SAMPLE_COUNT}
    [ "${TOTAL_INFER_IMAGES}" -lt "${INFER_COUNT}" ] && INFER_COUNT=${TOTAL_INFER_IMAGES}
    [ "${INFER_COUNT}" -eq 0 ] && INFER_COUNT=1
    sample_file_list "${INFER_ALL_LIST}" "${INFER_SAMPLE_LIST}" "${INFER_COUNT}" "${RANDOM_SEED}"

    rm -rf "${INFER_INPUT_DIR}"
    mkdir -p "${INFER_INPUT_DIR}"
    while IFS= read -r img_path; do
        ln -sf "$(realpath "${img_path}")" "${INFER_INPUT_DIR}/$(basename "${img_path}")"
    done < "${INFER_SAMPLE_LIST}"

    echo "Running Inference: ${corruption} | Len: ${current_p_len} | Output Dirs: ${DST_DIR_SEM} & ${DST_DIR_INS}"

    # 1. 语义分割推理 (SemSeg)
    python eval/coco_panoptic/painter_inference.py \
      --mode inference --task semseg \
      --global_winner_path ${global_winner_file} \
      --ckpt_path ${CKPT_PATH} \
      --model ${MODEL} \
      --prompt ${PROMPT} \
      --input_dir "${INFER_INPUT_DIR}" \
      --save_dir "${DST_DIR_SEM}" \
      --steps ${steps} \
      --lr ${lr_prompt} \
      --lr_backbone ${lr_backbone} \
      --vpt_type ${VPT_TYPE} \
      --prompt_length ${current_p_len} \
      --vpt_target_layer ${current_layers} \
      --num_bands ${NUM_BANDS}

    # 2. 实例分割推理 (Inst)
    python eval/coco_panoptic/painter_inference.py \
      --mode inference --task inst \
      --global_winner_path ${global_winner_file} \
      --ckpt_path ${CKPT_PATH} \
      --model ${MODEL} \
      --prompt ${PROMPT} \
      --input_dir "${INFER_INPUT_DIR}" \
      --save_dir "${DST_DIR_INS}" \
      --steps ${steps} \
      --lr ${lr_prompt} \
      --lr_backbone ${lr_backbone} \
      --vpt_type ${VPT_TYPE} \
      --prompt_length ${current_p_len} \
      --vpt_target_layer ${current_layers} \
      --num_bands ${NUM_BANDS}

    if [ $? -ne 0 ]; then
        echo "[Error] Inference failed for ${corruption}."
        return
    fi

    # 3. 评估指标：失败时打印完整 stdout/stderr，避免静默退出
    local EVAL_STDOUT_LOG="${TEMP_DIR}/eval_stdout_${corruption}.log"
    local EVAL_STDERR_LOG="${TEMP_DIR}/eval_stderr_${corruption}.log"
    set +e
    python eval/coco_panoptic/eval_with_pngs.py \
      --pred_path_sem "${DST_DIR_SEM}" \
      --pred_path_inst "${DST_DIR_INS}" \
      --gt_path "datasets/coco/annotations/instances_val2017.json" \
      --dist_thr 19 \
      --prompt ${PROMPT} \
      --ckpt_file "${CKPT_FILE##*/}" \
      --input_size 448 \
      > "${EVAL_STDOUT_LOG}" 2> "${EVAL_STDERR_LOG}"
    local EVAL_STATUS=$?
    set -e
    local OUTPUT
    OUTPUT=$(cat "${EVAL_STDOUT_LOG}")
    if [ ${EVAL_STATUS} -ne 0 ]; then
        echo "[Error] Evaluation failed for ${corruption}. Full stdout/stderr below:"
        echo "----- eval stdout -----"
        cat "${EVAL_STDOUT_LOG}"
        echo "----- eval stderr -----"
        cat "${EVAL_STDERR_LOG}"
        return 1
    fi

    PQ=$(echo "$OUTPUT" | awk -F'|' '/^PQ[[:space:]]*\|/ {gsub(/[[:space:]]/, "", $2); print $2; exit}')

    if [ -z "$PQ" ]; then
        PQ="N/A"
        echo "[Warning] 未提取到 PQ 数值，已写入 N/A。"
    fi

    WIN_STATS="N/A"
    if [ -f "${DST_DIR_SEM}/winning_groups.txt" ]; then
        WIN_STATS=$(cat "${DST_DIR_SEM}/winning_groups.txt")
    fi

    # 写入结果
    echo "${corruption},${steps},${lr_prompt},${lr_backbone},${LAYER_STR},${current_p_len},${WIN_STATS},${PQ},${NUM_BANDS},${LAMBDA_GATE}" >> ${RESULT_LOG}

    echo "[Done] Result saved to ${RESULT_LOG} (PQ: ${PQ})"
    echo "[Cleanup] Removing temporary inference directory..."
    rm -rf "${DST_DIR_SEM}" "${DST_DIR_INS}" "${INFER_INPUT_DIR}"
    rm -f "${INFER_ALL_LIST}" "${INFER_SAMPLE_LIST}" "${EVAL_STDOUT_LOG}" "${EVAL_STDERR_LOG}"
}

# =========================================================
# 主流程
# =========================================================
OLD_HEADER="corruption,steps,lr_prompt,lr_backbone,layers,prompt_len,winning_groups,pq"
NEW_HEADER="corruption,steps,lr_prompt,lr_backbone,layers,prompt_len,winning_groups,pq,num_bands,lambda_gate"

if [ -f "${RESULT_LOG}" ]; then
    FIRST_LINE="$(awk 'NR==1{print; exit}' "${RESULT_LOG}")"
    if [ "${FIRST_LINE}" = "${OLD_HEADER}" ]; then
        cp "${RESULT_LOG}" "${RESULT_LOG}.bak_with_no_band_lambda"
        awk -F',' -v OFS=',' -v new_header="${NEW_HEADER}" '
            NR==1 { print new_header; next }
            NF==0 { print; next }
            { print $0, "NA", "NA" }
        ' "${RESULT_LOG}.bak_with_no_band_lambda" > "${RESULT_LOG}"
        echo "[Info] Upgraded ${RESULT_LOG} header; historical rows are backfilled with num_bands/lambda_gate=NA."
    fi
else
    echo "${NEW_HEADER}" > "${RESULT_LOG}"
fi

for NUM_BANDS in "${NUM_BANDS_LIST[@]}"; do
    if [ "${NUM_BANDS}" = "10" ]; then
        GLOBAL_ANCHOR="${GLOBAL_ANCHOR_BASE}"
    else
        GLOBAL_ANCHOR="${GLOBAL_ANCHOR_BASE%.pt}_b${NUM_BANDS}.pt"
    fi

    # [阶段 0] 全局离线计算 Anchor 统计量 (仅一次)
    if [ ! -f "${GLOBAL_ANCHOR}" ]; then
        echo ">>> [Stage 0] Generating Global Clean Anchor Stats from ${CLEAN_DATA_DIR}..."
        find "${CLEAN_DATA_DIR}" -type f \( -name "*.jpg" -o -name "*.png" \) > "${TEMP_DIR}/all_clean_imgs.txt"
        sample_file_list "${TEMP_DIR}/all_clean_imgs.txt" "${TEMP_DIR}/clean_sample_list.txt" "50" "${RANDOM_SEED}"

        python eval/coco_panoptic/painter_inference.py \
          --mode anchor --task semseg \
          --sample_list "${TEMP_DIR}/clean_sample_list.txt" \
          --anchor_stats_path ${GLOBAL_ANCHOR} \
          --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
          --input_dir "dummy" --save_dir "dummy" \
          --vpt_type ${VPT_TYPE} --prompt_length 50 --vpt_target_layer ${TARGET_LAYERS} \
          --num_bands ${NUM_BANDS}
    else
        echo ">>> [Stage 0] Global Clean Anchor Stats already exists: ${GLOBAL_ANCHOR}"
    fi

    for LAMBDA_GATE in "${LAMBDA_GATE_LIST[@]}"; do
        # ==========================================
        # 遍历各损坏数据集进行 Test-Time Adaptation
        # ==========================================
        for corruption in "${CORRUPTIONS[@]}"; do
            echo "========================================================="
            echo ">>> [Dispatch] Task started for: ${corruption}"
            echo "========================================================="

            INPUT_DIR="datasets/coco-c/${corruption}/${SEVERITY}/val2017/"
            CORRUPT_SAMPLE_LIST="${TEMP_DIR}/sample_list_${corruption}.txt"
            GLOBAL_WINNER_FILE="${TEMP_DIR}/global_winner_${corruption}.txt"

            echo ">>> [Stage 2 & 3] Evaluating ${DRIFT_SAMPLE_COUNT} samples to vote for Winner Group..."
            find "${INPUT_DIR}" -type f \( -name "*.jpg" -o -name "*.png" \) > "${TEMP_DIR}/all_imgs_${corruption}.txt"
            TOTAL_IMAGES=$(wc -l < "${TEMP_DIR}/all_imgs_${corruption}.txt")

            SAMPLE_COUNT=${DRIFT_SAMPLE_COUNT}
            [ ${TOTAL_IMAGES} -lt ${SAMPLE_COUNT} ] && SAMPLE_COUNT=${TOTAL_IMAGES}
            [ $SAMPLE_COUNT -eq 0 ] && SAMPLE_COUNT=1
            sample_file_list "${TEMP_DIR}/all_imgs_${corruption}.txt" "${CORRUPT_SAMPLE_LIST}" "${SAMPLE_COUNT}" "${RANDOM_SEED}"

            # 使用 semseg prompt 提取跨层效用/投票
            python eval/coco_panoptic/painter_inference.py \
              --mode eval_drift --task semseg \
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
              --prompt_length 50 \
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
    done
done

# 清理统一的临时目录
echo "[Cleanup] Removing temporary directory: ${TEMP_DIR}"
rm -rf "${TEMP_DIR}"

echo "All evaluations finished! Results saved to ${RESULT_LOG}."