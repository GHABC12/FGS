#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

RUN_ROOT="${RUN_ROOT:-eval/coco_panoptic/auto_tune_15x2_50_20260719}"
TUNER="${SCRIPT_DIR}/auto_tune_coco_15x2_50.sh"
MAX_WORKERS="${MAX_WORKERS:-1}"
TUNE_SAMPLE_COUNT="${TUNE_SAMPLE_COUNT:-50}"
START_SPEC="${START_SPEC:-shot_noise:zero_shot}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export RUN_ROOT
worker_pids=()

on_master_exit() {
    local status=$?
    if [ -f "${RUN_ROOT}/master.status" ] && grep -q '^RUNNING' "${RUN_ROOT}/master.status"; then
        printf 'INTERRUPTED exit_code=%s finished=%s\n' \
            "${status}" "$(date -Iseconds)" > "${RUN_ROOT}/master.status"
    fi
}

on_master_signal() {
    local worker_pid
    for worker_pid in "${worker_pids[@]}"; do
        pkill -TERM -P "${worker_pid}" 2>/dev/null || true
        kill -TERM "${worker_pid}" 2>/dev/null || true
    done
    exit 130
}

trap on_master_exit EXIT
trap on_master_signal INT TERM

if [ "${MAX_WORKERS}" -lt 1 ] || [ "${MAX_WORKERS}" -gt 3 ]; then
    echo "[Error] MAX_WORKERS must be between 1 and 3 for the 80GB A100." >&2
    exit 1
fi
if [ ! -f "${TUNER}" ]; then
    echo "[Error] Missing tuner: ${TUNER}" >&2
    exit 1
fi

CORRUPTIONS=(
    gaussian_noise
    shot_noise
    impulse_noise
    defocus_blur
    glass_blur
    motion_blur
    zoom_blur
    snow
    frost
    fog
    brightness
    contrast
    elastic_transform
    pixelate
    jpeg_compression
)
MODES=(zero_shot one_shot)
ALL_JOBS=()
for corruption in "${CORRUPTIONS[@]}"; do
    for mode in "${MODES[@]}"; do
        ALL_JOBS+=("${corruption}:${mode}")
    done
done
JOBS=()
start_found=0
for spec in "${ALL_JOBS[@]}"; do
    if [ "${spec}" = "${START_SPEC}" ]; then
        start_found=1
    fi
    if [ "${start_found}" -eq 1 ]; then
        JOBS+=("${spec}")
    fi
done
if [ "${start_found}" -ne 1 ]; then
    echo "[Error] START_SPEC is not a known corruption/mode pair: ${START_SPEC}" >&2
    exit 1
fi
if [ "${TUNE_SAMPLE_COUNT}" -ne 50 ]; then
    echo "[Error] This continuation entry is fixed to exactly 50 images." >&2
    exit 1
fi

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/status" "${RUN_ROOT}/results"
exec 9>"${RUN_ROOT}/master.lock"
if ! flock -n 9; then
    echo "[Error] Another 15x2 auto-tune master is already running." >&2
    exit 1
fi

preflight() {
    local corruption
    local prompt
    for corruption in "${CORRUPTIONS[@]}"; do
        if [ "$(find "datasets/coco-c/${corruption}/5/val2017" -maxdepth 1 -type f | wc -l)" -ne 5000 ]; then
            echo "[Error] ${corruption} does not contain exactly 5000 validation images." >&2
            return 1
        fi
        prompt="datasets/coco/corrupted_prompts/${corruption}/5/000000391460.jpg"
        if [ ! -f "${prompt}" ]; then
            echo "[Error] Missing one-shot prompt: ${prompt}" >&2
            return 1
        fi
    done
    for required in \
        "datasets/coco/train2017/000000391460.jpg" \
        "datasets/coco/pano_sem_seg/panoptic_segm_train2017_with_color/000000391460.png" \
        "data/mmdet_custom/datasets/coco/pano_ca_inst/train_org/000000391460_image_train_org.png" \
        "data/mmdet_custom/datasets/coco/pano_ca_inst/train_org/000000391460_label_train_org.png" \
        "models/painter_vit_large/painter_vit_large.pth" \
        "${TUNER}"; do
        if [ ! -f "${required}" ]; then
            echo "[Error] Missing required file: ${required}" >&2
            return 1
        fi
    done
    if ! command -v setsid >/dev/null 2>&1; then
        echo "[Error] setsid is required for safe process-group cleanup." >&2
        return 1
    fi
}

run_worker() {
    local worker_id=$1
    local index
    local spec
    local corruption
    local mode
    local log
    local status_file
    local current_job_pgid=""

    on_worker_signal() {
        if [ -n "${current_job_pgid}" ]; then
            kill -TERM -- "-${current_job_pgid}" 2>/dev/null || true
        fi
        exit 130
    }
    trap on_worker_signal INT TERM

    for ((index=worker_id; index<${#JOBS[@]}; index+=MAX_WORKERS)); do
        spec="${JOBS[$index]}"
        corruption="${spec%%:*}"
        mode="${spec##*:}"
        log="${RUN_ROOT}/logs/tune_coco_${corruption}_${mode}.log"
        status_file="${RUN_ROOT}/status/${corruption}_${mode}.status"

        printf 'RUNNING worker=%s pid=%s started=%s\n' \
            "${worker_id}" "${BASHPID}" "$(date -Iseconds)" > "${status_file}"
        echo "[$(date -Iseconds)] Worker ${worker_id}: start ${corruption} ${mode}" | tee -a "${RUN_ROOT}/logs/master.log"

        setsid env \
            PROMPT_MODE="${mode}" \
            INFER_SAMPLE_COUNT="${TUNE_SAMPLE_COUNT}" \
            DRIFT_SAMPLE_COUNT="${TUNE_SAMPLE_COUNT}" \
            ALLOW_NON_100=0 \
            RUN_ROOT="${RUN_ROOT}" \
            CKPT_SHA256="${CKPT_SHA256}" \
            bash "${TUNER}" "${corruption}" >> "${log}" 2>&1 &
        current_job_pgid=$!
        if wait "${current_job_pgid}"; then
            current_job_pgid=""
            printf 'COMPLETED worker=%s pid=%s finished=%s\n' \
                "${worker_id}" "${BASHPID}" "$(date -Iseconds)" > "${status_file}"
            echo "[$(date -Iseconds)] Worker ${worker_id}: completed ${corruption} ${mode}" | tee -a "${RUN_ROOT}/logs/master.log"
        else
            current_job_pgid=""
            printf 'FAILED worker=%s pid=%s finished=%s\n' \
                "${worker_id}" "${BASHPID}" "$(date -Iseconds)" > "${status_file}"
            echo "[$(date -Iseconds)] Worker ${worker_id}: failed ${corruption} ${mode}; continuing remaining jobs" | tee -a "${RUN_ROOT}/logs/master.log"
        fi
    done
}

preflight
CKPT_SHA256=$(sha256sum "models/painter_vit_large/painter_vit_large.pth" | awk '{print $1}')
export CKPT_SHA256
printf 'RUNNING pid=%s workers=%s jobs=%s started=%s\n' \
    "$$" "${MAX_WORKERS}" "${#JOBS[@]}" "$(date -Iseconds)" > "${RUN_ROOT}/master.status"
echo "[$(date -Iseconds)] Starting ${#JOBS[@]} jobs with ${MAX_WORKERS} workers." | tee -a "${RUN_ROOT}/logs/master.log"

for ((worker_id=0; worker_id<MAX_WORKERS; worker_id++)); do
    run_worker "${worker_id}" &
    worker_pids+=("$!")
done

master_failed=0
for worker_pid in "${worker_pids[@]}"; do
    if ! wait "${worker_pid}"; then
        master_failed=1
    fi
done

failed_jobs=$(grep -l '^FAILED' "${RUN_ROOT}"/status/*.status 2>/dev/null | wc -l || true)
completed_jobs=$(grep -l '^COMPLETED' "${RUN_ROOT}"/status/*.status 2>/dev/null | wc -l || true)
if [ "${master_failed}" -ne 0 ] || [ "${failed_jobs}" -ne 0 ] || \
   [ "${completed_jobs}" -ne "${#JOBS[@]}" ]; then
    printf 'FINISHED_WITH_FAILURES completed=%s failed=%s finished=%s\n' \
        "${completed_jobs}" "${failed_jobs}" "$(date -Iseconds)" > "${RUN_ROOT}/master.status"
    exit 1
fi

printf 'COMPLETED completed=%s failed=0 finished=%s\n' \
    "${completed_jobs}" "$(date -Iseconds)" > "${RUN_ROOT}/master.status"
