#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

OLD_ROOT="${OLD_ROOT:-eval/coco_panoptic/auto_tune_15x2_100_20260717}"
NEW_ROOT="${NEW_ROOT:-eval/coco_panoptic/auto_tune_15x2_50_20260719}"
OLD_MASTER_PID="${OLD_MASTER_PID:-2589164}"
NEW_MASTER="${SCRIPT_DIR}/auto_tune_coco_15x2_50_master.sh"
TARGET_STATUS="${OLD_ROOT}/status/gaussian_noise_one_shot.status"
TARGET_BEST="${OLD_ROOT}/results/best_gaussian_noise_one_shot.txt"
POLL_SECONDS="${POLL_SECONDS:-30}"

mkdir -p "${NEW_ROOT}/logs"
exec 9>"${NEW_ROOT}/handoff.lock"
if ! flock -n 9; then
    echo "[Error] A 100-to-50 handoff controller is already active." >&2
    exit 1
fi

log() {
    echo "[$(date -Iseconds)] $*" | tee -a "${NEW_ROOT}/logs/handoff.log"
}

printf 'WAITING target=gaussian_noise:one_shot old_pid=%s started=%s\n' \
    "${OLD_MASTER_PID}" "$(date -Iseconds)" > "${NEW_ROOT}/handoff.status"
log "Waiting for the current 100-image gaussian_noise one_shot job to complete."

while true; do
    if [ -f "${TARGET_STATUS}" ] && grep -q '^COMPLETED' "${TARGET_STATUS}" && \
       [ -f "${TARGET_BEST}" ] && grep -q '^sample_count=100$' "${TARGET_BEST}"; then
        break
    fi
    if [ -f "${TARGET_STATUS}" ] && grep -q '^FAILED' "${TARGET_STATUS}"; then
        printf 'FAILED reason=target_job_failed finished=%s\n' "$(date -Iseconds)" > "${NEW_ROOT}/handoff.status"
        log "Target job failed; refusing to switch protocols."
        exit 1
    fi
    if ! kill -0 "${OLD_MASTER_PID}" 2>/dev/null; then
        printf 'FAILED reason=old_master_stopped_before_target finished=%s\n' "$(date -Iseconds)" > "${NEW_ROOT}/handoff.status"
        log "Old master stopped before the target job completed."
        exit 1
    fi
    sleep "${POLL_SECONDS}"
done

if ! kill -0 "${OLD_MASTER_PID}" 2>/dev/null || \
   ! tr '\0' ' ' < "/proc/${OLD_MASTER_PID}/cmdline" | grep -q 'auto_tune_coco_15x2_100_master.sh'; then
    printf 'FAILED reason=old_master_pid_validation_failed finished=%s\n' "$(date -Iseconds)" > "${NEW_ROOT}/handoff.status"
    log "Old master PID validation failed; refusing to signal an unrelated process."
    exit 1
fi

printf 'STOPPING_OLD old_pid=%s started=%s\n' "${OLD_MASTER_PID}" "$(date -Iseconds)" > "${NEW_ROOT}/handoff.status"
log "Target completed. Stopping the old 100-image master and any partial next job."
kill -TERM "${OLD_MASTER_PID}" 2>/dev/null || true

for _ in $(seq 1 60); do
    if ! kill -0 "${OLD_MASTER_PID}" 2>/dev/null && \
       ! pgrep -f 'auto_tune_coco_15x2_100.sh' >/dev/null 2>&1 && \
       ! pgrep -f 'auto_tune_15x2_100_20260717' >/dev/null 2>&1; then
        break
    fi
    sleep 2
done

if kill -0 "${OLD_MASTER_PID}" 2>/dev/null || \
   pgrep -f 'auto_tune_coco_15x2_100.sh' >/dev/null 2>&1 || \
   pgrep -f 'auto_tune_15x2_100_20260717' >/dev/null 2>&1; then
    log "Graceful stop timed out; sending a final TERM to the exact old-task processes."
    pkill -TERM -f 'auto_tune_coco_15x2_100.sh' 2>/dev/null || true
    pkill -TERM -f 'auto_tune_15x2_100_20260717' 2>/dev/null || true
    kill -TERM "${OLD_MASTER_PID}" 2>/dev/null || true
    sleep 10
fi

if kill -0 "${OLD_MASTER_PID}" 2>/dev/null || \
   pgrep -f 'auto_tune_coco_15x2_100.sh' >/dev/null 2>&1 || \
   pgrep -f 'auto_tune_15x2_100_20260717' >/dev/null 2>&1; then
    printf 'FAILED reason=old_processes_remain finished=%s\n' "$(date -Iseconds)" > "${NEW_ROOT}/handoff.status"
    log "Old-task processes remain; refusing to start a mixed GPU workload."
    exit 1
fi

printf 'STARTING_NEW start_spec=shot_noise:zero_shot sample_count=50 started=%s\n' \
    "$(date -Iseconds)" > "${NEW_ROOT}/handoff.status"
log "Starting the remaining 28 jobs at a fixed 50 images."
nohup env \
    RUN_ROOT="${NEW_ROOT}" \
    MAX_WORKERS=1 \
    TUNE_SAMPLE_COUNT=50 \
    START_SPEC=shot_noise:zero_shot \
    bash "${NEW_MASTER}" > "${NEW_ROOT}/master.nohup.log" 2>&1 &
new_pid=$!
echo "${new_pid}" > "${NEW_ROOT}/master.pid"
sleep 5

if ! kill -0 "${new_pid}" 2>/dev/null; then
    printf 'FAILED reason=new_master_did_not_start finished=%s\n' "$(date -Iseconds)" > "${NEW_ROOT}/handoff.status"
    log "The 50-image master did not remain running."
    exit 1
fi

printf 'COMPLETED new_pid=%s start_spec=shot_noise:zero_shot sample_count=50 finished=%s\n' \
    "${new_pid}" "$(date -Iseconds)" > "${NEW_ROOT}/handoff.status"
log "Handoff complete; the 50-image master is detached and running."
