#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

RUN_ROOT="${RUN_ROOT:-eval/coco_panoptic/auto_tune_15x2_50_20260719}"
EXPECTED_TOTAL="${EXPECTED_TOTAL:-2016}"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/iCompute/bin/python}"

echo "=== Master ==="
if [ -f "${RUN_ROOT}/master.status" ]; then
    cat "${RUN_ROOT}/master.status"
else
    echo "NOT_STARTED"
fi

echo "=== Jobs ==="
if compgen -G "${RUN_ROOT}/status/*.status" >/dev/null; then
    grep -H '' "${RUN_ROOT}"/status/*.status | sort
else
    echo "No job status files."
fi

echo "=== Candidate progress ==="
"${PYTHON_BIN}" - "${RUN_ROOT}" "${EXPECTED_TOTAL}" <<'PY'
import csv
import math
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
expected_total = int(sys.argv[2])
result_files = sorted((root / "results").glob("results_coco_*_tuning.csv"))
total_completed = 0
for result_file in result_files:
    completed = set()
    failed_rows = 0
    best_pq = None
    with result_file.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            status = row.get("status")
            if status != "OK":
                failed_rows += 1
                continue
            try:
                pq = float(row.get("pq", ""))
            except Exception:
                continue
            if not math.isfinite(pq):
                continue
            key = tuple(
                row.get(field, "")
                for field in (
                    "protocol_hash", "stage", "round", "param1", "param1_val",
                    "param2", "param2_val", "steps", "lr_prompt", "prompt_len",
                    "num_bands", "lambda_gate", "lambda_f", "rho",
                )
            )
            completed.add(key)
            best_pq = pq if best_pq is None else max(best_pq, pq)
    total_completed += len(completed)
    print(
        f"{result_file.name}: completed={len(completed)}/72, "
        f"failed_rows={failed_rows}, best_pq={best_pq if best_pq is not None else 'N/A'}"
    )
print(
    f"TOTAL: completed={total_completed}/{expected_total} "
    f"({total_completed / expected_total * 100:.2f}%)"
)
retained = 144
print(
    f"OVERALL: retained_gaussian_100={retained}, "
    f"completed_50={total_completed}, combined={retained + total_completed}/2160 "
    f"({(retained + total_completed) / 2160 * 100:.2f}%)"
)
PY

echo "=== GPU ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader

echo "=== Active processes ==="
pgrep -af 'auto_tune_coco_15x2_50|painter_inference_coco_zero_shot_correct_prompt|eval_with_pngs' || true

echo "=== Master log tail ==="
if [ -f "${RUN_ROOT}/logs/master.log" ]; then
    tail -n 20 "${RUN_ROOT}/logs/master.log"
else
    echo "No master log."
fi
