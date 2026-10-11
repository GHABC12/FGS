#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-csv", required=True)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--plot-png", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--expected-images", type=int, required=True)
    parser.add_argument("--expected-steps", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    grouped = defaultdict(dict)
    elapsed_by_image = defaultdict(list)
    with open(args.raw_csv, newline="", encoding="utf-8") as raw_file:
        reader = csv.DictReader(raw_file)
        if reader.fieldnames != ["image", "step", "psnr", "elapsed_s"]:
            raise ValueError(f"Unexpected raw CSV columns: {reader.fieldnames}")
        for row in reader:
            image = row["image"]
            step = int(row["step"])
            psnr = float(row["psnr"])
            elapsed = float(row["elapsed_s"])
            if not math.isfinite(psnr) or not math.isfinite(elapsed):
                raise ValueError(f"Non-finite value at {image}, step {step}")
            if step in grouped[image]:
                raise ValueError(f"Duplicate key: {image}, step {step}")
            grouped[image][step] = psnr
            elapsed_by_image[image].append(elapsed)

    expected_step_set = set(range(1, args.expected_steps + 1))
    if len(grouped) != args.expected_images:
        raise ValueError(
            f"Expected {args.expected_images} images, got {len(grouped)}"
        )
    for image, values in grouped.items():
        if set(values) != expected_step_set:
            raise ValueError(f"Missing or extra steps for {image}")
        times = elapsed_by_image[image]
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError(f"elapsed_s is not strictly increasing for {image}")

    rows = []
    previous_mean = None
    critical = student_t.ppf(0.975, df=args.expected_images - 1)
    for step in range(1, args.expected_steps + 1):
        values = np.array(
            [grouped[image][step] for image in sorted(grouped)], dtype=float
        )
        mean = float(values.mean())
        std = float(values.std(ddof=1))
        margin = float(critical * std / math.sqrt(len(values)))
        delta = None if previous_mean is None else mean - previous_mean
        rows.append({
            "step": step,
            "mean_psnr": mean,
            "std_psnr": std,
            "ci95_low": mean - margin,
            "ci95_high": mean + margin,
            "delta_psnr": delta,
            "successful_images": len(values),
        })
        previous_mean = mean

    Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_csv, "w", newline="", encoding="utf-8") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=[
            "step", "mean_psnr", "std_psnr", "ci95_low", "ci95_high",
            "delta_psnr", "successful_images",
        ])
        writer.writeheader()
        writer.writerows(rows)

    threshold = 0.01
    deltas_11_20 = [
        {
            "step": row["step"],
            "delta_psnr": row["delta_psnr"],
            "pass": abs(row["delta_psnr"]) <= threshold,
        }
        for row in rows if 11 <= row["step"] <= 20
    ]
    step10_sufficient = (
        len(deltas_11_20) == 10
        and all(item["pass"] for item in deltas_11_20)
    )
    first_plateau_start = None
    for start in range(1, args.expected_steps - 9):
        window = [rows[step - 1]["delta_psnr"] for step in range(start + 1, start + 11)]
        if all(delta is not None and abs(delta) <= threshold for delta in window):
            first_plateau_start = start
            break

    reference_step = 10 if args.expected_steps >= 10 else args.expected_steps
    step10_mean = rows[reference_step - 1]["mean_psnr"]
    checkpoints = {}
    for step in (10, 20, 50, 100):
        if step > args.expected_steps:
            continue
        mean = rows[step - 1]["mean_psnr"]
        checkpoints[str(step)] = {
            "mean_psnr": mean,
            "difference_from_step10": mean - step10_mean,
        }

    result = {
        "valid": True,
        "images": args.expected_images,
        "steps": args.expected_steps,
        "raw_rows": args.expected_images * args.expected_steps,
        "criterion": "abs(delta_psnr) <= 0.01 dB for steps 11 through 20",
        "step10_sufficient": step10_sufficient,
        "first_plateau_start": first_plateau_start,
        "deltas_11_20": deltas_11_20,
        "checkpoints": checkpoints,
    }
    with open(args.result_json, "w", encoding="utf-8") as result_file:
        json.dump(result, result_file, indent=2, ensure_ascii=False)

    x = np.array([row["step"] for row in rows])
    means = np.array([row["mean_psnr"] for row in rows])
    lower = np.array([row["ci95_low"] for row in rows])
    upper = np.array([row["ci95_high"] for row in rows])
    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    ax.plot(x, means, color="#1769aa", linewidth=2, label="Mean PSNR")
    ax.fill_between(x, lower, upper, color="#1769aa", alpha=0.18, label="95% CI")
    if args.expected_steps >= 20:
        ax.axvspan(10, 20, color="#f5a623", alpha=0.13, label="Plateau test: 10–20")
    if args.expected_steps >= 10:
        ax.axvline(10, color="#d32f2f", linestyle="--", linewidth=1.5, label="step=10")
        ax.scatter([10], [step10_mean], color="#d32f2f", zorder=5)
    verdict = "PASS" if step10_sufficient else "NOT PROVEN"
    ax.set_title(f"LOL-C Pixelate: PSNR vs TTA Steps — step=10 {verdict}")
    ax.set_xlabel("TTA step")
    ax.set_ylabel("Mean PSNR (dB)")
    ax.set_xlim(1, args.expected_steps)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.plot_png, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
