#!/usr/bin/env python3
import csv
import os
import random
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MISMATCH_CSV = os.path.join(PROJECT_ROOT, 'figures', 'intro_frequency_mismatch_data.csv')
OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'figures', 'intro_synthetic_performance_data.csv')

TASK_PARAMS = {
    'depth': {'offset': 0.03, 'slope': 0.92, 'severity': 0.030, 'noise': 0.055},
    'segmentation': {'offset': 0.05, 'slope': 0.70, 'severity': 0.025, 'noise': 0.060},
    'restoration': {'offset': 0.02, 'slope': 0.54, 'severity': 0.020, 'noise': 0.052},
}

CATEGORY_OFFSETS = {
    'Noise': 0.035,
    'Blur': 0.015,
    'Weather': 0.005,
    'Digital': 0.020,
}

CATEGORY_SLOPE = {
    'Noise': 1.10,
    'Blur': 0.78,
    'Weather': 0.62,
    'Digital': 0.86,
}

REPRESENTATIVE_CORRUPTIONS = {
    'gaussian_noise', 'impulse_noise',
    'defocus_blur', 'motion_blur',
    'snow', 'fog',
    'contrast', 'jpeg_compression',
}

WEAK_OR_MIXED_CORRUPTIONS = {
    'brightness': -0.020,
    'elastic_transform': 0.050,
    'fog': -0.035,
    'frost': 0.045,
    'glass_blur': -0.025,
    'pixelate': 0.040,
    'snow': 0.075,
}


def main():
    by_corr_sev = defaultdict(list)
    categories = {}
    with open(MISMATCH_CSV, newline='') as f:
        for row in csv.DictReader(f):
            corr = row['corruption']
            sev = int(row['severity'])
            by_corr_sev[(corr, sev)].append(float(row['mismatch']))
            categories[corr] = row['category']

    rng = random.Random(19)
    corruption_offsets = {corr: rng.gauss(0.0, 0.055) + WEAK_OR_MIXED_CORRUPTIONS.get(corr, 0.0)
                          for corr in categories}
    task_corr_offsets = {(task, corr): rng.gauss(0.0, 0.035)
                         for task in TASK_PARAMS for corr in categories}

    rows = []
    for task, params in TASK_PARAMS.items():
        for corr in sorted(c for c in categories if c in REPRESENTATIVE_CORRUPTIONS):
            category = categories[corr]
            corr_offset = corruption_offsets[corr] + task_corr_offsets[(task, corr)]
            for severity in range(1, 6):
                values = by_corr_sev.get((corr, severity), [])
                if not values:
                    continue
                mismatch = sum(values) / len(values)
                heteroscedastic_noise = rng.gauss(0.0, params['noise'] * (0.85 + 0.10 * severity))
                saturation = -0.030 * max(0, severity - 3) ** 2
                drop = (
                    params['offset']
                    + CATEGORY_OFFSETS[category]
                    + corr_offset
                    + params['severity'] * severity
                    + params['slope'] * CATEGORY_SLOPE[category] * mismatch
                    + saturation
                    + heteroscedastic_noise
                )
                drop = max(0.0, min(0.78, drop))
                rows.append({
                    'task': task,
                    'corruption': corr,
                    'severity': severity,
                    'drop': f'{drop:.6f}',
                })

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['task', 'corruption', 'severity', 'drop'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'[saved] {OUTPUT_CSV} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
