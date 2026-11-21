import argparse
import csv
import json
from pathlib import Path

import numpy as np

from data.dataset import FusionDataset
from scripts.evaluate_high_frequency_compensation import compensate
from utils.metrics import compute_fusion_metrics


METRICS = ('entropy', 'sd', 'sf', 'vif', 'qabf')


def minmax(image):
    return (image - image.min()) / max(float(image.max() - image.min()), 1e-6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--float_dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--size', type=int, default=256)
    args = parser.parse_args()

    dataset = FusionDataset(args.data_root, split='test', size=args.size)
    files = {path.stem: path for path in Path(args.float_dir).glob('*.npy')}
    missing = [name for name in dataset.names if name not in files]
    if missing:
        raise FileNotFoundError(f'Missing float images: {missing[:10]}')

    rows = []
    for sample in dataset:
        name = Path(sample['name']).stem
        fused = np.load(files[name], allow_pickle=False).astype(np.float32)
        if fused.ndim == 3 and fused.shape[0] == 1:
            fused = fused[0]
        if fused.shape != (args.size, args.size):
            raise ValueError(f'Expected {(args.size, args.size)} for {name}, got {fused.shape}')
        fused = np.clip(fused, 0, 1)
        ir = sample['ir'][0].numpy()
        vi = sample['vi'][0].numpy()
        compensated = compensate(fused, ir, vi, 'high_add', args.beta)
        compensated = minmax(np.clip(compensated, 0, 1))
        baseline = minmax(fused)
        baseline_values = compute_fusion_metrics(baseline, ir, vi)
        compensated_values = compute_fusion_metrics(compensated, ir, vi)
        rows.append({
            'name': sample['name'],
            **{f'baseline_{key}': baseline_values[key] for key in METRICS},
            **{f'compensated_{key}': compensated_values[key] for key in METRICS},
        })
    if not rows:
        raise ValueError('No float images evaluated')

    summary = {}
    for variant in ('baseline', 'compensated'):
        summary[variant] = {
            key: float(np.mean([row[f'{variant}_{key}'] for row in rows])) for key in METRICS
        }
    summary['delta'] = {
        key: summary['compensated'][key] - summary['baseline'][key] for key in METRICS
    }
    result = {
        'protocol': 'raw float32 -> clip -> additive source Haar compensation -> clip -> uniform per-image min-max',
        'beta': args.beta,
        'sample_count': len(rows),
        'summary': summary,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    with open(output.with_suffix('.csv'), 'w', encoding='utf-8-sig', newline='') as file:
        fieldnames = ('name', *(f'{variant}_{key}' for variant in ('baseline', 'compensated') for key in METRICS))
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f'Saved: {output}, {output.with_suffix(".csv")}')


if __name__ == '__main__':
    main()
