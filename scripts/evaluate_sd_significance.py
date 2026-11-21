import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]


def read_sd(path):
    with path.open(encoding='utf-8-sig', newline='') as file:
        rows = list(csv.DictReader(file))
    return {Path(row['name']).stem: float(row['sd']) for row in rows}


def holm_adjust(p_values):
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def main():
    comparisons = []
    for dataset in ('msrs', 'vifb'):
        reference = read_sd(ROOT / f'results/baselines_float/ifddbf/{dataset}/metrics_uniform_minmax.csv')
        for seed, suffix in ((42, ''), (43, '_seed43'), (44, '_seed44')):
            ours = read_sd(ROOT / f'results/research_float/haar_value_control{suffix}/{dataset}/metrics_uniform_minmax.csv')
            names = sorted(set(ours) & set(reference))
            if names != sorted(reference):
                raise ValueError(f'Name mismatch for {dataset}, seed {seed}')
            differences = np.array([ours[name] - reference[name] for name in names])
            nonzero = differences[differences != 0]
            result = wilcoxon(differences, alternative='greater', zero_method='wilcox')
            ranks = np.argsort(np.argsort(np.abs(nonzero))) + 1
            positive = float(ranks[nonzero > 0].sum())
            negative = float(ranks[nonzero < 0].sum())
            comparisons.append({
                'dataset': dataset,
                'seed': seed,
                'pairs': len(names),
                'mean_difference': float(differences.mean()),
                'median_difference': float(np.median(differences)),
                'p_one_sided': float(result.pvalue),
                'rank_biserial': (positive - negative) / (positive + negative),
            })
    adjusted = holm_adjust([item['p_one_sided'] for item in comparisons])
    for item, p_adjusted in zip(comparisons, adjusted):
        item['p_holm'] = float(p_adjusted)
    result = {
        'test': 'Paired one-sided Wilcoxon signed-rank, H1: Ours-VC SD > IFD-DBF SD',
        'effect': 'Rank-biserial correlation on nonzero paired differences',
        'correction': 'Holm across six dataset-seed tests',
        'comparisons': comparisons,
    }
    output = ROOT / 'results/comparison/ours_vc_sd_vs_ifddbf.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
