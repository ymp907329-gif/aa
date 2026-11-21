import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
METRICS = ('entropy', 'sd', 'sf', 'vif', 'qabf')


def holm(tests):
    order = np.argsort([test['p_value'] for test in tests])
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(tests) - rank) * tests[index]['p_value'])
        tests[index]['p_holm'] = float(min(1.0, running))


def main():
    tests = []
    for dataset in ('msrs', 'roadscene', 'tno', 'vifb'):
        path = ROOT / f'results/research_float/haar_adaptive_compensation/{dataset}/three_way_scale130.csv'
        with path.open(encoding='utf-8-sig', newline='') as file:
            rows = list(csv.DictReader(file))
        for reference in ('none', 'fixed'):
            for metric in METRICS:
                differences = np.array([
                    float(row[f'adaptive_{metric}']) - float(row[f'{reference}_{metric}']) for row in rows
                ])
                nonzero = differences[differences != 0]
                ranks = rankdata(np.abs(nonzero))
                effect = float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum())
                tests.append({
                    'dataset': dataset,
                    'reference': reference,
                    'metric': metric,
                    'pairs': len(rows),
                    'mean_difference': float(differences.mean()),
                    'p_value': float(wilcoxon(differences).pvalue),
                    'rank_biserial': effect,
                })
    holm(tests)
    result = {
        'test': 'Two-sided paired Wilcoxon signed-rank; Holm across all 40 tests',
        'tests': tests,
    }
    output = ROOT / 'results/comparison/adaptive_compensation_paired.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    for test in tests:
        if test['reference'] == 'fixed' and test['metric'] in ('sf', 'qabf'):
            print(test['dataset'], test['metric'], f"delta={test['mean_difference']:+.6f}",
                  f"Holm p={test['p_holm']:.3g}", f"effect={test['rank_biserial']:+.3f}")
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
