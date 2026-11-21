import argparse
import csv
import json
import os

import numpy as np
from scipy.stats import wilcoxon


METRICS = ('entropy', 'sd', 'sf', 'mi', 'vif', 'qabf')


def read_rows(path):
    with open(path, encoding='utf-8-sig', newline='') as file:
        return {row['name']: row for row in csv.DictReader(file)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline_root', required=True)
    parser.add_argument('--candidate_root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    results = {}
    tests = []
    for dataset in ('msrs', 'roadscene', 'tno'):
        baseline = read_rows(os.path.join(args.baseline_root, dataset, 'metrics.csv'))
        candidate = read_rows(os.path.join(args.candidate_root, dataset, 'metrics.csv'))
        if baseline.keys() != candidate.keys():
            raise ValueError(f'Image mismatch: {dataset}')
        names = sorted(baseline)
        results[dataset] = {}
        for metric in METRICS:
            before = np.array([float(baseline[name][metric]) for name in names])
            after = np.array([float(candidate[name][metric]) for name in names])
            difference = after - before
            test = {
                'baseline_mean': float(before.mean()),
                'candidate_mean': float(after.mean()),
                'mean_difference': float(difference.mean()),
                'improved_images': int(np.sum(difference > 0)),
                'image_count': len(names),
                'p_value': float(wilcoxon(after, before, alternative='two-sided').pvalue),
            }
            results[dataset][metric] = test
            tests.append(test)
    ordered = sorted(tests, key=lambda test: test['p_value'])
    running = 0.0
    for rank, test in enumerate(ordered):
        running = max(running, (len(ordered) - rank) * test['p_value'])
        test['holm_p_value'] = float(min(1.0, running))
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    for dataset, metrics in results.items():
        print(dataset)
        for metric, test in metrics.items():
            print(metric, f"delta={test['mean_difference']:+.6f}", f"Holm p={test['holm_p_value']:.3g}")
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
