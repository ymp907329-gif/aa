import argparse
import csv
import json
from pathlib import Path

import numpy as np

from utils.value_control import ridge_apply, ridge_fit, source_features


METRICS = ('entropy', 'sd', 'sf', 'mi', 'vif', 'qabf')


def read_depths(path, mode):
    with open(path, encoding='utf-8-sig', newline='') as file:
        rows = [row for row in csv.DictReader(file) if row['output'] == mode]
    names = sorted({row['name'] for row in rows})
    return names, {
        name: {int(row['depth']): row for row in rows if row['name'] == name}
        for name in names
    }


def metric_values(rows, names, depth, metric):
    return np.array([float(rows[name][depth][metric]) for name in names])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_metrics', required=True)
    parser.add_argument('--test_metrics', required=True)
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--target', choices=('mi', 'qabf', 'balanced'), required=True)
    parser.add_argument('--mode', choices=('identity', 'minmax'), default='minmax')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    train_names, train_rows = read_depths(args.train_metrics, args.mode)
    test_names, test_rows = read_depths(args.test_metrics, args.mode)
    train_x = source_features(args.data_root, 'train', train_names, args.size)
    test_x = source_features(args.data_root, 'test', test_names, args.size)
    depths_available = sorted(next(iter(train_rows.values())))
    controller = {
        'target': args.target, 'size': args.size,
        'depths': np.asarray(depths_available),
        'metric_mean': np.zeros(len(METRICS)), 'metric_scale': np.ones(len(METRICS)),
    }
    result = {
        'target': args.target, 'mode': args.mode,
        'train_samples': len(train_names), 'test_samples': len(test_names),
        'train_test_name_overlap': len(set(train_names) & set(test_names)),
        'metrics': {},
    }
    if args.target == 'balanced':
        predictions = np.zeros((len(test_names), len(depths_available), len(METRICS)))
        for metric_index, metric in enumerate(METRICS):
            train_values = np.concatenate([
                metric_values(train_rows, train_names, depth, metric) for depth in depths_available
            ])
            controller['metric_mean'][metric_index] = train_values.mean()
            controller['metric_scale'][metric_index] = train_values.std() + 1e-8
            for depth_index, depth in enumerate(depths_available):
                train_y = metric_values(train_rows, train_names, depth, metric)
                fitted = ridge_fit(train_x, train_y)
                predictions[:, depth_index, metric_index] = ridge_apply(fitted, test_x)
                controller[f'weights_{depth_index}_{metric_index}'] = fitted['weights']
                controller[f'intercept_{depth_index}_{metric_index}'] = fitted['intercept']
        controller['feature_mean'] = fitted['feature_mean']
        controller['feature_scale'] = fitted['feature_scale']
        utility = ((predictions - controller['metric_mean']) / controller['metric_scale']).mean(axis=2)
        depths = np.asarray(depths_available)[utility.argmax(axis=1)]
        result['balanced_utility'] = 'Equal-weight mean of six metrics standardized on training data'
    else:
        train_y = (metric_values(train_rows, train_names, 4, args.target)
                   - metric_values(train_rows, train_names, 3, args.target))
        test_y = (metric_values(test_rows, test_names, 4, args.target)
                  - metric_values(test_rows, test_names, 3, args.target))
        fitted = ridge_fit(train_x, train_y)
        predicted = ridge_apply(fitted, test_x)
        controller.update(fitted)
        depths = np.where(predicted > 0, 4, 3)
        result['marginal_r2'] = float(
            1 - np.sum((test_y - predicted) ** 2) / np.sum((test_y - test_y.mean()) ** 2)
        )
        result['marginal_sign_accuracy'] = float(np.mean((predicted > 0) == (test_y > 0)))
    result['selected_depth4_fraction'] = float(np.mean(depths == 4))
    result['depth_histogram'] = {
        str(depth): int(np.sum(depths == depth)) for depth in depths_available
    }
    fixed = depths_available if args.target == 'balanced' else [3, 4]
    for metric in METRICS:
        result['metrics'][metric] = {}
        for depth in fixed:
            result['metrics'][metric][f'depth{depth}'] = float(
                metric_values(test_rows, test_names, depth, metric).mean()
            )
        result['metrics'][metric]['value_controlled'] = float(np.mean([
            float(test_rows[name][int(depth)][metric]) for name, depth in zip(test_names, depths)
        ]))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    np.savez(output.with_suffix('.npz'), **controller)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
