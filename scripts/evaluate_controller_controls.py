import json
from pathlib import Path

import numpy as np

from scripts.evaluate_value_depth_control import metric_values, read_depths
from utils.value_control import source_features


METRICS = ('entropy', 'sd', 'sf', 'mi', 'vif', 'qabf')
DATASETS = ('msrs', 'roadscene')


def selected_metrics(rows, names, depths):
    return {
        metric: float(np.mean([float(rows[name][int(depth)][metric]) for name, depth in zip(names, depths)]))
        for metric in METRICS
    }


def summarize(result, depths, metrics):
    result['metrics'] = metrics
    result['mean_depth'] = float(np.mean(depths))
    result['tracked_conv_g'] = float(np.mean(220.94 + (np.asarray(depths) - 1) * (336.91 - 220.94) / 3))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--float_protocol', action='store_true')
    args = parser.parse_args()
    mode = 'identity' if args.float_protocol else 'minmax'
    value_root = 'results/research_float/haar_value_control' if args.float_protocol else 'results/research/haar_value_control_minmax'
    output_name = 'controller_controls_float.json' if args.float_protocol else 'controller_controls.json'
    output = {}
    for dataset in DATASETS:
        suffix = '' if dataset == 'msrs' else '_roadscene'
        train_path = Path(f'results/diagnostics/iteration_quality/haar_{dataset}_train.csv')
        test_path = Path(f'results/diagnostics/iteration_quality/haar_{dataset}.csv')
        if args.float_protocol and dataset == 'roadscene':
            test_path = Path('results/diagnostics/iteration_quality/haar_roadscene_float.csv')
        train_names, train_rows = read_depths(train_path, 'minmax')
        test_names, test_rows = read_depths(test_path, mode)
        train_x = source_features(f'data/{dataset.upper() if dataset == "msrs" else "RoadScene"}', 'train', train_names, 256)
        test_x = source_features(f'data/{dataset.upper() if dataset == "msrs" else "RoadScene"}', 'test', test_names, 256)
        train_entropy = train_x[:, [2, 6]].mean(axis=1)
        test_entropy = test_x[:, [2, 6]].mean(axis=1)
        threshold = float(np.median(train_entropy))
        heuristic_depths = np.where(test_entropy >= threshold, 4, 1)
        rng = np.random.default_rng(42)
        random_depths = rng.integers(1, 5, size=len(test_names))
        metric_mean = np.array([
            np.concatenate([metric_values(train_rows, train_names, depth, metric) for depth in (1, 2, 3, 4)]).mean()
            for metric in METRICS
        ])
        metric_scale = np.array([
            np.concatenate([metric_values(train_rows, train_names, depth, metric) for depth in (1, 2, 3, 4)]).std() + 1e-8
            for metric in METRICS
        ])
        oracle_depths = []
        for name in test_names:
            values = np.array([
                [float(test_rows[name][depth][metric]) for metric in METRICS]
                for depth in (1, 2, 3, 4)
            ])
            utility = ((values - metric_mean) / metric_scale).mean(axis=1)
            oracle_depths.append(int(np.argmax(utility)) + 1)
        oracle_depths = np.asarray(oracle_depths)
        result = {'train_test_name_overlap': len(set(train_names) & set(test_names)), 'controls': {}}
        for name, depths in (
            ('random', random_depths), ('entropy_threshold', heuristic_depths), ('oracle', oracle_depths)
        ):
            control = {'entropy_threshold': threshold} if name == 'entropy_threshold' else {}
            summarize(control, depths, selected_metrics(test_rows, test_names, depths))
            result['controls'][name] = control
        value_path = Path(f'{value_root}/{dataset}/metrics.json')
        value = json.loads(value_path.read_text(encoding='utf-8'))
        depth_path = Path(f'results/research/haar_value_control_minmax/{dataset}/metrics.json')
        depth_value = json.loads(depth_path.read_text(encoding='utf-8'))
        control = {}
        summarize(control, [round(depth_value['mean_depth'])] * len(test_names), {metric: value[metric] for metric in METRICS})
        control['tracked_conv_g'] = float(220.94 + (depth_value['mean_depth'] - 1) * (336.91 - 220.94) / 3)
        control['mean_depth'] = depth_value['mean_depth']
        result['controls']['value_controlled'] = control
        output[dataset] = result
    path = Path(f'results/comparison/{output_name}')
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    for dataset, result in output.items():
        print(dataset)
        for name, control in result['controls'].items():
            print(name, 'depth', round(control['mean_depth'], 4), 'ops', round(control['tracked_conv_g'], 2),
                  'VIF', round(control['metrics']['vif'], 4), 'Qabf', round(control['metrics']['qabf'], 4))
    print(f'Saved: {path}')


if __name__ == '__main__':
    main()
