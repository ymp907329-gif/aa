import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METRICS = ('entropy', 'sd', 'sf', 'vif', 'qabf')
METHODS = ('CDDFuse', 'CDDFuse+HF', 'IFD-DBF', 'PPIFuse', 'PPIFuse+HF',
           'SwinFusion', 'SwinFusion+HF', 'SETFusion', 'U2Fusion', 'DenseFuse',
           'Average', 'Maximum', 'Ours-HF', 'Ours-AHF', 'Ours-VC')


def load_json(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def values(method, dataset, summary, controls):
    if method == 'Ours-HF':
        model = 'haar_source_reconstruction_highadd50'
    elif method == 'Ours-AHF':
        result = load_json('results/comparison/adaptive_main_balanced_three_seed_summary.json')
        return result[dataset]['adaptive']
    elif method == 'Ours-VC':
        model = 'haar_value_control'
    elif method in ('Average', 'Maximum'):
        return controls[dataset][method.lower()]
    elif method.endswith('+HF'):
        folder = method.removesuffix('+HF').lower()
        result = load_json(f'results/baselines_float/{folder}/{dataset}/compensation_beta50.json')
        return result['summary']['compensated']
    else:
        folder = {'IFD-DBF': 'ifddbf'}.get(method, method.lower())
        return load_json(f'results/baselines_float/{folder}/{dataset}/metrics_uniform_minmax.json')
    return summary[model][dataset]


def format_value(value, best, ours=False):
    if ours:
        mean = f"{value['mean']:.4f}±{value['std']:.4f}"
    else:
        mean = f'{value:.4f}'
    return f'**{mean}**' if best else mean


def replace_table(lines, dataset, table_number, summary, controls, chinese):
    caption = f'**{"表" if chinese else "Table"} {table_number}.**'
    caption_index = next(i for i, line in enumerate(lines) if line.startswith(caption))
    dataset_name = {'msrs': 'MSRS', 'roadscene': 'RoadScene', 'tno': 'TNO', 'vifb': 'VIFB'}[dataset]
    count = {'msrs': 130, 'roadscene': 221, 'tno': 229, 'vifb': 21}[dataset]
    if chinese:
        lines[caption_index] = (f'**表 {table_number}.** {dataset_name} 的 {count} 个测试图像对，采用校准协议。'
                                'Ours-HF：固定深度 4 + 固定补偿；Ours-AHF：固定深度 4 + 自适应补偿；'
                                'Ours-VC：价值控制深度，无补偿；+HF：对公开输出施加相同的固定补偿。'
                                '本文自有配置均为三种子均值±总体标准差，粗体表示每列最佳值。')
    else:
        lines[caption_index] = (f'**Table {table_number}.** {dataset_name} results on {count} pairs using the calibrated track. '
                                'Ours-HF: fixed depth 4 + fixed compensation; Ours-AHF: fixed depth 4 + adaptive compensation; '
                                'Ours-VC: value-controlled depth, no compensation; +HF: identical fixed compensation on public outputs. '
                                'Our configurations are mean±population SD over three seeds. Bold denotes the best value in each column.')
    start = next(i for i in range(caption_index, len(lines)) if lines[i].startswith('| Method |') or lines[i].startswith('| 方法 |'))
    end = start
    while end < len(lines) and lines[end].startswith('|'):
        end += 1
    raw = {method: values(method, dataset, summary, controls) for method in METHODS}
    means = {method: [raw[method][metric]['mean'] if isinstance(raw[method][metric], dict) else raw[method][metric]
                      for metric in METRICS] for method in METHODS}
    best = [max(METHODS, key=lambda method: means[method][index]) for index in range(len(METRICS))]
    header = '| 方法 | EN | SD | SF | VIF | Qabf |' if chinese else '| Method | EN | SD | SF | VIF | Qabf |'
    separator = '|---|---:|---:|---:|---:|---:|'
    rows = [header, separator]
    for method in METHODS:
        cells = [format_value(raw[method][metric], best[index] == method, method in ('Ours-HF', 'Ours-AHF', 'Ours-VC'))
                 for index, metric in enumerate(METRICS)]
        rows.append('| ' + method + ' | ' + ' | '.join(cells) + ' |')
    return lines[:start] + rows + lines[end:]


def main():
    summary = load_json('results/comparison/uniform_minmax_three_seed_summary.json')
    controls = load_json('results/comparison/learning_free_float.json')
    for language in ('en', 'zh'):
        path = ROOT / f'paper/submission_draft_{language}.md'
        lines = path.read_text(encoding='utf-8').splitlines()
        for table_number, dataset in enumerate(('msrs', 'roadscene', 'tno', 'vifb'), start=1):
            lines = replace_table(lines, dataset, table_number, summary, controls, language == 'zh')
        path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        print(f'Updated main tables: {path}')


if __name__ == '__main__':
    main()
