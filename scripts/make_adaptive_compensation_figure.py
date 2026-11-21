import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('msrs', 'roadscene', 'tno', 'vifb')
LABELS = ('MSRS', 'RoadScene', 'TNO', 'VIFB')


def load(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def main():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.linewidth': 0.8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.7))

    ax = axes[0]
    scales = (1.0, 1.3, 1.6, 1.9)
    points = [load(f'results/diagnostics/adaptive_compensation_main_balanced/validation_scale_{scale}.json') for scale in scales]
    sf = [point['summary']['adaptive']['sf'] for point in points]
    qabf = [point['summary']['adaptive']['qabf'] for point in points]
    ax.plot(sf, qabf, color='#4C78A8', marker='o', linewidth=1.5, label='Adaptive β(x) scale')
    for x, y, scale in zip(sf, qabf, scales):
        ax.annotate(f'{scale:.1f}', (x, y), xytext=(0, 7), textcoords='offset points', ha='center', fontsize=8)
    reference = points[0]['summary']
    ax.scatter(reference['none']['sf'], reference['none']['qabf'], marker='o', s=65,
               color='#BAB0AC', edgecolor='black', linewidth=0.5, label='No compensation', zorder=3)
    ax.scatter(reference['fixed']['sf'], reference['fixed']['qabf'], marker='s', s=65,
               color='#F58518', edgecolor='black', linewidth=0.5, label='Fixed β=0.5', zorder=3)
    selected = scales.index(1.3)
    ax.scatter(sf[selected], qabf[selected], marker='*', s=130, color='#E45756', edgecolor='black',
               linewidth=0.5, label='Selected scale=1.3', zorder=4)
    ax.set_xlabel('SF')
    ax.set_ylabel('Qabf')
    ax.set_title('(a) Validation coefficient selection', fontweight='bold')
    ax.grid(color='#E5E5E5', linewidth=0.6)
    ax.legend(frameon=False, fontsize=7, loc='lower right')

    ours_delta = {metric: [] for metric in ('sf', 'qabf')}
    ppi_delta = {metric: [] for metric in ('sf', 'qabf')}
    for dataset in DATASETS:
        ours = load(f'results/research_float/haar_adaptive_compensation_main_balanced/{dataset}/three_way_scale130.json')['summary']
        ppi = load(f'results/baselines_float/ppifuse/{dataset}/adaptive_compensation_main_balanced_scale130.json')['summary']
        for metric in ('sf', 'qabf'):
            ours_delta[metric].append(ours['adaptive'][metric] - ours['fixed'][metric])
            ppi_delta[metric].append(ppi['adaptive'][metric] - ppi['fixed'][metric])

    x = np.arange(len(DATASETS))
    width = 0.35
    for ax, metric, panel in zip(axes[1:], ('sf', 'qabf'), ('(b)', '(c)')):
        ax.bar(x - width / 2, ours_delta[metric], width, color='#4C78A8', label='Ours backbone')
        ax.bar(x + width / 2, ppi_delta[metric], width, color='#54A24B', label='PPIFuse')
        ax.axhline(0, color='#333333', linewidth=0.8)
        ax.set_xticks(x, LABELS)
        ax.set_ylabel(f'Adaptive − fixed Δ{metric.upper()}')
        ax.set_title(f'{panel} Test-set change after compensation', fontweight='bold')
        ax.grid(axis='y', color='#E5E5E5', linewidth=0.6)
        ax.legend(frameon=False, fontsize=7.5)
    figure.tight_layout(w_pad=2.0)
    output = ROOT / 'paper/figures/adaptive_compensation_analysis.png'
    figure.savefig(output, dpi=400, bbox_inches='tight', facecolor='white')
    figure.savefig(output.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(figure)
    print(f'Saved: {output} and .pdf')


if __name__ == '__main__':
    main()
