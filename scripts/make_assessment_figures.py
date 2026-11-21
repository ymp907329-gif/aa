import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper' / 'figures' / 'control_and_beta_pareto.png'


def main():
    controls = json.loads((ROOT / 'results/comparison/controller_controls_float.json').read_text(encoding='utf-8'))
    beta = json.loads((ROOT / 'results/diagnostics/high_frequency_compensation/beta_pareto_float_validation.json').read_text(encoding='utf-8'))
    fixed = json.loads((ROOT / 'results/research/haar_channelwise_control/roadscene/metrics.json').read_text(encoding='utf-8'))
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.linewidth': 0.8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    ax = axes[0]
    road = controls['roadscene']['controls']
    points = {
        'Fixed depth 4': (336.91, fixed['vif'], '#4C78A8', 's'),
        'Random': (road['random']['tracked_conv_g'], road['random']['metrics']['vif'], '#888888', 'o'),
        'Entropy threshold': (road['entropy_threshold']['tracked_conv_g'], road['entropy_threshold']['metrics']['vif'], '#54A24B', '^'),
        'Value-controlled': (road['value_controlled']['tracked_conv_g'], road['value_controlled']['metrics']['vif'], '#F58518', 'D'),
        'Oracle depth': (road['oracle']['tracked_conv_g'], road['oracle']['metrics']['vif'], '#B279A2', '*'),
    }
    for name, (ops, vif, color, marker) in points.items():
        ax.scatter(ops, vif, s=80 if name == 'Oracle depth' else 60, color=color, marker=marker,
                   edgecolor='black', linewidth=0.5, label=name, zorder=3)
    ax.set_xlabel('Tracked convolution operations (G)')
    ax.set_ylabel('VIF')
    ax.set_title('(a) Controller quality–computation trade-off\nRoadScene, seed 42', fontweight='bold')
    ax.grid(color='#E2E2E2', linewidth=0.6)
    ax.legend(frameon=False, fontsize=7.5, loc='lower right')

    ax = axes[1]
    rows = beta['summary']['high_add']
    betas = sorted(float(key) for key in rows)
    vif = [rows[str(key)]['vif'] for key in betas]
    qabf = [rows[str(key)]['qabf'] for key in betas]
    ax.plot(qabf, vif, color='#4C78A8', linewidth=1.4, marker='o')
    for x, y, value in zip(qabf, vif, betas):
        ax.annotate(f'{value:g}', (x, y), xytext=(0, 7), textcoords='offset points', ha='center', fontsize=8)
    ax.scatter(qabf[betas.index(0.5)], vif[betas.index(0.5)], marker='D', s=70,
               color='#F58518', edgecolor='black', linewidth=0.5, zorder=4, label='Selected β=0.5')
    ax.set_xlabel('Qabf')
    ax.set_ylabel('VIF')
    ax.set_title('(b) Source-detail compensation trade-off\nMSRS validation, seed 42', fontweight='bold')
    ax.grid(color='#E2E2E2', linewidth=0.6)
    ax.legend(frameon=False, loc='lower left')
    figure.tight_layout(w_pad=2.5)
    figure.savefig(OUT, dpi=400, bbox_inches='tight', facecolor='white')
    figure.savefig(OUT.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(figure)
    print(f'Saved: {OUT} and .pdf')


if __name__ == '__main__':
    main()
