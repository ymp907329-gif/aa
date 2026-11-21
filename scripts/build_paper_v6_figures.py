from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

NAVY = '#234E70'
TEAL = '#2A9D8F'
ORANGE = '#E9A23B'
RED = '#D1495B'
PALE = '#EDF3F7'
GRAY = '#66737F'

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'axes.linewidth': 0.7,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 160,
    'savefig.dpi': 320,
    'savefig.bbox': 'tight',
})


def save(fig, name):
    fig.savefig(OUT / f'{name}.png', facecolor='white')
    fig.savefig(OUT / f'{name}.pdf', facecolor='white')
    plt.close(fig)


def box(ax, xy, width, height, text, color=NAVY, fill=PALE, size=9):
    patch = FancyBboxPatch(
        xy, width, height, boxstyle='round,pad=0.012,rounding_size=0.018',
        linewidth=1.2, edgecolor=color, facecolor=fill,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text,
            ha='center', va='center', fontsize=size, color='#17242D')
    return patch


def arrow(ax, start, end, color=GRAY, style='-|>', lw=1.25, connectionstyle='arc3'):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=12,
                                linewidth=lw, color=color,
                                connectionstyle=connectionstyle))


def method_overview():
    fig, ax = plt.subplots(figsize=(10.2, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.text(0.02, 0.94, '(a) Training: source-referenced optimization', weight='bold', color=NAVY)
    box(ax, (0.02, 0.68), 0.105, 0.13, 'Infrared\n$I_{ir}$', fill='#F5F7F8')
    box(ax, (0.02, 0.49), 0.105, 0.13, 'Visible\n$I_{vi}$', fill='#F5F7F8')
    box(ax, (0.18, 0.57), 0.16, 0.18, 'AID–DFD backbone\n+ global context', color=NAVY)
    box(ax, (0.405, 0.57), 0.13, 0.18, 'Fused output\n$F$', color=TEAL, fill='#EAF6F3')
    arrow(ax, (0.125, 0.745), (0.18, 0.69))
    arrow(ax, (0.125, 0.555), (0.18, 0.62))
    arrow(ax, (0.34, 0.66), (0.405, 0.66))

    box(ax, (0.18, 0.27), 0.16, 0.15, 'Source-energy\nconsistency (SEC)', color=ORANGE, fill='#FFF6E7')
    box(ax, (0.405, 0.27), 0.13, 0.15, 'Intensity, vector\ngradient and SSIM', color=ORANGE, fill='#FFF6E7', size=8)
    box(ax, (0.64, 0.57), 0.13, 0.18, 'CDDFuse\nteacher $F_t$', color=RED, fill='#FCEEF0')
    box(ax, (0.64, 0.27), 0.28, 0.15,
        'Reliability-gated gradient distillation\n'
        r'$m_t \leq \rho\,m_s$ and source-consistent direction',
        color=RED, fill='#FCEEF0', size=8)
    arrow(ax, (0.255, 0.49), (0.255, 0.42), color=ORANGE)
    arrow(ax, (0.47, 0.57), (0.47, 0.42), color=ORANGE)
    arrow(ax, (0.705, 0.57), (0.705, 0.42), color=RED)
    arrow(ax, (0.535, 0.345), (0.64, 0.345), color=ORANGE)
    arrow(ax, (0.78, 0.42), (0.535, 0.57), color=RED,
          connectionstyle='arc3,rad=-0.18')
    ax.text(0.60, 0.12, 'Training-only regularizers; no teacher or reconstruction decoder at inference',
            ha='center', color=GRAY, fontsize=8.5)

    ax.plot([0.02, 0.96], [0.08, 0.08], color='#C8D1D8', lw=0.8)
    ax.text(0.02, 0.035, '(b) Inference: deterministic gradient anchoring', weight='bold', color=NAVY)
    box(ax, (0.03, -0.255), 0.15, 0.17, 'Training-free\nSingle global $\lambda$\nNo learned parameters',
        color=GRAY, fill='#F5F7F8', size=8)
    box(ax, (0.25, -0.13), 0.15, 0.11, 'Network output $F$', color=TEAL, fill='#EAF6F3')
    box(ax, (0.25, -0.32), 0.15, 0.11, 'Source maximum $T$', color=NAVY, fill=PALE)
    box(ax, (0.52, -0.255), 0.20, 0.15, 'Gradient blending\n+ Poisson projection', color=ORANGE, fill='#FFF6E7')
    box(ax, (0.82, -0.255), 0.15, 0.15, 'Anchored output $U_\\lambda$', color=ORANGE, fill='#FFF6E7', size=8)
    arrow(ax, (0.40, -0.075), (0.52, -0.15))
    arrow(ax, (0.40, -0.265), (0.52, -0.21))
    arrow(ax, (0.72, -0.18), (0.82, -0.18))
    ax.text(0.62, -0.35, r'$G_\lambda=(1-\lambda)\nabla F+\lambda\nabla T$',
            ha='center', va='top', fontsize=9, color=GRAY)
    ax.set_ylim(-0.43, 1.0)
    save(fig, 'v6_method_overview')


def gradient_analysis():
    fig = plt.figure(figsize=(9.4, 3.65))
    grid = fig.add_gridspec(1, 2, width_ratios=[0.85, 1.55], wspace=0.30)
    ax = fig.add_subplot(grid[0, 0])
    ax.set_aspect('equal')
    ax.axis('off')
    origin = np.array([0.08, 0.12])
    gs = np.array([0.72, 0.0])
    gw = np.array([0.36, 0.48])
    parallel = np.array([0.36, 0.0])
    perp = np.array([0.0, 0.48])
    arrow(ax, origin, origin + gs, color=NAVY, lw=2)
    arrow(ax, origin, origin + gw, color=TEAL, lw=2)
    arrow(ax, origin, origin + parallel, color=ORANGE, lw=1.5)
    arrow(ax, origin + parallel, origin + gw, color=RED, lw=1.8)
    arrow(ax, origin + gs, origin + gs + perp, color=RED, lw=1.8)
    arrow(ax, origin, origin + gs + perp, color='#5C4B8A', lw=2)
    ax.plot([origin[0] + gw[0], origin[0] + gs[0] + perp[0]],
            [origin[1] + gw[1], origin[1] + gs[1] + perp[1]], ls=':', color=GRAY)
    ax.text(*(origin + gs * 0.55 + [0, -0.07]), r'$g_s$', color=NAVY, fontsize=12)
    ax.text(*(origin + gw * 0.50 + [-0.08, 0.02]), r'$g_w$', color=TEAL, fontsize=12)
    ax.text(*(origin + parallel * 0.52 + [0, 0.05]), 'parallel', color=ORANGE, fontsize=8)
    ax.text(*(origin + gs + perp * 0.52 + [0.025, 0]), r'$g_w^\perp$', color=RED, fontsize=11)
    ax.text(0.49, 0.72, r'$g^*=g_s+g_w^\perp$', color='#5C4B8A', fontsize=10)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.23, 0.86)
    ax.text(0.0, 1.04, '(a) Conservative reference construction', transform=ax.transAxes,
            weight='bold', color=NAVY)
    ax.text(0.0, -0.12, 'A diagnostic reference, not a universal upper bound.',
            transform=ax.transAxes, fontsize=8, color=GRAY)

    ax = fig.add_subplot(grid[0, 1])
    datasets = ['MSRS', 'RoadScene', 'TNO']
    strong = [16.57, 14.81, 6.57]
    fused = [15.4, 11.0, 7.1]
    comp = [2.0, 4.9, 4.1]
    x = np.arange(3)
    width = 0.32
    bars1 = ax.bar(x - width / 2, strong, width, color=NAVY, label='Stronger-source SF')
    bars2 = ax.bar(x + width / 2, fused, width, color=TEAL, label='Baseline fused SF')
    ax.set_ylabel('Spatial frequency (255-level scale)')
    ax.set_xticks(x, datasets)
    ax.set_ylim(0, 20)
    ax.grid(axis='y', alpha=0.18)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.bar_label(bars1, fmt='%.2f', padding=2, fontsize=7.5)
    ax.bar_label(bars2, fmt='%.1f', padding=2, fontsize=7.5)
    ax2 = ax.twinx()
    ax2.plot(x, comp, color=RED, marker='o', lw=1.5, label='Orthogonal-energy ratio')
    ax2.set_ylim(0, 7)
    ax2.set_ylabel('Orthogonal energy / stronger-source energy (%)', color=RED)
    ax2.tick_params(axis='y', colors=RED)
    ax2.spines['top'].set_visible(False)
    for xi, value in zip(x, comp):
        ax2.annotate(f'{value:.1f}%', (xi, value), xytext=(0, 6),
                     textcoords='offset points', ha='center', color=RED, fontsize=8)
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc='upper right', frameon=False)
    ax.set_title('(b) Measured source-reference statistics', loc='left', weight='bold', color=NAVY)
    save(fig, 'v6_gradient_analysis')


def qualitative():
    source = Image.open(OUT / 'v6_qualitative_source.png').convert('RGB')
    source = np.asarray(source)
    fig, axes = plt.subplots(2, 4, figsize=(10.0, 5.25))
    titles = ['Infrared', 'Visible', 'Network output', 'After GCA']
    for row in range(2):
        for col in range(4):
            crop = source[row * 256:(row + 1) * 256, col * 256:(col + 1) * 256]
            axes[row, col].imshow(crop, cmap='gray')
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            for spine in axes[row, col].spines.values():
                spine.set_linewidth(0.7)
                spine.set_edgecolor('#AAB4BC')
            if row == 0:
                axes[row, col].set_title(titles[col], fontsize=10, pad=5)
    fig.text(0.035, 0.72, 'MSRS', rotation=90, va='center', ha='center', fontsize=10, weight='bold')
    fig.text(0.035, 0.27, 'RoadScene', rotation=90, va='center', ha='center', fontsize=10, weight='bold')
    fig.subplots_adjust(left=0.065, right=0.995, top=0.93, bottom=0.015, wspace=0.025, hspace=0.035)
    save(fig, 'v6_qualitative_labeled')


def tradeoffs():
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.65))
    ax = axes[0]
    labels = ['Backbone', '+GCA', '+SEC', 'Full']
    sf = [17.60, 17.53, 18.36, 17.92]
    qabf = [0.665, 0.717, 0.672, 0.720]
    colors = [GRAY, TEAL, ORANGE, RED]
    for i, (label, x, y, color) in enumerate(zip(labels, sf, qabf, colors)):
        ax.scatter(x, y, s=58, color=color, zorder=3)
        ax.annotate(label, (x, y), xytext=(5, 5 if i != 1 else -13),
                    textcoords='offset points', fontsize=8)
    ax.set_xlabel('MSRS SF')
    ax.set_ylabel('MSRS Qabf')
    ax.grid(alpha=0.2)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_title('(a) Ablation: fidelity–frequency trade-off', loc='left', weight='bold', color=NAVY)

    ax = axes[1]
    settings = [
        ('1.08, no GCA', 17.72, 0.480, GRAY, 'o'),
        ('1.08, GCA 0.5', 18.58, 0.512, TEAL, 's'),
        ('1.30, no GCA', 18.75, 0.476, ORANGE, '^'),
        ('1.30, GCA 0.3', 18.79, 0.509, '#5C4B8A', 'D'),
        ('1.30, GCA 0.5', 19.03, 0.516, RED, 'P'),
    ]
    for label, x, y, color, marker in settings:
        ax.scatter(x, y, s=52, color=color, marker=marker, label=label)
    ax.set_xlabel('RoadScene SF')
    ax.set_ylabel('RoadScene Qabf')
    ax.grid(alpha=0.2)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, loc='upper left', fontsize=7)
    ax.set_title('(b) SEC margin and anchoring strength', loc='left', weight='bold', color=NAVY)
    fig.tight_layout()
    save(fig, 'v6_tradeoff_analysis')


if __name__ == '__main__':
    method_overview()
    gradient_analysis()
    qualitative()
    tradeoffs()
    print('Generated four figure sets in', OUT)
