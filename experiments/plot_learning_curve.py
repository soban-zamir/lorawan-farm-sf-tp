"""Plot of delivery over time (main scenario, mean over the evaluation farms): grouped bars for
standard ADR, Always SF12 and our method in four periods after start-up. Run run_main.py first.

    python experiments/plot_learning_curve.py
Output: runs/figures/fig_learning_curve.pdf and .png
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import _common  # noqa: E402
from _common import Results, parse_args  # noqa: E402
from farmlora import EVAL_SEED_BASE, seed_count  # noqa: E402

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'Liberation Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 10, 'font.weight': 'bold',
    'axes.labelsize': 13, 'axes.labelweight': 'bold', 'axes.linewidth': 1.0,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'axes.axisbelow': True,
    'legend.frameon': True, 'legend.edgecolor': 'black', 'legend.fancybox': False, 'legend.framealpha': 1.0,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03, 'savefig.dpi': 300, 'pdf.fonttype': 42,
})
PERIODS = [(0, 1, '0-1 h'), (1, 6, '1-6 h'), (6, 24, '6-24 h'), (24, 72, 'Days 2-3')]
SHOW = ['adr', 'sf12', 'ours']
NAME = {'adr': 'Standard ADR', 'sf12': 'Always SF12', 'ours': 'Our method'}
STYLE = {'adr': dict(color='white', hatch='///'), 'sf12': dict(color='white', hatch='...'),
         'ours': dict(color='#3A3A3A', hatch='')}

if __name__ == '__main__':
    args = parse_args(__doc__)
    res = Results('main')
    seeds = [EVAL_SEED_BASE + i for i in range(seed_count('main_mix_200', args.mode))]
    fig, ax = plt.subplots(figsize=(3.42, 2.85))          # IEEE single column
    w = 0.26
    for j, m in enumerate(SHOW):
        hp = np.array([res.get('main_mix_200', s, m)['hourly_pdr'] for s in seeds]) * 100   # farms x hours
        vals = [hp[:, a:b].mean() for a, b, _ in PERIODS]
        ax.bar(np.arange(len(PERIODS)) + (j - 1) * w, vals, w, label=NAME[m], edgecolor='black', lw=0.8,
               zorder=3, **STYLE[m])
        print(f'{NAME[m]:20s} ' + '  '.join(f'{lab}: {v:5.1f} %' for (_, _, lab), v in zip(PERIODS, vals)))
    ax.axhline(95, color='black', lw=0.9, ls='--', zorder=4)
    ax.set_xticks(np.arange(len(PERIODS)))
    ax.set_xticklabels([lab for _, _, lab in PERIODS])
    ax.set_xlabel('Time since start')
    ax.set_ylabel('Delivery ratio (%)')
    ax.set_ylim(0, 120)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(axis='y', ls=':', lw=0.6, color='#999999', zorder=0)
    lg = ax.legend(prop={'weight': 'normal', 'size': 8.5}, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.0),
                   handlelength=1.4, columnspacing=0.8, handletextpad=0.4, borderaxespad=0.3)
    lg.get_frame().set_linewidth(0.8)
    out = os.path.join(_common.RUNS, 'figures')
    os.makedirs(out, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out, f'fig_learning_curve.{ext}'))
    print('Saved to', out)
