"""Plot completed DAPT evidence without changing any experiment artifact.

The paired intervals are recomputed from saved test-run scores. This script
does not run inference, train models, or reuse validation scores as test data.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'papers/target_pretraining/plots'
RUNS = json.loads((ROOT / 'experiments/dapt/final_cpu/results.json').read_text())['runs']
EQUAL_RUNS = json.loads(
    (ROOT / 'experiments/dapt/equal_update/results.json').read_text()
)['runs']
SEEDS = [7, 42, 99, 123, 2026]
COLORS = {'A': '#21618c', 'S': '#b45309', 'M': '#42734b'}


def scores(condition, fraction=100, seeds=SEEDS):
    experiment = 'A' if fraction == 100 else 'C'
    return np.array([RUNS[f'{experiment}_{condition}_s{s}_{fraction}']['f1'] for s in seeds])


def effect(condition, fraction=100, seeds=SEEDS):
    differences = scores(condition, fraction, seeds) - scores('N', fraction, seeds)
    half_width = t.ppf(.975, len(seeds)-1) * differences.std(ddof=1) / np.sqrt(len(seeds))
    return differences.mean(), half_width


def equal_scores(condition, seeds=SEEDS):
    return np.array([EQUAL_RUNS[f'D_{condition}_s{s}_100']['f1'] for s in seeds])


def equal_effect(condition, seeds=SEEDS):
    differences = equal_scores(condition, seeds) - equal_scores('N', seeds)
    half_width = t.ppf(.975, len(seeds)-1) * differences.std(ddof=1) / np.sqrt(len(seeds))
    return differences.mean(), half_width


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(OUT / f'{name}.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    assert len(RUNS) == 47
    assert len(EQUAL_RUNS) == 20
    np.testing.assert_allclose([scores(c).mean() for c in 'NASM'],
                               [85.862624,85.746362,85.391397,85.478109], atol=1e-6)
    np.testing.assert_allclose([equal_scores(c).mean() for c in 'NASM'],
                               [85.862624,85.851511,85.912546,85.581982], atol=1e-6)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.1), layout='constrained')
    values = np.array([scores(c) for c in 'NASM'])
    palette = plt.get_cmap('tab10')
    for i, seed in enumerate(SEEDS):
        ax.plot(range(4), values[:,i], color=palette(i), marker='o', ms=4,
                lw=.8, alpha=.8, label=str(seed))
    ax.scatter(range(4), values.mean(axis=1), marker='_', s=320, linewidths=3,
               color='black', zorder=5, label='Mean')
    ax.set_xticks(range(4), ['No DAPT', 'Authentic', 'Synthetic', 'Mixed'])
    ax.set_ylabel('Test F1 (0-100 scale)')
    ax.set_title('(a) Full-data scores: five matched seeds', fontsize=11)
    ax.grid(axis='y', alpha=.18)
    ax.legend(title='Training seed', ncol=3, fontsize=8, title_fontsize=9,
              loc='upper center', bbox_to_anchor=(.5,-.12), frameon=False)
    for y, c in enumerate('ASM'):
        mean, half = effect(c)
        bx.errorbar(mean, y, xerr=half, fmt='o', capsize=4, color=COLORS[c], lw=1.6)
    bx.axvline(0, color='grey', ls='--', lw=1)
    bx.set_yticks(range(3), ['Authentic - N', 'Synthetic - N', 'Mixed - N'])
    bx.invert_yaxis()
    bx.set_ylim(2.5,-.5)
    bx.set_xlabel('Paired F1 difference (95% interval)')
    bx.set_title('(b) Effects relative to no DAPT', fontsize=11)
    bx.grid(axis='x', alpha=.18)
    save(fig, '07_final_full_data')

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.1), layout='constrained')
    equal_values = np.array([equal_scores(c) for c in 'NASM'])
    palette = plt.get_cmap('tab10')
    for i, seed in enumerate(SEEDS):
        ax.plot(range(4), equal_values[:,i], color=palette(i), marker='o', ms=4,
                lw=.8, alpha=.8, label=str(seed))
    ax.scatter(range(4), equal_values.mean(axis=1), marker='_', s=320, linewidths=3,
               color='black', zorder=5, label='Mean')
    ax.set_xticks(range(4), ['No DAPT', 'Authentic', 'Synthetic', 'Mixed'])
    ax.set_ylabel('Test F1 (0-100 scale)')
    ax.set_title('(a) Equal-update scores: five matched seeds', fontsize=11)
    ax.grid(axis='y', alpha=.18)
    ax.legend(title='Training seed', ncol=3, fontsize=8, title_fontsize=9,
              loc='upper center', bbox_to_anchor=(.5,-.12), frameon=False)
    for y, c in enumerate('ASM'):
        mean, half = equal_effect(c)
        bx.errorbar(mean, y, xerr=half, fmt='o', capsize=4, color=COLORS[c], lw=1.6)
    bx.axvline(0, color='grey', ls='--', lw=1)
    bx.set_yticks(range(3), ['Authentic - N', 'Synthetic - N', 'Mixed - N'])
    bx.invert_yaxis()
    bx.set_ylim(2.5,-.5)
    bx.set_xlabel('Paired F1 difference (95% interval)')
    bx.set_title('(b) Effects relative to no DAPT', fontsize=11)
    bx.grid(axis='x', alpha=.18)
    save(fig, '09_equal_update_full_data')

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), sharey=True, layout='constrained')
    for ax, c, title in zip(axes, 'ASM', ['Authentic','Synthetic','Mixed']):
        means, halves = zip(*(effect(c, f, SEEDS[:3]) for f in (25,50,100)))
        ax.errorbar(range(3), means, yerr=halves, fmt='o', capsize=5,
                    lw=1.6, color=COLORS[c])
        ax.axhline(0, color='grey', ls='--', lw=1)
        ax.set_xticks(range(3), ['25%', '50%', '100%'])
        ax.set_xlim(-.4,2.4)
        ax.set_title(title)
        ax.grid(axis='y', alpha=.18)
        ax.set_xlabel('IPR fine-tuning fraction')
    axes[0].set_ylabel('DAPT - no DAPT F1 (95% interval)')
    save(fig, '08_final_supervision')
    print('Wrote three final-study figures from the audited A/B/C and D test records.')


if __name__ == '__main__':
    main()
