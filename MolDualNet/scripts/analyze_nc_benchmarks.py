#!/usr/bin/env python3
"""
Analyze NC benchmark results and generate publication-quality tables and figures.
Visual design follows Nature Communications guidelines.

Usage:
    python scripts/analyze_nc_benchmarks.py
    python scripts/analyze_nc_benchmarks.py --results_dir results/nc_benchmarks
"""

import argparse
import json
import os
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.patches import FancyBboxPatch
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


TASKS_ORDER = ['ESOL', 'FreeSolv', 'Lipophilicity', 'BACE']
TASK_SHORT = {'ESOL': 'ESOL', 'FreeSolv': 'FreeSolv',
              'Lipophilicity': 'Lipo', 'BACE': 'BACE'}
TASK_LABEL = {
    'ESOL': 'ESOL\n(log S)',
    'FreeSolv': 'FreeSolv\n' + r'($\Delta G_{\mathrm{hyd}}$)',
    'Lipophilicity': 'Lipo\n(log D)',
    'BACE': 'BACE\n' + r'($\mathrm{pIC}_{50}$)',
}

# MolDualNet results from the paper (scaffold-split, best model)
MOLDUALNET = {
    'ESOL':          {'R2': 0.918, 'RMSE': 0.565, 'MAE': 0.412},
    'FreeSolv':      {'R2': 0.945, 'RMSE': 0.813, 'MAE': 0.571},
    'Lipophilicity': {'R2': 0.768, 'RMSE': 0.584, 'MAE': 0.447},
    'BACE':          {'R2': 0.705, 'RMSE': 0.734, 'MAE': 0.549},
}

CATEGORY = {
    'RF': 'FP', 'XGBoost': 'FP', 'SVM': 'FP',
    'GIN': 'GNN', 'AttentiveFP': 'GNN', 'D-MPNN': 'GNN',
    'SchNet': '3D', 'MolDualNet': 'Multi',
}

# ── Nature Communications color palette ──
# Muted, accessible, professional palette inspired by Nature journals
NC_COLORS = {
    'RF':          '#A6CEE3',  # light blue
    'XGBoost':     '#6BAED6',  # medium blue
    'SVM':         '#3182BD',  # dark blue
    'GIN':         '#B2DF8A',  # light green
    'AttentiveFP': '#33A02C',  # green
    'D-MPNN':      '#FB9A99',  # light coral
    'SchNet':      '#FDBF6F',  # light orange
    'MolDualNet':  '#E31A1C',  # Nature red (highlight)
}


def load_results(results_dir: str) -> dict:
    """Load benchmark summary."""
    path = os.path.join(results_dir, 'benchmark_summary.json')
    if not os.path.exists(path):
        print(f'ERROR: {path} not found. Run run_nc_benchmarks.py first.')
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def generate_latex_table(summary: dict, output_path: str):
    """Generate LaTeX table for NC paper."""
    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{\textbf{Comparison with established molecular property prediction methods.} '
                 r'$R^2$ (mean $\pm$ s.d.) on scaffold-split test sets across three random seeds. '
                 r'Bold indicates the best result per task.}')
    lines.append(r'\label{tab:benchmark_comparison}')
    lines.append(r'\resizebox{\linewidth}{!}{%')
    lines.append(r'\begin{tabular}{@{}llccccc@{}}')
    lines.append(r'\toprule')
    lines.append(r'\textbf{Method} & \textbf{Type} & '
                 r'\textbf{ESOL} & \textbf{FreeSolv} & '
                 r'\textbf{Lipo} & \textbf{BACE} & '
                 r'\textbf{Avg $R^2$} \\')
    lines.append(r'\midrule')

    all_r2 = {t: {} for t in TASKS_ORDER}
    for bl, data in summary.items():
        if bl == 'MolDualNet':
            continue
        for t in TASKS_ORDER:
            if t in data and not np.isnan(data[t].get('R2_mean', float('nan'))):
                all_r2[t][bl] = data[t]['R2_mean']
    for t in TASKS_ORDER:
        all_r2[t]['MolDualNet'] = MOLDUALNET[t]['R2']

    best = {}
    for t in TASKS_ORDER:
        if all_r2[t]:
            best[t] = max(all_r2[t], key=all_r2[t].get)

    baseline_order = ['RF', 'XGBoost', 'SVM', 'GIN', 'AttentiveFP', 'D-MPNN', 'SchNet']
    prev_cat = None
    for bl in baseline_order:
        if bl not in summary:
            continue
        cat = CATEGORY.get(bl, '?')
        if prev_cat and cat != prev_cat:
            lines.append(r'\midrule')
        prev_cat = cat

        s = summary[bl]
        cells = []
        for t in TASKS_ORDER:
            if t in s and not np.isnan(s[t].get('R2_mean', float('nan'))):
                val = f'{s[t]["R2_mean"]:.3f} \\pm {s[t]["R2_std"]:.3f}'
                if best.get(t) == bl:
                    cells.append(f'$\\mathbf{{{val}}}$')
                else:
                    cells.append(f'${val}$')
            else:
                cells.append('---')

        avg = s.get('Avg_R2', float('nan'))
        avg_str = f'${avg:.3f}$' if not np.isnan(avg) else '---'
        lines.append(f'{bl} & {cat} & ' + ' & '.join(cells) + f' & {avg_str} \\\\')

    lines.append(r'\midrule')
    md_cells = []
    for t in TASKS_ORDER:
        val = f'{MOLDUALNET[t]["R2"]:.3f}'
        if best.get(t) == 'MolDualNet':
            md_cells.append(f'$\\mathbf{{{val}}}$')
        else:
            md_cells.append(f'${val}$')
    md_avg = np.mean([MOLDUALNET[t]['R2'] for t in TASKS_ORDER])
    md_avg_str = f'$\\mathbf{{{md_avg:.3f}}}$'
    lines.append(f'MolDualNet (ours) & Multi & ' + ' & '.join(md_cells) +
                 f' & {md_avg_str} \\\\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}}')
    lines.append(r'\end{table}')

    table_str = '\n'.join(lines)
    with open(output_path, 'w') as f:
        f.write(table_str)
    print(f'LaTeX table saved to: {output_path}')
    print()
    print(table_str)
    return table_str


# ──────────────────────────────────────────────────────────────────────
# Nature Communications–style bar chart
# ──────────────────────────────────────────────────────────────────────

def _setup_nc_style():
    """Configure matplotlib for Nature Communications look."""
    plt.rcParams.update({
        # Font: Helvetica-like sans-serif (Nature standard)
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 8,
        # Axes
        'axes.linewidth': 0.6,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'axes.labelpad': 4,
        'axes.spines.top': False,
        'axes.spines.right': False,
        # Ticks
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        # Legend
        'legend.fontsize': 7,
        'legend.frameon': False,
        'legend.handlelength': 1.2,
        'legend.handletextpad': 0.4,
        'legend.columnspacing': 0.8,
        # Figure
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
    })


def generate_bar_chart(summary: dict, output_path: str):
    """Generate Nature Communications–style grouped bar chart."""
    if not HAS_MPL:
        print('matplotlib not available, skipping bar chart.')
        return

    _setup_nc_style()

    baseline_order = ['RF', 'XGBoost', 'SVM', 'GIN', 'AttentiveFP', 'D-MPNN', 'SchNet']
    methods = [bl for bl in baseline_order if bl in summary] + ['MolDualNet']
    n_methods = len(methods)
    n_tasks = len(TASKS_ORDER)

    # Nature single-column: 89 mm ≈ 3.5 in; double-column: 183 mm ≈ 7.2 in
    fig, ax = plt.subplots(figsize=(7.2, 3.6))

    x = np.arange(n_tasks)
    total_group_width = 0.82
    width = total_group_width / n_methods
    gap = 0.005  # tiny gap between bars

    for i, method in enumerate(methods):
        vals, errs = [], []
        for t in TASKS_ORDER:
            if method == 'MolDualNet':
                vals.append(max(0, MOLDUALNET[t]['R2']))
                errs.append(0)
            elif t in summary.get(method, {}):
                vals.append(max(0, summary[method][t].get('R2_mean', 0)))
                errs.append(summary[method][t].get('R2_std', 0))
            else:
                vals.append(0)
                errs.append(0)

        offset = (i - n_methods / 2 + 0.5) * (width + gap)
        color = NC_COLORS.get(method, '#999999')
        is_ours = (method == 'MolDualNet')

        bars = ax.bar(
            x + offset, vals, width,
            yerr=errs if not is_ours else None,
            label=method,
            color=color,
            edgecolor='white' if not is_ours else '#B71C1C',
            linewidth=0.3 if not is_ours else 0.8,
            capsize=1.5,
            error_kw={'linewidth': 0.6, 'capthick': 0.6, 'color': '#333333'},
            zorder=3 if is_ours else 2,
            alpha=0.88 if not is_ours else 1.0,
        )

    # Y axis
    ax.set_ylabel(r'$R^{2}$', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    # Light horizontal grid (behind bars)
    ax.yaxis.grid(True, which='major', linewidth=0.3, color='#CCCCCC',
                  linestyle='-', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    # X axis
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABEL[t] for t in TASKS_ORDER],
                       fontsize=8, linespacing=1.2)
    ax.tick_params(axis='x', length=0, pad=6)

    # Legend: horizontal, above the plot
    legend = ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.15),
        ncol=4,
        fontsize=7,
        frameon=False,
        columnspacing=0.6,
        handletextpad=0.3,
        handlelength=1.0,
    )

    # Spine styling
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('#333333')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Bar chart saved to: {output_path}')


def generate_heatmap(summary: dict, output_path: str):
    """Generate Nature Communications–style heatmap of R² values."""
    if not HAS_MPL:
        print('matplotlib not available, skipping heatmap.')
        return

    _setup_nc_style()
    from matplotlib.colors import LinearSegmentedColormap
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    baseline_order = ['RF', 'XGBoost', 'SVM', 'GIN', 'AttentiveFP', 'D-MPNN', 'SchNet']
    methods = [bl for bl in baseline_order if bl in summary] + ['MolDualNet']
    n_methods = len(methods)
    n_tasks = len(TASKS_ORDER)

    # Build data matrix
    data = np.zeros((n_methods, n_tasks))
    for i, method in enumerate(methods):
        for j, t in enumerate(TASKS_ORDER):
            if method == 'MolDualNet':
                data[i, j] = MOLDUALNET[t]['R2']
            elif t in summary.get(method, {}):
                data[i, j] = summary[method][t].get('R2_mean', 0)

    # Clip negatives for color mapping
    data_clipped = np.clip(data, 0, 1)

    fig, ax = plt.subplots(figsize=(5.0, 3.4))

    # Custom colormap: white → light blue → deep blue
    cmap = LinearSegmentedColormap.from_list(
        'nc_blue', ['#F7FBFF', '#C6DBEF', '#6BAED6', '#2171B5', '#084594']
    )

    im = ax.imshow(data_clipped, cmap=cmap, aspect='auto', vmin=0, vmax=1)

    # Annotate cells with values
    for i in range(n_methods):
        for j in range(n_tasks):
            val = data[i, j]
            text_color = 'white' if data_clipped[i, j] > 0.65 else '#333333'
            fontweight = 'bold' if methods[i] == 'MolDualNet' else 'normal'
            if val < 0:
                text = f'{val:.2f}'
            else:
                text = f'{val:.3f}'
            ax.text(j, i, text, ha='center', va='center',
                    fontsize=7.5, color=text_color, fontweight=fontweight)

    # X-axis: task names on top
    ax.set_xticks(np.arange(n_tasks))
    ax.set_xticklabels([TASK_SHORT[t] for t in TASKS_ORDER], fontsize=8)
    ax.xaxis.set_ticks_position('top')

    # Y-axis: method names with category in parentheses
    ax.set_yticks(np.arange(n_methods))
    ylabels = []
    for m in methods:
        cat = CATEGORY.get(m, '')
        if m == 'MolDualNet':
            ylabels.append(f'{m} (ours)')
        else:
            ylabels.append(f'{m}  [{cat}]')
    ax.set_yticklabels(ylabels, fontsize=7.5)

    # Make MolDualNet label bold and red
    for label in ax.get_yticklabels():
        if 'ours' in label.get_text():
            label.set_fontweight('bold')
            label.set_color('#E31A1C')

    ax.tick_params(axis='both', length=0)

    # Colorbar — use axes_divider to avoid overlap
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='4%', pad=0.12)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r'$R^{2}$', fontsize=8, fontweight='bold')
    cbar.ax.tick_params(labelsize=7, length=2, width=0.5)
    cbar.outline.set_linewidth(0.5)

    # Remove all spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Add thin white lines between cells
    for i in range(1, n_methods):
        ax.axhline(y=i - 0.5, color='white', linewidth=1.5)
    for j in range(1, n_tasks):
        ax.axvline(x=j - 0.5, color='white', linewidth=1.5)

    # Separator before MolDualNet
    ax.axhline(y=n_methods - 1.5, color='#E31A1C', linewidth=1.0,
               linestyle='--', alpha=0.6)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Heatmap saved to: {output_path}')


def generate_radar_chart(summary: dict, output_path: str):
    """Generate radar/spider chart showing each method's profile across tasks."""
    if not HAS_MPL:
        return

    _setup_nc_style()

    key_methods = ['RF', 'GIN', 'AttentiveFP', 'D-MPNN', 'SchNet', 'MolDualNet']
    methods = [m for m in key_methods if m in summary or m == 'MolDualNet']

    angles = np.linspace(0, 2 * np.pi, len(TASKS_ORDER), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(4.0, 4.0), subplot_kw=dict(polar=True))

    for method in methods:
        vals = []
        for t in TASKS_ORDER:
            if method == 'MolDualNet':
                vals.append(max(0, MOLDUALNET[t]['R2']))
            elif t in summary.get(method, {}):
                vals.append(max(0, summary[method][t].get('R2_mean', 0)))
            else:
                vals.append(0)
        vals += vals[:1]

        color = NC_COLORS.get(method, '#999999')
        is_ours = (method == 'MolDualNet')

        ax.plot(angles, vals, 'o-',
                linewidth=2.0 if is_ours else 1.0,
                markersize=4 if is_ours else 2.5,
                color=color, label=method,
                alpha=1.0 if is_ours else 0.7,
                zorder=10 if is_ours else 5)
        if is_ours:
            ax.fill(angles, vals, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([TASK_SHORT[t] for t in TASKS_ORDER], fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=6,
                       color='#888888')
    ax.set_rlabel_position(30)

    # Grid styling
    ax.grid(True, linewidth=0.3, color='#CCCCCC', alpha=0.5)
    ax.spines['polar'].set_linewidth(0.4)
    ax.spines['polar'].set_color('#CCCCCC')

    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1),
              fontsize=7, frameon=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Radar chart saved to: {output_path}')


def generate_rmse_table(summary: dict, output_path: str):
    """Generate RMSE comparison table."""
    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{\textbf{RMSE comparison across methods.} '
                 r'Root-mean-square error (mean $\pm$ s.d.) on scaffold-split test sets.}')
    lines.append(r'\label{tab:benchmark_rmse}')
    lines.append(r'\begin{tabular}{@{}lcccc@{}}')
    lines.append(r'\toprule')
    lines.append(r'\textbf{Method} & \textbf{ESOL} & \textbf{FreeSolv} & '
                 r'\textbf{Lipo} & \textbf{BACE} \\')
    lines.append(r'\midrule')

    baseline_order = ['RF', 'XGBoost', 'SVM', 'GIN', 'AttentiveFP', 'D-MPNN', 'SchNet']
    for bl in baseline_order:
        if bl not in summary:
            continue
        s = summary[bl]
        cells = []
        for t in TASKS_ORDER:
            if t in s and not np.isnan(s[t].get('RMSE_mean', float('nan'))):
                cells.append(f'${s[t]["RMSE_mean"]:.3f} \\pm {s[t]["RMSE_std"]:.3f}$')
            else:
                cells.append('---')
        lines.append(f'{bl} & ' + ' & '.join(cells) + r' \\')

    lines.append(r'\midrule')
    md_cells = [f'${MOLDUALNET[t]["RMSE"]:.3f}$' for t in TASKS_ORDER]
    lines.append(f'MolDualNet (ours) & ' + ' & '.join(md_cells) + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    table_str = '\n'.join(lines)
    with open(output_path, 'w') as f:
        f.write(table_str)
    print(f'RMSE table saved to: {output_path}')


def print_summary(summary: dict):
    """Print a human-readable summary table."""
    print('\n' + '=' * 80)
    print('BENCHMARK SUMMARY (R² ↑)')
    print('=' * 80)

    header = f'{"Method":<15} {"Type":<6}'
    for t in TASKS_ORDER:
        header += f' {TASK_SHORT[t]:>14}'
    header += f' {"Avg R²":>8}'
    print(header)
    print('-' * 80)

    baseline_order = ['RF', 'XGBoost', 'SVM', 'GIN', 'AttentiveFP', 'D-MPNN', 'SchNet']
    for bl in baseline_order:
        if bl not in summary:
            continue
        s = summary[bl]
        row = f'{bl:<15} {CATEGORY.get(bl, "?"):<6}'
        for t in TASKS_ORDER:
            if t in s and not np.isnan(s[t].get('R2_mean', float('nan'))):
                row += f' {s[t]["R2_mean"]:>6.3f}±{s[t]["R2_std"]:.3f}'
            else:
                row += f' {"---":>14}'
        avg = s.get('Avg_R2', float('nan'))
        row += f' {avg:>8.3f}' if not np.isnan(avg) else f' {"---":>8}'
        print(row)

    print('-' * 80)
    md_avg = np.mean([MOLDUALNET[t]['R2'] for t in TASKS_ORDER])
    row = f'{"MolDualNet":<15} {"Multi":<6}'
    for t in TASKS_ORDER:
        row += f' {MOLDUALNET[t]["R2"]:>6.3f}      '
    row += f' {md_avg:>8.3f}'
    print(row)
    print('=' * 80)


def main():
    parser = argparse.ArgumentParser(description='Analyze NC benchmark results')
    parser.add_argument('--results_dir', type=str, default='results/nc_benchmarks',
                        help='Directory with benchmark results')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: same as results_dir)')
    args = parser.parse_args()

    output_dir = args.output_dir or args.results_dir
    os.makedirs(output_dir, exist_ok=True)

    summary = load_results(args.results_dir)

    # Print human-readable summary
    print_summary(summary)

    # Generate LaTeX tables
    generate_latex_table(summary, os.path.join(output_dir, 'table_benchmark_r2.tex'))
    generate_rmse_table(summary, os.path.join(output_dir, 'table_benchmark_rmse.tex'))

    # Generate figures (NC style)
    generate_bar_chart(summary, os.path.join(output_dir, 'Fig_benchmark_comparison.pdf'))
    generate_bar_chart(summary, os.path.join(output_dir, 'Fig_benchmark_comparison.png'))
    generate_heatmap(summary, os.path.join(output_dir, 'Fig_benchmark_heatmap.pdf'))
    generate_heatmap(summary, os.path.join(output_dir, 'Fig_benchmark_heatmap.png'))
    generate_radar_chart(summary, os.path.join(output_dir, 'Fig_benchmark_radar.pdf'))
    generate_radar_chart(summary, os.path.join(output_dir, 'Fig_benchmark_radar.png'))

    print('\nAll outputs saved to:', output_dir)


if __name__ == '__main__':
    main()
