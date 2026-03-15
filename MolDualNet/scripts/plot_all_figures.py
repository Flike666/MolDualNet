#!/usr/bin/env python3
"""
Publication-quality figure generation for MolDualNet paper.
Strict Nature Communications style: Arial, 5-7pt labels, 8pt bold panels,
colorblind-safe palette, 300+ DPI, no top/right spines.

Generates:
  1. Fig_main_results.png       — 2×4 grid: scatter (a-d) + residual (e-h)
  2. Fig_cross_dataset.png      — 3-row composite: scatter (a-d), violin (e), bars (f-h)
  3. Fig_attention.png           — 2-panel attention heatmap (a, b)
  4. Individual figures in results/figures/ for flexible LaTeX use
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

import scienceplots  # noqa: F401
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from scipy.stats import pearsonr, gaussian_kde
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ═══════════════════════════════════════════════════════════════════════════════
# Nature Communications Style
# ═══════════════════════════════════════════════════════════════════════════════
plt.style.use(['science', 'nature', 'no-latex'])
plt.rcParams.update({
    # Font
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':         7,
    'axes.labelsize':    7,
    'axes.titlesize':    8,
    'xtick.labelsize':   6,
    'ytick.labelsize':   6,
    'legend.fontsize':   6,
    # Axes
    'axes.linewidth':    0.5,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    # Ticks
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size':  3,
    'ytick.major.size':  3,
    'xtick.minor.size':  1.5,
    'ytick.minor.size':  1.5,
    'xtick.direction':   'in',
    'ytick.direction':   'in',
    # Lines
    'lines.linewidth':   0.8,
    # Save
    'savefig.dpi':       600,
    'savefig.bbox':      'tight',
    'savefig.pad_inches': 0.02,
    'figure.dpi':        150,
})

# ═══════════════════════════════════════════════════════════════════════════════
# Colorblind-safe palette (blue-orange-green-purple)
# ═══════════════════════════════════════════════════════════════════════════════
COLORS = {
    'ESOL_logS':          '#4878D0',  # muted blue
    'FreeSolv_hydration': '#D65F5F',  # muted red
    'Lipophilicity_logD': '#6ACC65',  # muted green
    'BACE_pIC50':         '#956CB4',  # muted purple
}

TASK_LABELS = {
    'ESOL_logS':          ('ESOL',         'log $S$ (mol L$^{-1}$)'),
    'FreeSolv_hydration': ('FreeSolv',     r'$\Delta G_{\mathrm{hyd}}$ (kcal mol$^{-1}$)'),
    'Lipophilicity_logD': ('Lipophilicity', 'log $D$'),
    'BACE_pIC50':         ('BACE',         'pIC$_{50}$'),
}

TASK_ORDER_MAIN = ['ESOL_logS', 'FreeSolv_hydration', 'Lipophilicity_logD', 'BACE_pIC50']
TASK_ORDER_CROSS = ['ESOL_logS', 'Lipophilicity_logD', 'FreeSolv_hydration', 'BACE_pIC50']
PANEL = 'abcdefghijklmnop'

# Nature: single col = 89mm = 3.5in, double col = 183mm = 7.2in, max height = 247mm = 9.7in
FIG_DOUBLE = 7.2   # inches
FIG_SINGLE = 3.5


def panel_label(ax, letter, x=-0.14, y=1.06):
    """Bold 8pt panel label."""
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=8, fontweight='bold', va='top', ha='left')


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════
def load_main_predictions(npz_path='results/predictions.npz'):
    data = np.load(npz_path, allow_pickle=True)
    out = {}
    for task in TASK_ORDER_MAIN:
        pred = data[f'{task}_regression']
        true = data[f'target_{task}_value']
        mask = data[f'target_{task}_mask'].astype(bool)
        out[task] = (true[mask], pred[mask])
    return out


def load_cross_predictions(csv_dir='results/cross_dataset_validation/per_task_predictions'):
    out = {}
    for task in TASK_ORDER_CROSS:
        fname = f'{task}_predictions.csv'
        path = os.path.join(csv_dir, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        out[task] = {
            'y_true': df['experimental'].values,
            'y_pred': df['predicted'].values,
            'baseline': df['baseline'].values if 'baseline' in df.columns else None,
            'error': df['error'].values,
        }
    return out


def load_cross_metrics(json_path='results/cross_dataset_validation/cross_dataset_validation_results.json'):
    with open(json_path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Main results (2×4: scatter + residual)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_main_results(main_data, save_path):
    fig, axes = plt.subplots(2, 4, figsize=(FIG_DOUBLE, 3.6))

    # Row 1: scatter
    for idx, task in enumerate(TASK_ORDER_MAIN):
        ax = axes[0, idx]
        true, pred = main_data[task]
        name, unit = TASK_LABELS[task]
        color = COLORS[task]

        r2 = r2_score(true, pred)
        rmse = np.sqrt(mean_squared_error(true, pred))
        mae = mean_absolute_error(true, pred)

        # density coloring
        if len(true) > 30:
            xy = np.vstack([true, pred])
            z = gaussian_kde(xy)(xy)
            order = z.argsort()
            true_s, pred_s, z_s = true[order], pred[order], z[order]
        else:
            true_s, pred_s, z_s = true, pred, np.ones_like(true)

        ax.scatter(true_s, pred_s, c=z_s, cmap='viridis', s=10, alpha=0.8,
                   edgecolors='none', rasterized=True, zorder=2)

        lims = [min(true.min(), pred.min()) - 0.4,
                max(true.max(), pred.max()) + 0.4]
        ax.plot(lims, lims, '-', color='#999999', linewidth=0.5, zorder=1)
        ax.fill_between(lims, [l - 1 for l in lims], [l + 1 for l in lims],
                        color='#DDDDDD', alpha=0.3, zorder=0, linewidth=0)

        # linear fit
        c = np.polyfit(true, pred, 1)
        fit_x = np.linspace(lims[0], lims[1], 100)
        ax.plot(fit_x, np.polyval(c, fit_x), '-', color=color, linewidth=0.9, zorder=3)

        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(f'Experimental {unit}')
        ax.set_ylabel(f'Predicted {unit}')

        # metrics box
        txt = f'$R^2$ = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}'
        ax.text(0.04, 0.96, txt, transform=ax.transAxes, fontsize=5,
                va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#CCCCCC',
                          alpha=0.92, linewidth=0.4))

        ax.set_title(name, fontweight='bold', fontsize=7, pad=2)
        panel_label(ax, PANEL[idx], x=-0.22, y=1.10)

    # Row 2: residual distribution
    for idx, task in enumerate(TASK_ORDER_MAIN):
        ax = axes[1, idx]
        true, pred = main_data[task]
        res = pred - true
        name, unit = TASK_LABELS[task]
        color = COLORS[task]

        nbins = min(20, max(10, len(res) // 6))
        ax.hist(res, bins=nbins, density=True, color=color, alpha=0.30,
                edgecolor='white', linewidth=0.3, zorder=2)

        if len(res) > 10:
            kde = gaussian_kde(res)
            xr = np.linspace(res.min() - 0.5, res.max() + 0.5, 200)
            ax.plot(xr, kde(xr), '-', color=color, linewidth=1.0, zorder=3)

        ax.axvline(0, color='#999999', ls='--', lw=0.5, zorder=1)

        mu, sigma = res.mean(), res.std()
        txt = f'$\\mu$ = {mu:.3f}\n$\\sigma$ = {sigma:.3f}\n$n$ = {len(res)}'
        ax.text(0.96, 0.96, txt, transform=ax.transAxes, fontsize=5,
                va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#CCCCCC',
                          alpha=0.92, linewidth=0.4))

        ax.set_xlabel(f'Residual ({unit})')
        if idx == 0:
            ax.set_ylabel('Density')
        ax.set_title(name, fontweight='bold', fontsize=7, pad=2)
        panel_label(ax, PANEL[idx + 4], x=-0.22, y=1.10)

    plt.tight_layout(w_pad=0.8, h_pad=1.2)
    fig.savefig(save_path, dpi=600)
    fig.savefig(save_path.replace('.png', '.pdf'))
    plt.close(fig)
    print(f'  Saved: {save_path}')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Cross-dataset generalization composite
# ═══════════════════════════════════════════════════════════════════════════════
def fig_cross_dataset(cross_data, cross_metrics, save_path):
    fig = plt.figure(figsize=(FIG_DOUBLE, 7.8))
    gs = gridspec.GridSpec(3, 4, figure=fig,
                           height_ratios=[1.0, 0.7, 0.7],
                           hspace=0.40, wspace=0.40)

    mol_m = cross_metrics['moldualnet_metrics']
    base_m = cross_metrics['baseline_metrics']

    # ── Row 1: 2×2 scatter (a-d) ──────────────────────────────────────────────
    scatter_colors = ['#4878D0', '#6ACC65', '#D65F5F', '#956CB4']
    scatter_tasks = TASK_ORDER_CROSS
    validation_labels = {
        'ESOL_logS': 'AqSolDB (external)',
        'Lipophilicity_logD': 'Scaffold split',
        'FreeSolv_hydration': 'Scaffold split',
        'BACE_pIC50': 'ChEMBL (external)',
    }

    for idx, task in enumerate(scatter_tasks):
        row, col = divmod(idx, 2)
        # Map to 2×2 in top half
        ax = fig.add_subplot(gs[0, idx])
        d = cross_data.get(task)
        if d is None:
            continue

        y_true = d['y_true']
        y_pred = d['y_pred']
        n = len(y_true)
        name, unit = TASK_LABELS[task]
        color = scatter_colors[idx]
        m = mol_m.get(task, {})

        # Choose plot style based on sample size
        if n > 1000:
            ax.hexbin(y_true, y_pred, gridsize=50, cmap='YlOrBr',
                      mincnt=1, linewidths=0.1, edgecolors='#CCCCCC',
                      alpha=0.9, rasterized=True, zorder=2)
        elif n > 200:
            xy = np.vstack([y_true, y_pred])
            z = gaussian_kde(xy)(xy)
            order = z.argsort()
            ax.scatter(y_true[order], y_pred[order], c=z[order], cmap='viridis',
                       s=6, alpha=0.7, edgecolors='none', rasterized=True, zorder=2)
        else:
            ax.scatter(y_true, y_pred, c=color, s=12, alpha=0.6,
                       edgecolors='white', linewidths=0.3, zorder=2)

        # Diagonal + band
        all_v = np.concatenate([y_true, y_pred])
        margin = (all_v.max() - all_v.min()) * 0.06 + 0.3
        lo, hi = all_v.min() - margin, all_v.max() + margin
        ax.plot([lo, hi], [lo, hi], '-', color='#999999', lw=0.5, zorder=1)
        ax.fill_between([lo, hi], [lo - 1, hi - 1], [lo + 1, hi + 1],
                        color='#DDDDDD', alpha=0.25, zorder=0, linewidth=0)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect('equal', adjustable='box')

        ax.set_xlabel(f'Experimental {unit}')
        ax.set_ylabel(f'Predicted {unit}')

        # Metrics
        r2_v = m.get('R2', {}).get('value', 0)
        rmse_v = m.get('RMSE', {}).get('value', 0)
        txt = f"$R^2$ = {r2_v:.3f}\nRMSE = {rmse_v:.3f}\n$n$ = {n:,}"
        ax.text(0.04, 0.96, txt, transform=ax.transAxes, fontsize=5,
                va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#CCCCCC',
                          alpha=0.92, linewidth=0.4))

        # Validation type
        vt = validation_labels.get(task, '')
        ax.text(0.96, 0.04, vt, transform=ax.transAxes, fontsize=4.5,
                ha='right', va='bottom', fontstyle='italic', color='#888888')

        ax.set_title(name, fontweight='bold', fontsize=7, pad=2)
        panel_label(ax, PANEL[idx])

    # ── Row 2: violin error distribution (e) + bar RMSE (f) ──────────────────
    # Violin spanning 2 cols
    ax_v = fig.add_subplot(gs[1, :2])
    positions = []
    labels = []
    all_errors = []

    for i, task in enumerate(TASK_ORDER_CROSS):
        d = cross_data.get(task)
        if d is None:
            continue
        err = d['error']
        err = err[np.isfinite(err)]
        all_errors.append(err)
        positions.append(i + 1)
        n = mol_m[task]['n']
        labels.append(f"{TASK_LABELS[task][0]}\n($n$={n:,})")

    vp = ax_v.violinplot(all_errors, positions=positions, showmeans=False,
                         showmedians=False, showextrema=False, widths=0.65)
    for i, body in enumerate(vp['bodies']):
        task = TASK_ORDER_CROSS[i]
        c = COLORS[task]
        body.set_facecolor(c)
        body.set_alpha(0.25)
        body.set_edgecolor(c)
        body.set_linewidth(0.6)

    bp = ax_v.boxplot(all_errors, positions=positions, widths=0.15,
                      patch_artist=True, showfliers=False,
                      whiskerprops=dict(lw=0.6, color='#555555'),
                      capprops=dict(lw=0.6, color='#555555'),
                      boxprops=dict(lw=0.6, fc='white', ec='#555555'),
                      medianprops=dict(lw=1.0, color='#333333'))

    ax_v.axhline(0, color='#999999', ls='--', lw=0.5)
    for i, task in enumerate(TASK_ORDER_CROSS):
        mse = mol_m[task]['Mean_Signed_Error']
        ax_v.plot(positions[i], mse, 'D', color=COLORS[task], ms=3,
                  mec='white', mew=0.4, zorder=5)

    ax_v.set_xticks(positions)
    ax_v.set_xticklabels(labels, fontsize=5.5)
    ax_v.set_ylabel('Signed error (pred $-$ exp)')
    ax_v.grid(True, axis='y', alpha=0.1, ls=':', lw=0.3)
    panel_label(ax_v, 'e')

    # Bar chart: RMSE comparison (f)
    ax_b1 = fig.add_subplot(gs[1, 2])
    tasks_bl = [t for t in ['ESOL_logS', 'Lipophilicity_logD'] if t in base_m]
    x = np.arange(len(tasks_bl))
    w = 0.32
    mol_vals = [mol_m[t]['RMSE']['value'] for t in tasks_bl]
    base_vals = [base_m[t]['RMSE']['value'] for t in tasks_bl]

    ax_b1.bar(x - w/2, mol_vals, w, color='#4878D0', ec='white', lw=0.3, label='MolDualNet')
    ax_b1.bar(x + w/2, base_vals, w, color='#BDBDBD', ec='white', lw=0.3, label='RDKit')

    for i in range(len(tasks_bl)):
        ax_b1.text(x[i] - w/2, mol_vals[i] + 0.05, f'{mol_vals[i]:.2f}',
                   ha='center', va='bottom', fontsize=5, color='#4878D0')
        ax_b1.text(x[i] + w/2, base_vals[i] + 0.05, f'{base_vals[i]:.2f}',
                   ha='center', va='bottom', fontsize=5, color='#888888')
        red = (base_vals[i] - mol_vals[i]) / base_vals[i] * 100
        ax_b1.annotate(f'{red:.0f}%↓', xy=(x[i], max(mol_vals[i], base_vals[i]) + 0.25),
                       fontsize=5, ha='center', color='#D65F5F', fontweight='bold')

    bl_labels = ['log $S$', 'log $P$']
    ax_b1.set_xticks(x)
    ax_b1.set_xticklabels(bl_labels, fontsize=6)
    ax_b1.set_ylabel('RMSE')
    ax_b1.set_title('RMSE', fontweight='bold', fontsize=7, pad=2)
    ax_b1.legend(fontsize=5, frameon=True, framealpha=0.9, edgecolor='#CCCCCC')
    ax_b1.grid(True, axis='y', alpha=0.1, ls=':', lw=0.3)
    panel_label(ax_b1, 'f')

    # Bar chart: MAE (g)
    ax_b2 = fig.add_subplot(gs[1, 3])
    mol_mae = [mol_m[t]['MAE']['value'] for t in tasks_bl]
    base_mae = [base_m[t]['MAE']['value'] for t in tasks_bl]

    ax_b2.bar(x - w/2, mol_mae, w, color='#4878D0', ec='white', lw=0.3)
    ax_b2.bar(x + w/2, base_mae, w, color='#BDBDBD', ec='white', lw=0.3)

    for i in range(len(tasks_bl)):
        ax_b2.text(x[i] - w/2, mol_mae[i] + 0.03, f'{mol_mae[i]:.2f}',
                   ha='center', va='bottom', fontsize=5, color='#4878D0')
        ax_b2.text(x[i] + w/2, base_mae[i] + 0.03, f'{base_mae[i]:.2f}',
                   ha='center', va='bottom', fontsize=5, color='#888888')

    ax_b2.set_xticks(x)
    ax_b2.set_xticklabels(bl_labels, fontsize=6)
    ax_b2.set_ylabel('MAE')
    ax_b2.set_title('MAE', fontweight='bold', fontsize=7, pad=2)
    ax_b2.grid(True, axis='y', alpha=0.1, ls=':', lw=0.3)
    panel_label(ax_b2, 'g')

    # ── Row 3: R² comparison (h) + BACE baselines (i) ────────────────────────
    ax_r2 = fig.add_subplot(gs[2, :2])
    mol_r2 = [mol_m[t]['R2']['value'] for t in tasks_bl]
    base_r2 = [base_m[t]['R2']['value'] for t in tasks_bl]
    mol_ci = []
    for t in tasks_bl:
        v = mol_m[t]['R2']['value']
        lo = mol_m[t]['R2'].get('CI_95_lo', v)
        hi = mol_m[t]['R2'].get('CI_95_hi', v)
        mol_ci.append([v - lo, hi - v])
    ci_arr = np.array(mol_ci).T

    ax_r2.bar(x - w/2, mol_r2, w, color='#4878D0', ec='white', lw=0.3,
              yerr=ci_arr, capsize=2, error_kw={'lw': 0.5, 'capthick': 0.5},
              label='MolDualNet')
    ax_r2.bar(x + w/2, base_r2, w, color='#BDBDBD', ec='white', lw=0.3,
              label='RDKit')

    for i in range(len(tasks_bl)):
        y_mol = mol_r2[i] + (0.03 if mol_r2[i] >= 0 else -0.06)
        y_base = base_r2[i] + (0.03 if base_r2[i] >= 0 else -0.06)
        va_mol = 'bottom' if mol_r2[i] >= 0 else 'top'
        va_base = 'bottom' if base_r2[i] >= 0 else 'top'
        ax_r2.text(x[i] - w/2, y_mol, f'{mol_r2[i]:.3f}',
                   ha='center', va=va_mol, fontsize=5, color='#4878D0')
        ax_r2.text(x[i] + w/2, y_base, f'{base_r2[i]:.3f}',
                   ha='center', va=va_base, fontsize=5, color='#888888')

    ax_r2.axhline(0, color='#999999', ls='-', lw=0.3)
    ax_r2.set_xticks(x)
    ax_r2.set_xticklabels(bl_labels, fontsize=6)
    ax_r2.set_ylabel('$R^2$')
    ax_r2.set_title('$R^2$ (cross-dataset)', fontweight='bold', fontsize=7, pad=2)
    ax_r2.legend(fontsize=5, frameon=True, framealpha=0.9, edgecolor='#CCCCCC')
    ax_r2.grid(True, axis='y', alpha=0.1, ls=':', lw=0.3)
    panel_label(ax_r2, 'h')

    # BACE baselines (i) - load from separate JSON
    bace_json = 'results/cross_dataset_validation/bace_baseline_comparison.json'
    if os.path.exists(bace_json):
        ax_bace = fig.add_subplot(gs[2, 2:])
        with open(bace_json) as f:
            bace_data = json.load(f)

        methods = list(bace_data['baselines'].keys()) + ['MolDualNet']
        rmse_vals = [bace_data['baselines'][m]['rmse'] for m in bace_data['baselines']]
        rmse_vals.append(mol_m['BACE_pIC50']['RMSE']['value'])
        r2_vals = [bace_data['baselines'][m]['r2'] for m in bace_data['baselines']]
        r2_vals.append(mol_m['BACE_pIC50']['R2']['value'])

        short_names = ['Ridge', 'RF', 'GBT', 'Ours']
        colors_bace = ['#BDBDBD', '#BDBDBD', '#BDBDBD', '#4878D0']
        xb = np.arange(len(short_names))

        ax_bace.bar(xb, rmse_vals, 0.5, color=colors_bace, ec='white', lw=0.3)
        for i, v in enumerate(rmse_vals):
            ax_bace.text(xb[i], v + 0.02, f'{v:.2f}', ha='center', va='bottom',
                         fontsize=5, color=colors_bace[i] if colors_bace[i] != '#BDBDBD' else '#888888')

        ax_bace.set_xticks(xb)
        ax_bace.set_xticklabels(short_names, fontsize=6)
        ax_bace.set_ylabel('RMSE')
        ax_bace.set_title('BACE pIC$_{50}$ (ChEMBL)', fontweight='bold', fontsize=7, pad=2)
        ax_bace.grid(True, axis='y', alpha=0.1, ls=':', lw=0.3)
        panel_label(ax_bace, 'i')

    fig.savefig(save_path, dpi=600)
    fig.savefig(save_path.replace('.png', '.pdf'))
    plt.close(fig)
    print(f'  Saved: {save_path}')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Attention heatmaps (redrawn from raw data if available, else embed)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_attention(save_path):
    import matplotlib.image as mpimg
    imgs = [
        'results/attention_viz/attn_mol2_0_gnn_to_trans.png',
        'results/attention_viz/attn_mol1_0_gnn_to_trans.png',
    ]
    fig, axes = plt.subplots(2, 1, figsize=(FIG_DOUBLE, 5.5))
    for ax, img_path, letter in zip(axes, imgs, ['a', 'b']):
        img = mpimg.imread(img_path)
        ax.imshow(img, aspect='auto')
        ax.set_axis_off()
        panel_label(ax, letter, x=-0.02, y=1.04)

    plt.tight_layout(h_pad=0.3)
    fig.savefig(save_path, dpi=600)
    plt.close(fig)
    print(f'  Saved: {save_path}')


# ═══════════════════════════════════════════════════════════════════════════════
# Individual figures for results/figures/
# ═══════════════════════════════════════════════════════════════════════════════
def save_individual_scatter(main_data, out_dir):
    for task in TASK_ORDER_MAIN:
        true, pred = main_data[task]
        name, unit = TASK_LABELS[task]
        color = COLORS[task]
        r2 = r2_score(true, pred)
        rmse = np.sqrt(mean_squared_error(true, pred))
        mae = mean_absolute_error(true, pred)

        fig, ax = plt.subplots(figsize=(FIG_SINGLE * 0.85, FIG_SINGLE * 0.8))
        if len(true) > 30:
            xy = np.vstack([true, pred])
            z = gaussian_kde(xy)(xy)
            order = z.argsort()
            ax.scatter(true[order], pred[order], c=z[order], cmap='viridis',
                       s=14, alpha=0.8, edgecolors='none', rasterized=True)
        else:
            ax.scatter(true, pred, c=color, s=14, alpha=0.7,
                       edgecolors='white', linewidths=0.3)

        lims = [min(true.min(), pred.min()) - 0.4, max(true.max(), pred.max()) + 0.4]
        ax.plot(lims, lims, '-', color='#999999', lw=0.5)
        ax.fill_between(lims, [l-1 for l in lims], [l+1 for l in lims],
                        color='#DDDDDD', alpha=0.3, linewidth=0)
        c = np.polyfit(true, pred, 1)
        fit_x = np.linspace(lims[0], lims[1], 100)
        ax.plot(fit_x, np.polyval(c, fit_x), '-', color=color, lw=0.9)

        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(f'Experimental {unit}')
        ax.set_ylabel(f'Predicted {unit}')

        txt = f'$R^2$ = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}'
        ax.text(0.04, 0.96, txt, transform=ax.transAxes, fontsize=5.5,
                va='top', bbox=dict(boxstyle='round,pad=0.3', fc='white',
                                    ec='#CCCCCC', alpha=0.92, lw=0.4))
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f'{task}_regression_scatter.png'), dpi=600)
        plt.close(fig)
    print(f'  Saved individual scatter → {out_dir}')


def save_individual_residual(main_data, out_dir):
    for task in TASK_ORDER_MAIN:
        true, pred = main_data[task]
        res = pred - true
        name, unit = TASK_LABELS[task]
        color = COLORS[task]

        fig, ax = plt.subplots(figsize=(FIG_SINGLE * 0.85, FIG_SINGLE * 0.65))
        nbins = min(20, max(10, len(res) // 6))
        ax.hist(res, bins=nbins, density=True, color=color, alpha=0.3,
                edgecolor='white', linewidth=0.3)
        if len(res) > 10:
            kde = gaussian_kde(res)
            xr = np.linspace(res.min() - 0.5, res.max() + 0.5, 200)
            ax.plot(xr, kde(xr), '-', color=color, lw=1.0)
        ax.axvline(0, color='#999999', ls='--', lw=0.5)

        mu, sigma = res.mean(), res.std()
        txt = f'$\\mu$ = {mu:.3f}\n$\\sigma$ = {sigma:.3f}\n$n$ = {len(res)}'
        ax.text(0.96, 0.96, txt, transform=ax.transAxes, fontsize=5.5,
                va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.3', fc='white',
                          ec='#CCCCCC', alpha=0.92, lw=0.4))
        ax.set_xlabel(f'Residual ({unit})')
        ax.set_ylabel('Density')
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f'{task}_residual_dist.png'), dpi=600)
        plt.close(fig)
    print(f'  Saved individual residual → {out_dir}')


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    os.makedirs('figures', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)

    print('Loading data...')
    main_data = load_main_predictions()
    cross_data = load_cross_predictions()
    cross_metrics = load_cross_metrics()

    print('\nGenerating figures (Nature style, 600 DPI)...\n')

    fig_main_results(main_data, 'figures/Fig_main_results.png')
    fig_cross_dataset(cross_data, cross_metrics, 'figures/Fig_cross_dataset.png')
    fig_attention('figures/Fig_attention.png')

    save_individual_scatter(main_data, 'results/figures')
    save_individual_residual(main_data, 'results/figures')

    print('\nAll figures saved to figures/ and results/figures/')
