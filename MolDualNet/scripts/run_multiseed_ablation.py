#!/usr/bin/env python3
"""
Multi-seed ablation suite for MolDualNet.
Runs key ablation configurations across 5 seeds and aggregates results
with mean±std and paired t-test statistics.

Usage:
    python scripts/run_multiseed_ablation.py --config config_107k.yaml --epochs 30 --device cuda
"""

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np


SEEDS = [42, 123, 456, 789, 1024]

# Key configurations for multi-seed ablation
VARIANTS = [
    ('full',              []),
    ('no_cross_attention', ['--no_cross_attention']),
    ('no_expert',          ['--no_expert']),
    ('no_3d',              ['--no_3d']),
    ('gnn_only',           ['--no_cross_attention', '--no_expert', '--no_3d']),
]


def run_single(config, variant_name, flags, seed, epochs, device, root_dir):
    """Run a single ablation configuration with a given seed."""
    results_dir = os.path.join(root_dir, f'{variant_name}_seed{seed}')
    ckpt_dir = os.path.join(results_dir, 'checkpoints')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Resume: skip if already completed successfully
    summary_path = os.path.join(results_dir, 'training_summary.json')
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r') as f:
                existing = json.load(f)
            if existing.get('metrics'):
                print(f'  [{variant_name}] seed={seed} — already done, skipping.', flush=True)
                return existing
        except Exception:
            pass  # corrupt file, re-run

    cmd = [
        sys.executable, 'scripts/ablation_train_eval.py',
        '--config', config,
        '--device', device,
        '--seed', str(seed),
        '--results_dir', results_dir,
        '--checkpoint_dir', ckpt_dir,
        '--epochs', str(epochs),
    ] + flags

    print(f'  [{variant_name}] seed={seed} ...', flush=True)
    try:
        subprocess.check_call(cmd)
        # Read training summary
        summary_path = os.path.join(results_dir, 'training_summary.json')
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                return json.load(f)
        else:
            print(f'  WARNING: no training_summary.json for {variant_name} seed={seed}')
            return None
    except subprocess.CalledProcessError as e:
        print(f'  ERROR: {variant_name} seed={seed} failed: {e}')
        return None


def aggregate_results(all_results):
    """Aggregate per-seed results into mean±std with statistical tests."""
    from scipy import stats

    tasks = ['ESOL_logS', 'Lipophilicity_logD', 'FreeSolv_hydration', 'BACE_pIC50']
    metrics = ['R2', 'RMSE', 'MAE']

    summary = {}
    for variant, seed_results in all_results.items():
        valid = [r for r in seed_results if r is not None]
        if not valid:
            continue

        variant_summary = {'n_seeds': len(valid)}
        for task in tasks:
            task_data = {}
            for metric in metrics:
                values = [r['metrics'].get(task, {}).get(metric, float('nan'))
                          for r in valid]
                values = [v for v in values if not np.isnan(v)]
                if values:
                    task_data[metric] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                        'values': values,
                    }
            variant_summary[task] = task_data
        summary[variant] = variant_summary

    # Paired t-tests: each variant vs 'full'
    if 'full' in summary:
        full_r2 = {}
        for task in tasks:
            full_r2[task] = summary['full'].get(task, {}).get('R2', {}).get('values', [])

        for variant in summary:
            if variant == 'full':
                continue
            for task in tasks:
                var_vals = summary[variant].get(task, {}).get('R2', {}).get('values', [])
                full_vals = full_r2.get(task, [])
                n = min(len(var_vals), len(full_vals))
                if n >= 3:
                    t_stat, p_val = stats.ttest_rel(full_vals[:n], var_vals[:n])
                    summary[variant][task]['R2']['ttest_p'] = float(p_val)
                    summary[variant][task]['R2']['ttest_t'] = float(t_stat)

    return summary


def print_latex_table(summary):
    """Generate LaTeX table for the paper."""
    tasks = ['ESOL_logS', 'Lipophilicity_logD', 'FreeSolv_hydration', 'BACE_pIC50']
    task_short = ['ESOL', 'Lipo', 'FreeSolv', 'BACE']

    print('\n% LaTeX table for multi-seed ablation')
    print('\\begin{table*}[!htbp]')
    print('    \\centering')
    print('    \\caption{Ablation study with 5 random seeds: $R^2$ (mean $\\pm$ std). '
          '$p$-values from paired $t$-tests vs.\\ Full Model.}')
    print('    \\label{tab:ablation_multiseed}')
    cols = 'l' + 'c' * len(tasks) + 'c'
    print(f'    \\begin{{tabular}}{{@{{}}{cols}@{{}}}}')
    print('        \\toprule')
    header = ' & '.join(['\\textbf{Configuration}'] +
                         [f'\\textbf{{{t}}}' for t in task_short] +
                         ['\\textbf{Avg $R^2$}'])
    print(f'        {header} \\\\')
    print('        \\midrule')

    variant_order = ['full', 'no_cross_attention', 'no_expert', 'no_3d', 'gnn_only']
    variant_names = {
        'full': 'Full Model',
        'no_cross_attention': '\\quad $-$ Cross-Attention',
        'no_expert': '\\quad $-$ Expert Features',
        'no_3d': '\\quad $-$ 3D Geometry',
        'gnn_only': 'GNN Only',
    }

    for var in variant_order:
        if var not in summary:
            continue
        cells = [variant_names.get(var, var)]
        avg_means = []
        for task in tasks:
            r2 = summary[var].get(task, {}).get('R2', {})
            m = r2.get('mean', float('nan'))
            s = r2.get('std', 0)
            avg_means.append(m)
            p = r2.get('ttest_p', None)
            star = ''
            if p is not None and p < 0.05:
                star = '*'
            if p is not None and p < 0.01:
                star = '**'
            cells.append(f'{m:.3f}$\\pm${s:.3f}{star}')
        avg_m = np.nanmean(avg_means)
        cells.append(f'{avg_m:.3f}')
        print(f'        {" & ".join(cells)} \\\\')

    print('        \\bottomrule')
    print('    \\end{tabular}')
    print('\\end{table*}')


def main():
    parser = argparse.ArgumentParser(description='Multi-seed ablation suite')
    parser.add_argument('--config', type=str, default='config_107k.yaml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--root', type=str, default='results_multiseed_ablation')
    args = parser.parse_args()

    os.makedirs(args.root, exist_ok=True)

    all_results = defaultdict(list)

    for variant_name, flags in VARIANTS:
        print(f'\n=== {variant_name} ===', flush=True)
        for seed in SEEDS:
            result = run_single(
                args.config, variant_name, flags, seed,
                args.epochs, args.device, args.root
            )
            all_results[variant_name].append(result)

    # Aggregate
    summary = aggregate_results(all_results)

    # Save
    output_path = os.path.join(args.root, 'multiseed_ablation_summary.json')
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        raise TypeError(f'{type(obj)} not serializable')

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, default=convert)
    print(f'\nSaved summary to {output_path}')

    # Print LaTeX table
    print_latex_table(summary)

    print('\nDone!')


if __name__ == '__main__':
    main()
