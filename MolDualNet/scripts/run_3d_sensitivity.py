#!/usr/bin/env python3
"""
3D Feature Sensitivity Analysis for MolDualNet.
Tests different RBF center counts (K=4 vs K=8) and conformer strategies
(single vs 3-conformer ensemble).

Usage:
    python scripts/run_3d_sensitivity.py --config config_107k.yaml --epochs 30 --device cuda
"""

import argparse
import json
import os
import subprocess
import sys
import copy

import yaml
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config


def modify_config_for_variant(config, variant):
    """Modify config for a specific 3D sensitivity variant."""
    cfg = copy.deepcopy(config)
    geo = cfg.setdefault('model', {}).setdefault('geometry_3d', {})

    if variant == 'k4_single':
        # Default: K=4, single conformer
        geo['distance_rbf_centers'] = [0.5, 1.0, 1.5, 2.0]
        geo['enabled'] = True

    elif variant == 'k8_single':
        # K=8, single conformer
        geo['distance_rbf_centers'] = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        geo['enabled'] = True
        # Adjust edge_input_dim: base 12 + 1(dist) + 8(rbf) + 3(bin) = 24
        cfg['model']['gnn']['edge_input_dim'] = 24

    elif variant == 'k4_3conf':
        # K=4, 3-conformer (use average 3D coords)
        geo['distance_rbf_centers'] = [0.5, 1.0, 1.5, 2.0]
        geo['enabled'] = True
        geo['num_conformers'] = 3

    elif variant == 'no_3d':
        # Baseline: no 3D features
        geo['enabled'] = False
        cfg['model']['gnn']['edge_input_dim'] = 12

    return cfg


VARIANTS = ['no_3d', 'k4_single', 'k8_single', 'k4_3conf']
VARIANT_LABELS = {
    'no_3d':       'No 3D',
    'k4_single':   '$K_{\\text{rbf}}=4$, 1 conf.',
    'k8_single':   '$K_{\\text{rbf}}=8$, 1 conf.',
    'k4_3conf':    '$K_{\\text{rbf}}=4$, 3 conf.',
}


def main():
    parser = argparse.ArgumentParser(description='3D sensitivity analysis')
    parser.add_argument('--config', type=str, default='config_107k.yaml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--root', type=str, default='results_3d_sensitivity')
    args = parser.parse_args()

    os.makedirs(args.root, exist_ok=True)
    base_config = load_config(args.config)

    results = {}

    for variant in VARIANTS:
        print(f'\n=== {variant} ===', flush=True)
        cfg = modify_config_for_variant(base_config, variant)

        # Save modified config
        variant_dir = os.path.join(args.root, variant)
        ckpt_dir = os.path.join(variant_dir, 'checkpoints')
        os.makedirs(variant_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)

        config_path = os.path.join(variant_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)

        # Build flags for ablation script
        flags = []
        if variant == 'no_3d':
            flags = ['--no_3d']

        cmd = [
            sys.executable, 'scripts/ablation_train_eval.py',
            '--config', config_path,
            '--device', args.device,
            '--seed', str(args.seed),
            '--results_dir', variant_dir,
            '--checkpoint_dir', ckpt_dir,
            '--epochs', str(args.epochs),
        ] + flags

        try:
            subprocess.check_call(cmd)
            summary_path = os.path.join(variant_dir, 'training_summary.json')
            if os.path.exists(summary_path):
                with open(summary_path, 'r') as f:
                    results[variant] = json.load(f)
        except subprocess.CalledProcessError as e:
            print(f'  ERROR: {variant} failed: {e}')

    # Print summary table
    tasks = ['ESOL_logS', 'Lipophilicity_logD', 'FreeSolv_hydration', 'BACE_pIC50']
    task_short = ['ESOL', 'Lipo', 'FreeSolv', 'BACE']

    print('\n\n=== 3D Sensitivity Results ===')
    print(f'{"Variant":<20}', end='')
    for t in task_short:
        print(f'  {t:>10}', end='')
    print(f'  {"Avg R2":>10}')
    print('-' * 70)

    for variant in VARIANTS:
        if variant not in results:
            continue
        metrics = results[variant].get('metrics', {})
        print(f'{VARIANT_LABELS.get(variant, variant):<20}', end='')
        r2s = []
        for task in tasks:
            r2 = metrics.get(task, {}).get('R2', float('nan'))
            r2s.append(r2)
            print(f'  {r2:>10.3f}', end='')
        print(f'  {np.nanmean(r2s):>10.3f}')

    # Save
    output_path = os.path.join(args.root, '3d_sensitivity_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved to {output_path}')


if __name__ == '__main__':
    main()
