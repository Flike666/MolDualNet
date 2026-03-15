#!/usr/bin/env python3
"""
FreeSolv regularization experiment for cross-attention overfitting analysis.

Compares three cross-attention configurations on FreeSolv (single-task):
  1. full_cross_attn  - default (dropout=0.15, modality_dropout=0.20, layers=2)
  2. no_cross_attn    - cross-attention disabled
  3. reg_cross_attn   - regularized (dropout=0.30, modality_dropout=0.30, layers=1)

Addresses: Major Comment 3 – cross-attention hurts FreeSolv.

Usage:
    python scripts/run_freesolv_regularization.py \
        --config configs/config_107k.yaml \
        --epochs 60 --device cuda \
        --root results_freesolv_reg
"""

import argparse
import copy
import json
import os
import subprocess
import sys

import yaml
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_config


VARIANTS = ['no_cross_attn', 'full_cross_attn', 'reg_cross_attn']

VARIANT_LABEL = {
    'no_cross_attn':   'No Cross-Attn',
    'full_cross_attn': 'Full Cross-Attn',
    'reg_cross_attn':  'Regularized Cross-Attn',
}


def make_freesolv_only_config(base_config, variant):
    """Derive a FreeSolv-only config for one variant."""
    cfg = copy.deepcopy(base_config)

    # ── Single task ──────────────────────────────────────────────
    cfg['tasks'] = {
        'FreeSolv_hydration': {'task_type': 'regression', 'loss_weight': 1.0}
    }

    # ── Cross-attention settings ──────────────────────────────────
    fusion = cfg.setdefault('model', {}).setdefault('fusion', {})

    if variant == 'no_cross_attn':
        fusion['use_cross_attention'] = False

    elif variant == 'full_cross_attn':
        fusion['use_cross_attention'] = True
        fusion['cross_attention_layers'] = 2
        fusion['cross_attention_heads'] = 8
        # keep default dropouts from base config
        cfg['model']['fusion']['dropout'] = 0.15
        cfg['model'].setdefault('fusion', {})['modality_dropout_p'] = 0.20

    elif variant == 'reg_cross_attn':
        fusion['use_cross_attention'] = True
        fusion['cross_attention_layers'] = 1        # fewer layers
        fusion['cross_attention_heads'] = 8
        fusion['dropout'] = 0.30                    # higher dropout
        fusion['modality_dropout_p'] = 0.30         # more modality dropout

    return cfg


def main():
    parser = argparse.ArgumentParser(description='FreeSolv cross-attention regularization')
    parser.add_argument('--config', type=str, default='configs/config_107k.yaml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--root', type=str, default='results_freesolv_reg')
    args = parser.parse_args()

    os.makedirs(args.root, exist_ok=True)
    base_config = load_config(args.config)

    results = {}

    for variant in VARIANTS:
        variant_dir = os.path.join(args.root, variant)
        ckpt_dir = os.path.join(variant_dir, 'checkpoints')
        summary_path = os.path.join(variant_dir, 'training_summary.json')

        # Skip completed runs
        if os.path.exists(summary_path):
            print(f'\n[SKIP] {variant} already has training_summary.json')
            with open(summary_path) as f:
                results[variant] = json.load(f)
            continue

        print(f'\n{"=" * 60}')
        print(f'Running: {variant}')
        print(f'{"=" * 60}', flush=True)

        os.makedirs(variant_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)

        cfg = make_freesolv_only_config(base_config, variant)
        cfg['data']['random_seed'] = args.seed
        cfg['training']['epochs'] = args.epochs
        cfg['save']['results_dir'] = variant_dir
        cfg['save']['checkpoint_dir'] = ckpt_dir

        config_path = os.path.join(variant_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)

        flags = []
        if variant == 'no_cross_attn':
            flags = ['--no_cross_attention']

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
            if os.path.exists(summary_path):
                with open(summary_path) as f:
                    results[variant] = json.load(f)
        except subprocess.CalledProcessError as e:
            print(f'  ERROR: {variant} failed: {e}')

    # ── Summary ───────────────────────────────────────────────────
    print('\n\n' + '=' * 60)
    print('FreeSolv Regularization Results')
    print('=' * 60)
    print(f'{"Variant":<25}  {"R2":>8}  {"RMSE":>8}  {"MAE":>8}')
    print('-' * 55)

    for variant in VARIANTS:
        if variant not in results:
            print(f'{VARIANT_LABEL[variant]:<25}  --')
            continue
        m = results[variant].get('metrics', {}).get('FreeSolv_hydration', {})
        r2 = m.get('R2', float('nan'))
        rmse = m.get('RMSE', float('nan'))
        mae = m.get('MAE', float('nan'))
        print(f'{VARIANT_LABEL[variant]:<25}  {r2:>8.3f}  {rmse:>8.3f}  {mae:>8.3f}')

    # Save
    output_path = os.path.join(args.root, 'freesolv_reg_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved to {output_path}')

    # LaTeX one-liner
    print('\n--- For paper ---')
    for variant in VARIANTS:
        if variant not in results:
            continue
        m = results[variant].get('metrics', {}).get('FreeSolv_hydration', {})
        r2 = m.get('R2', float('nan'))
        rmse = m.get('RMSE', float('nan'))
        print(f'{VARIANT_LABEL[variant]}: R²={r2:.3f}, RMSE={rmse:.3f}')


if __name__ == '__main__':
    main()
