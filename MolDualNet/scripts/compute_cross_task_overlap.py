#!/usr/bin/env python3
"""
Cross-task molecule overlap analysis for MolDualNet.

Computes scaffold-level (generic Murcko scaffold) and molecule-level (InChIKey)
overlap between the test sets of the four regression tasks under two protocols:
  Protocol 1 - random split (train_ratio=0.8, val=0.1, test=0.1, seed=42)
  Protocol 2 - global scaffold split (molecules sharing a scaffold stay together)

Usage:
    python scripts/compute_cross_task_overlap.py \
        --config configs/config_107k.yaml \
        --output results/cross_task_overlap.json
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from collections import defaultdict

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split


TASKS = ['ESOL_logS', 'Lipophilicity_logD', 'FreeSolv_hydration', 'BACE_pIC50']
TASK_SHORT = {'ESOL_logS': 'ESOL', 'Lipophilicity_logD': 'Lipo',
              'FreeSolv_hydration': 'FreeSolv', 'BACE_pIC50': 'BACE'}


def get_generic_scaffold(smiles):
    """Return generic Murcko scaffold SMILES (no atom labels)."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        generic = MurckoScaffold.MakeScaffoldGeneric(scaffold)
        return Chem.MolToSmiles(generic, isomericSmiles=False)
    except Exception:
        return None


def get_inchikey(smiles):
    """Return InChIKey for molecule-level dedup."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        from rdkit.Chem.inchi import MolToInchiKey
        return MolToInchiKey(mol)
    except Exception:
        return None


def scaffold_split(smiles_list, seed=42, test_ratio=0.1, val_ratio=0.1):
    """Scaffold-based split: molecules sharing a scaffold go to the same split."""
    scaffolds = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        s = get_generic_scaffold(smi)
        scaffolds[s].append(i)

    scaffold_groups = list(scaffolds.values())
    np.random.seed(seed)
    np.random.shuffle(scaffold_groups)

    n_total = len(smiles_list)
    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)

    test_idx, val_idx, train_idx = [], [], []
    for group in scaffold_groups:
        if len(test_idx) < n_test:
            test_idx.extend(group)
        elif len(val_idx) < n_val:
            val_idx.extend(group)
        else:
            train_idx.extend(group)

    return set(train_idx), set(val_idx), set(test_idx)


def compute_overlap(set_a, set_b, label_a, label_b):
    """Compute overlap statistics between two sets."""
    inter = set_a & set_b
    union = set_a | set_b
    n_a, n_b = len(set_a), len(set_b)
    n_inter = len(inter)
    jaccard = n_inter / len(union) if union else 0.0
    overlap_a = n_inter / n_a if n_a else 0.0
    overlap_b = n_inter / n_b if n_b else 0.0
    return {
        'pair': f'{label_a} vs {label_b}',
        'n_a': n_a, 'n_b': n_b,
        'n_intersection': n_inter,
        'jaccard': round(jaccard, 4),
        'overlap_rate_a': round(overlap_a, 4),
        'overlap_rate_b': round(overlap_b, 4),
        'overlap_pct_avg': round((overlap_a + overlap_b) / 2 * 100, 2),
    }


def run_protocol(df, smiles_col, protocol, seed=42):
    """Run overlap analysis for one split protocol."""
    results = {}

    # Build per-task test sets (scaffold and inchikey)
    task_test_scaffolds = {}
    task_test_inchikeys = {}

    for task in TASKS:
        if task not in df.columns:
            continue
        task_df = df[df[task].notna()].copy()
        smiles = task_df[smiles_col].tolist()

        if protocol == 'random':
            idx = list(range(len(smiles)))
            _, test_idx = train_test_split(idx, test_size=0.1, random_state=seed)
            test_smiles = [smiles[i] for i in test_idx]
        elif protocol == 'scaffold':
            _, _, test_idx = scaffold_split(smiles, seed=seed)
            test_smiles = [smiles[i] for i in sorted(test_idx)]

        task_test_scaffolds[task] = set(
            s for s in (get_generic_scaffold(sm) for sm in test_smiles)
            if s is not None
        )
        task_test_inchikeys[task] = set(
            k for k in (get_inchikey(sm) for sm in test_smiles)
            if k is not None
        )

    # Pairwise overlaps
    task_list = [t for t in TASKS if t in task_test_scaffolds]
    scaffold_overlaps = []
    mol_overlaps = []

    for i in range(len(task_list)):
        for j in range(i + 1, len(task_list)):
            ta, tb = task_list[i], task_list[j]
            sa = TASK_SHORT[ta]
            sb = TASK_SHORT[tb]
            scaffold_overlaps.append(
                compute_overlap(task_test_scaffolds[ta], task_test_scaffolds[tb], sa, sb))
            mol_overlaps.append(
                compute_overlap(task_test_inchikeys[ta], task_test_inchikeys[tb], sa, sb))

    # Global average
    avg_scaffold = np.mean([o['overlap_pct_avg'] for o in scaffold_overlaps])
    avg_mol = np.mean([o['overlap_pct_avg'] for o in mol_overlaps])

    results['scaffold_overlaps'] = scaffold_overlaps
    results['molecule_overlaps'] = mol_overlaps
    results['avg_scaffold_overlap_pct'] = round(float(avg_scaffold), 2)
    results['avg_molecule_overlap_pct'] = round(float(avg_mol), 2)

    return results


def main():
    parser = argparse.ArgumentParser(description='Cross-task overlap analysis')
    parser.add_argument('--data_csv', type=str, default='data/merged_dataset.csv')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='results/cross_task_overlap.json')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    df = pd.read_csv(args.data_csv)
    smiles_col = next((c for c in ['smiles', 'SMILES', 'Smiles'] if c in df.columns), None)
    if smiles_col is None:
        print('Cannot find SMILES column')
        return

    print(f'Dataset: {len(df)} molecules, {smiles_col} column')
    print(f'Tasks: {[t for t in TASKS if t in df.columns]}')

    all_results = {}

    for protocol in ['random', 'scaffold']:
        print(f'\n=== Protocol: {protocol} split (seed={args.seed}) ===')
        res = run_protocol(df, smiles_col, protocol, seed=args.seed)
        all_results[protocol] = res

        print(f'\nScaffold-level overlap:')
        for o in res['scaffold_overlaps']:
            print(f"  {o['pair']:30s}  Jaccard={o['jaccard']:.3f}  "
                  f"avg overlap={o['overlap_pct_avg']:.1f}%  "
                  f"(n_inter={o['n_intersection']})")
        print(f"  Average scaffold overlap: {res['avg_scaffold_overlap_pct']:.1f}%")

        print(f'\nMolecule-level overlap (InChIKey):')
        for o in res['molecule_overlaps']:
            print(f"  {o['pair']:30s}  Jaccard={o['jaccard']:.3f}  "
                  f"avg overlap={o['overlap_pct_avg']:.1f}%  "
                  f"(n_inter={o['n_intersection']})")
        print(f"  Average molecule overlap: {res['avg_molecule_overlap_pct']:.1f}%")

    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nSaved to {args.output}')

    # Summary for paper
    print('\n=== Summary for paper ===')
    r = all_results['random']
    s = all_results['scaffold']
    print(f"Protocol 1 (random): molecule-level {r['avg_molecule_overlap_pct']:.1f}%, "
          f"scaffold-level {r['avg_scaffold_overlap_pct']:.1f}%")
    print(f"Protocol 2 (scaffold): molecule-level {s['avg_molecule_overlap_pct']:.1f}%, "
          f"scaffold-level {s['avg_scaffold_overlap_pct']:.1f}%")


if __name__ == '__main__':
    main()
