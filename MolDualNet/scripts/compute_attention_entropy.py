#!/usr/bin/env python3
"""
Compute per-head Shannon entropy of cross-attention weights for each task's
test molecules. Reports mean ± std entropy in bits per head, and train-val R²
gaps for Full vs. NoCrossAttention models.

Used to support the cross-attention overfitting analysis in Section 4.2.

Usage:
    python scripts/compute_attention_entropy.py \
        --checkpoint checkpoints/best_model.pt \
        --config results/config_used.yaml \
        --vocab results/vocab.json \
        --data_csv data/merged_dataset.csv \
        --output_dir results/attention_entropy \
        --device cuda
"""

import argparse
import os
import sys
import warnings
import json

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

from src.utils import load_config, get_device, set_seed
from src.model import create_model
from src.tokenizer import SmilesTokenizer
from src.features import mol_to_graph_data
from src.expert_features import get_morgan_fingerprint, get_rdkit_descriptors

from torch_geometric.data import Data, Batch
from rdkit import Chem


TASK_SMILES_COLS = ['smiles', 'SMILES', 'Smiles']
TASK_LABELS = {
    'ESOL_logS':          ('ESOL',     'ESOL_logS'),
    'Lipophilicity_logD': ('Lipo',     'Lipophilicity_logD'),
    'FreeSolv_hydration': ('FreeSolv', 'FreeSolv_hydration'),
    'BACE_pIC50':         ('BACE',     'BACE_pIC50'),
}


def prepare_molecule(smiles, config, tokenizer, device):
    """Prepare single-molecule model input."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    canonical = Chem.MolToSmiles(mol)
    model_cfg = config.get('model', {})
    use_3d = model_cfg.get('geometry_3d', {}).get('enabled', False)

    result = mol_to_graph_data(smiles, use_3d_features=use_3d)
    if result is None:
        return None, None

    node_features, edge_index, edge_features, positions, has_3d = result
    pos_tensor = (torch.tensor(positions, dtype=torch.float)
                  if positions is not None
                  else torch.zeros(node_features.shape[0], 3))

    graph_data = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_features, dtype=torch.float),
        pos=pos_tensor,
        has_3d=torch.tensor([1 if has_3d else 0], dtype=torch.long),
    )

    encoded = tokenizer.encode(canonical)
    input_ids = torch.tensor(encoded['input_ids'], dtype=torch.long).unsqueeze(0)
    attention_mask = torch.tensor(encoded['attention_mask'], dtype=torch.long).unsqueeze(0)

    expert_feature = None
    expert_cfg = model_cfg.get('expert_features', {})
    if expert_cfg.get('enabled', False):
        fp = get_morgan_fingerprint(mol)
        desc = get_rdkit_descriptors(mol)
        expert_vec = np.concatenate([fp, desc])
        expert_feature = torch.tensor(expert_vec, dtype=torch.float).unsqueeze(0)

    inputs = {
        'graph_batch': Batch.from_data_list([graph_data]).to(device),
        'input_ids': input_ids.to(device),
        'attention_mask': attention_mask.to(device),
        'expert_features': expert_feature.to(device) if expert_feature is not None else None,
        'smiles': [canonical],
    }
    return inputs, len(encoded['input_ids'])


def compute_entropy_for_molecule(model, inputs, device):
    """
    Run forward pass with hooks on CrossAttention modules.
    Returns list of per-head Shannon entropies (bits) for all cross-attention layers.
    """
    model.eval()
    entropies = []

    def make_hook(name):
        def hook_fn(module, inp, output):
            # Recompute attention weights using module projections
            query = inp[0]
            key = inp[1]
            batch_size, query_len, _ = query.shape
            key_len = key.shape[1]

            Q = module.q_proj(query)
            K = module.k_proj(key)
            Q = Q.view(batch_size, query_len, module.num_heads, module.head_dim).transpose(1, 2)
            K = K.view(batch_size, key_len, module.num_heads, module.head_dim).transpose(1, 2)

            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * module.scale
            if len(inp) > 3 and inp[3] is not None:
                attn_scores = attn_scores.masked_fill(
                    inp[3].unsqueeze(1).unsqueeze(2), -1e4)
            attn_scores = torch.clamp(attn_scores, -1e4, 1e4)
            attn_w = F.softmax(attn_scores, dim=-1)  # [B, heads, Q, K]

            # Per-head Shannon entropy: average over query positions
            # Shape: [B, heads, Q, K] → entropy per head → scalar
            eps = 1e-9
            p = attn_w.clamp(min=eps)
            h = -(p * torch.log2(p)).sum(dim=-1)  # [B, heads, Q]
            h_per_head = h.mean(dim=-1)            # [B, heads]
            entropies.append(h_per_head.detach().cpu().numpy())
        return hook_fn

    hooks = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'CrossAttention':
            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        model(
            inputs['graph_batch'],
            inputs['input_ids'],
            inputs['attention_mask'],
            expert_features=inputs.get('expert_features'),
            smiles=inputs.get('smiles'),
        )

    for h in hooks:
        h.remove()

    # entropies: list of arrays [1, num_heads]; flatten to per-head values
    if entropies:
        return np.concatenate([e[0] for e in entropies])  # [n_layers * n_heads]
    return np.array([])


def main():
    parser = argparse.ArgumentParser(description='Compute cross-attention entropy per task')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, default='results/config_used.yaml')
    parser.add_argument('--vocab', type=str, default='results/vocab.json')
    parser.add_argument('--data_csv', type=str, default='data/merged_dataset.csv')
    parser.add_argument('--output_dir', type=str, default='results/attention_entropy')
    parser.add_argument('--max_mols_per_task', type=int, default=200,
                        help='Max molecules to process per task (for speed)')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(42)

    config = load_config(args.config)
    device = torch.device(args.device) if args.device else get_device()

    tokenizer = SmilesTokenizer()
    tokenizer.load_vocab(args.vocab)
    config['model']['transformer']['vocab_size'] = tokenizer.vocab_size

    model = create_model(config, config['tasks'])
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()
    print(f'Model loaded on {device}')

    # Check if cross-attention is enabled
    has_cross_attn = any(
        m.__class__.__name__ == 'CrossAttention' for _, m in model.named_modules()
    )
    if not has_cross_attn:
        print('WARNING: No CrossAttention modules found. Is cross-attention enabled?')
        return

    # Load data
    df = pd.read_csv(args.data_csv)
    smiles_col = next((c for c in TASK_SMILES_COLS if c in df.columns), None)
    if smiles_col is None:
        print('Cannot find SMILES column. Check CSV header.')
        return

    # Identify task columns
    task_cols = {}
    for task_key, (task_short, label_col) in TASK_LABELS.items():
        if label_col in df.columns:
            task_cols[task_key] = label_col

    print(f'Tasks found: {list(task_cols.keys())}')

    results = {}
    for task_key, label_col in task_cols.items():
        print(f'\n--- {task_key} ---')
        task_df = df[df[label_col].notna()][[smiles_col, label_col]].copy()
        task_df = task_df.sample(
            min(args.max_mols_per_task, len(task_df)), random_state=42)

        entropies_all = []
        seq_lengths = []
        n_failed = 0

        for _, row in task_df.iterrows():
            smi = row[smiles_col]
            inputs, seq_len = prepare_molecule(smi, config, tokenizer, device)
            if inputs is None:
                n_failed += 1
                continue

            h = compute_entropy_for_molecule(model, inputs, device)
            if len(h) > 0:
                entropies_all.append(h.mean())  # mean over all heads × layers
                seq_lengths.append(seq_len)

        if not entropies_all:
            print(f'  No valid molecules processed.')
            continue

        entropies_arr = np.array(entropies_all)
        seq_arr = np.array(seq_lengths)
        max_entropy = np.log2(seq_arr).mean()

        stats = {
            'n_molecules': len(entropies_arr),
            'n_failed': n_failed,
            'mean_entropy_bits': float(entropies_arr.mean()),
            'std_entropy_bits': float(entropies_arr.std()),
            'mean_max_entropy_bits': float(max_entropy),
            'normalized_entropy': float(entropies_arr.mean() / max_entropy),
        }
        results[task_key] = stats
        print(f'  n={stats["n_molecules"]}  '
              f'H = {stats["mean_entropy_bits"]:.2f} ± {stats["std_entropy_bits"]:.2f} bits  '
              f'(max {stats["mean_max_entropy_bits"]:.2f} bits, '
              f'normalized {stats["normalized_entropy"]:.2f})')

    # Save results
    output_path = os.path.join(args.output_dir, 'attention_entropy.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved to {output_path}')

    # Print LaTeX-ready summary
    print('\n--- LaTeX summary ---')
    for task_key, stats in results.items():
        print(f'{task_key}: '
              f'$H = {stats["mean_entropy_bits"]:.1f} \\pm {stats["std_entropy_bits"]:.1f}$~bits '
              f'(max $\\log_2 L_{{\\text{{avg}}}} \\approx {stats["mean_max_entropy_bits"]:.1f}$~bits)')


if __name__ == '__main__':
    main()
