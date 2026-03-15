#!/usr/bin/env python3
"""
Visualize cross-attention patterns between GNN nodes (atoms) and
Transformer tokens (SMILES characters) for selected molecules.

Produces node-token heatmaps showing which atoms attend to which
SMILES tokens (and vice versa), supporting the mechanism analysis
in the paper.

Usage:
    python scripts/visualize_cross_attention.py \
        --checkpoint checkpoints/best_model.pt \
        --config results/config_used.yaml \
        --vocab results/vocab.json \
        --smiles "c1ccc(CC(=O)O)cc1" "CC(=O)Oc1ccccc1C(=O)O" "CCO" \
        --output_dir results/attention_viz
"""

import argparse
import os
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, get_device, set_seed
from src.model import create_model
from src.tokenizer import SmilesTokenizer
from src.features import mol_to_graph_data
from src.expert_features import get_morgan_fingerprint, get_rdkit_descriptors

from torch_geometric.data import Data, Batch
from rdkit import Chem

warnings.filterwarnings('ignore')


def prepare_molecule_batch(smiles, config, tokenizer, device):
    """Prepare a single molecule for model input, matching the model's forward signature."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None

    canonical = Chem.MolToSmiles(mol)
    atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]

    model_cfg = config.get('model', {})
    use_3d = model_cfg.get('geometry_3d', {}).get('enabled', False)

    result = mol_to_graph_data(smiles, use_3d_features=use_3d)
    if result is None:
        return None, None, None

    node_features, edge_index, edge_features, positions, has_3d = result

    pos_tensor = torch.tensor(positions, dtype=torch.float) if positions is not None else torch.zeros(node_features.shape[0], 3)
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

    # Expert features
    expert_feature = None
    expert_cfg = model_cfg.get('expert_features', {})
    if expert_cfg.get('enabled', False):
        if mol is not None:
            fp = get_morgan_fingerprint(mol)
            desc = get_rdkit_descriptors(mol)
            expert_vec = np.concatenate([fp, desc])
            expert_feature = torch.tensor(expert_vec, dtype=torch.float).unsqueeze(0)

    graph_batch = Batch.from_data_list([graph_data]).to(device)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    if expert_feature is not None:
        expert_feature = expert_feature.to(device)

    # Token labels for visualization
    token_labels = [tokenizer.idx2token.get(tid, f'[{tid}]') for tid in encoded['input_ids']]

    model_inputs = {
        'graph_batch': graph_batch,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'expert_features': expert_feature,
        'smiles': [canonical],
    }

    return model_inputs, atom_symbols, token_labels


def extract_attention_weights(model, model_inputs, device):
    """
    Forward pass through the model, hooking into cross-attention layers
    to extract attention weight matrices.
    """
    model.eval()
    attention_maps = {}

    def make_hook(name):
        def hook_fn(module, input, output):
            # Recompute attention weights (the module doesn't return them)
            query, key, value = input[0], input[1], input[2]
            key_padding_mask = input[3] if len(input) > 3 else None

            batch_size, query_len, _ = query.shape
            key_len = key.shape[1]

            Q = module.q_proj(query)
            K = module.k_proj(key)

            Q = Q.view(batch_size, query_len, module.num_heads, module.head_dim).transpose(1, 2)
            K = K.view(batch_size, key_len, module.num_heads, module.head_dim).transpose(1, 2)

            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * module.scale

            if key_padding_mask is not None:
                attn_scores = attn_scores.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2), -1e4
                )

            attn_scores = torch.clamp(attn_scores, min=-1e4, max=1e4)
            attn_weights = F.softmax(attn_scores, dim=-1)

            attention_maps[name] = attn_weights.detach().cpu().numpy()

        return hook_fn

    # Register hooks on CrossAttention modules
    hooks = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'CrossAttention':
            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)

    with torch.no_grad():
        model(
            model_inputs['graph_batch'],
            model_inputs['input_ids'],
            model_inputs['attention_mask'],
            expert_features=model_inputs.get('expert_features'),
            smiles=model_inputs.get('smiles'),
        )

    for h in hooks:
        h.remove()

    return attention_maps


def plot_attention_heatmap(attn_matrix, row_labels, col_labels, title,
                           output_path, direction='gnn_to_trans'):
    """Plot a heatmap of attention weights."""
    fig_w = min(20, max(6, len(col_labels) * 0.45 + 2))
    fig_h = min(12, max(3, len(row_labels) * 0.35 + 1.5))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Normalize per row for better visibility
    row_sums = attn_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    attn_norm = attn_matrix / row_sums

    im = ax.imshow(attn_norm, cmap='YlOrRd', aspect='auto',
                    interpolation='nearest')

    if direction == 'gnn_to_trans':
        ax.set_ylabel('Atom (GNN node)', fontsize=11)
        ax.set_xlabel('SMILES token (Transformer)', fontsize=11)
    else:
        ax.set_ylabel('SMILES token (Transformer)', fontsize=11)
        ax.set_xlabel('Atom (GNN node)', fontsize=11)

    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=9, rotation=45, ha='right')

    ax.set_title(title, fontsize=13, pad=10)
    plt.colorbar(im, ax=ax, shrink=0.8, label='Attention weight')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f'  Saved: {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Cross-attention visualization')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, default='results/config_used.yaml')
    parser.add_argument('--vocab', type=str, default='results/vocab.json')
    parser.add_argument('--smiles', nargs='+', required=True,
                        help='SMILES strings to visualize')
    parser.add_argument('--output_dir', type=str, default='results/attention_viz')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    config = load_config(args.config)
    device = torch.device(args.device) if args.device else get_device()
    set_seed(42)

    # Load tokenizer
    tokenizer = SmilesTokenizer()
    tokenizer.load_vocab(args.vocab)
    config['model']['transformer']['vocab_size'] = tokenizer.vocab_size

    # Load model
    model = create_model(config, config['tasks'])
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model = model.to(device)
    model.eval()
    print(f'Model loaded on {device}')

    for idx, smi in enumerate(args.smiles):
        print(f'\n--- Molecule {idx+1}: {smi} ---')
        model_inputs, atom_symbols, token_labels = prepare_molecule_batch(
            smi, config, tokenizer, device
        )
        if model_inputs is None:
            print(f'  Failed to process: {smi}')
            continue

        canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smi))

        # Extract attention
        attn_maps = extract_attention_weights(model, model_inputs, device)

        if not attn_maps:
            print('  No cross-attention layers found!')
            continue

        print(f'  Found {len(attn_maps)} cross-attention layers')
        print(f'  Atoms: {len(atom_symbols)}, Tokens: {len(token_labels)}')

        # Find actual token length (strip [PAD] tokens for cleaner visualization)
        try:
            sep_idx = token_labels.index('[SEP]')
            real_token_len = sep_idx + 1  # include [SEP]
        except ValueError:
            real_token_len = len(token_labels)
        real_token_labels = token_labels[:real_token_len]

        for layer_name, attn in attn_maps.items():
            # attn shape: [1, num_heads, query_len, key_len]
            attn_avg = attn[0].mean(axis=0)  # Average over heads

            q_len, k_len = attn_avg.shape
            short_name = layer_name.split('.')[-2] if '.' in layer_name else layer_name

            # Determine direction based on dimensions and trim PAD tokens
            if q_len == len(atom_symbols) and k_len == len(token_labels):
                direction = 'gnn_to_trans'
                row_labels = atom_symbols
                col_labels = real_token_labels
                attn_avg = attn_avg[:, :real_token_len]  # trim PAD columns
                title_dir = 'GNN->Transformer'
            elif q_len == len(token_labels) and k_len == len(atom_symbols):
                direction = 'trans_to_gnn'
                row_labels = real_token_labels
                col_labels = atom_symbols
                attn_avg = attn_avg[:real_token_len, :]  # trim PAD rows
                title_dir = 'Transformer->GNN'
            else:
                print(f'  Skipping {layer_name}: shape {attn_avg.shape} '
                      f'does not match atoms({len(atom_symbols)}) x tokens({len(token_labels)})')
                continue

            fname = f'attn_mol{idx+1}_{short_name}_{direction}.png'
            output_path = os.path.join(args.output_dir, fname)

            plot_attention_heatmap(
                attn_avg, row_labels, col_labels,
                title=f'{title_dir} attention: {canonical[:40]}',
                output_path=output_path,
                direction=direction
            )

    print('\nDone!')


if __name__ == '__main__':
    main()
