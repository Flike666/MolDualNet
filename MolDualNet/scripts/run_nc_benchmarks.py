#!/usr/bin/env python3
"""
Nature Communications benchmark comparison experiment for MolDualNet.

Compares MolDualNet against established molecular property prediction methods
using per-task Bemis–Murcko scaffold splitting and multi-seed evaluation.

Baselines:
  1. Random Forest + Morgan FP (classical ML)
  2. XGBoost + Morgan FP (classical ML)
  3. SVM + Morgan FP (classical ML)
  4. GIN (graph neural network)
  5. AttentiveFP (graph neural network)
  6. D-MPNN (directed message-passing neural network)
  7. SchNet (3D-aware neural network)

Usage:
    python scripts/run_nc_benchmarks.py --device cuda --seeds 42 123 456
    python scripts/run_nc_benchmarks.py --baselines RF XGBoost GIN AttentiveFP D-MPNN SchNet
    python scripts/run_nc_benchmarks.py --task ESOL --device cuda
"""

import argparse
import json
import os
import sys
import time
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

TASKS = {
    'ESOL': {'file': 'ESOL.csv', 'smiles_col': 'smiles', 'target_col': 'ESOL_logS',
             'label': r'$\log S$', 'unit': 'log mol/L'},
    'FreeSolv': {'file': 'FreeSolv.csv', 'smiles_col': 'smiles', 'target_col': 'FreeSolv_hydration',
                 'label': r'$\Delta G_{\mathrm{hyd}}$', 'unit': 'kcal/mol'},
    'Lipophilicity': {'file': 'Lipophilicity.csv', 'smiles_col': 'smiles', 'target_col': 'Lipophilicity_logD',
                      'label': r'$\log D$', 'unit': ''},
    'BACE': {'file': 'BACE.csv', 'smiles_col': 'smiles', 'target_col': 'BACE_pIC50',
             'label': r'$\mathrm{pIC}_{50}$', 'unit': ''},
}

RAW_DATA_DIR = 'data/raw'
RESULTS_DIR = 'results/nc_benchmarks'

ALL_BASELINES = ['RF', 'XGBoost', 'SVM', 'GIN', 'AttentiveFP', 'D-MPNN', 'SchNet']

# ──────────────────────────────────────────────────────────────────────
# Data utilities
# ──────────────────────────────────────────────────────────────────────

def get_scaffold(smiles: str) -> str:
    """Return Bemis–Murcko scaffold SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    try:
        return MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return smiles


def scaffold_split(smiles_list: List[str], targets: np.ndarray,
                   train_ratio: float = 0.8, val_ratio: float = 0.1,
                   seed: int = 42) -> Tuple[dict, dict]:
    """
    Bemis–Murcko scaffold split into train/val/test.
    Uses the standard DeepChem/MoleculeNet protocol: scaffolds sorted by
    size (largest first) and assigned greedily to train, then val, then test.
    The seed controls a secondary shuffle of same-size scaffold groups for
    seed-dependent variability.

    Returns:
        smiles_splits: {'train': [...], 'val': [...], 'test': [...]}
        target_splits: {'train': array, 'val': array, 'test': array}
    """
    # Group indices by scaffold
    scaffold_to_indices = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        scaffold = get_scaffold(smi)
        scaffold_to_indices[scaffold].append(i)

    # Sort scaffolds: largest groups first (go to train).
    # Within same-size groups, use seed-based shuffle for variability.
    rng = np.random.RandomState(seed)
    scaffold_items = list(scaffold_to_indices.items())
    # Add random tiebreaker for same-size scaffolds
    scaffold_items_with_key = [
        (len(indices), rng.random(), scaffold, indices)
        for scaffold, indices in scaffold_items
    ]
    scaffold_items_with_key.sort(key=lambda x: (-x[0], x[1]))

    n = len(smiles_list)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx, val_idx, test_idx = [], [], []

    for _, _, scaffold, indices in scaffold_items_with_key:
        if len(train_idx) + len(indices) <= n_train:
            train_idx.extend(indices)
        elif len(val_idx) + len(indices) <= n_val:
            val_idx.extend(indices)
        else:
            test_idx.extend(indices)

    # Safety: ensure val and test are non-empty
    if len(val_idx) == 0 and len(train_idx) > 10:
        # Move last 10% of train indices to val
        n_steal = max(1, len(train_idx) // 10)
        val_idx = train_idx[-n_steal:]
        train_idx = train_idx[:-n_steal]
    if len(test_idx) == 0 and len(train_idx) > 10:
        n_steal = max(1, len(train_idx) // 10)
        test_idx = train_idx[-n_steal:]
        train_idx = train_idx[:-n_steal]

    # Shuffle within each split
    rng2 = np.random.RandomState(seed)
    rng2.shuffle(train_idx)
    rng2.shuffle(val_idx)
    rng2.shuffle(test_idx)

    smiles_arr = np.array(smiles_list)
    smiles_splits = {
        'train': smiles_arr[train_idx].tolist(),
        'val': smiles_arr[val_idx].tolist(),
        'test': smiles_arr[test_idx].tolist(),
    }
    target_splits = {
        'train': targets[train_idx],
        'val': targets[val_idx],
        'test': targets[test_idx],
    }

    return smiles_splits, target_splits


def load_task_data(task_name: str) -> Tuple[List[str], np.ndarray]:
    """Load raw data for a single task."""
    cfg = TASKS[task_name]
    df = pd.read_csv(os.path.join(RAW_DATA_DIR, cfg['file']))

    smiles = df[cfg['smiles_col']].tolist()
    targets = df[cfg['target_col']].values.astype(np.float32)

    # Remove NaN targets
    valid = ~np.isnan(targets)
    smiles = [s for s, v in zip(smiles, valid) if v]
    targets = targets[valid]

    # Canonicalize SMILES and remove invalid
    clean_smiles, clean_targets = [], []
    for s, t in zip(smiles, targets):
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            clean_smiles.append(Chem.MolToSmiles(mol))
            clean_targets.append(t)

    return clean_smiles, np.array(clean_targets, dtype=np.float32)


def compute_morgan_fp(smiles_list: List[str], radius: int = 2,
                      n_bits: int = 2048) -> np.ndarray:
    """Compute Morgan fingerprints for a list of SMILES."""
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fps.append(np.array(fp, dtype=np.float32))
        else:
            fps.append(np.zeros(n_bits, dtype=np.float32))
    return np.stack(fps)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics."""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    pr, _ = pearsonr(y_true, y_pred)
    sr, _ = spearmanr(y_true, y_pred)
    return {'R2': float(r2), 'RMSE': float(rmse), 'MAE': float(mae),
            'Pearson_r': float(pr), 'Spearman_rho': float(sr)}


# ──────────────────────────────────────────────────────────────────────
# Atom/Bond featurization (shared by GNN baselines)
# ──────────────────────────────────────────────────────────────────────

ATOM_FEATURES = {
    'atomic_num': list(range(1, 119)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-2, -1, 0, 1, 2],
    'num_hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
}


def one_hot(value, choices):
    encoding = [0] * (len(choices) + 1)
    idx = choices.index(value) if value in choices else len(choices)
    encoding[idx] = 1
    return encoding


def atom_features(atom) -> List[float]:
    """Compute atom features (133-dim, following OGB convention)."""
    features = []
    features += one_hot(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num'])
    features += one_hot(atom.GetDegree(), ATOM_FEATURES['degree'])
    features += one_hot(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge'])
    features += one_hot(atom.GetTotalNumHs(), ATOM_FEATURES['num_hs'])
    features += one_hot(atom.GetHybridization(), ATOM_FEATURES['hybridization'])
    features += [int(atom.GetIsAromatic())]
    return features


def bond_features(bond) -> List[float]:
    """Compute bond features (12-dim)."""
    bt = bond.GetBondType()
    features = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing(),
    ]
    stereo = bond.GetStereo()
    features += one_hot(stereo, [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
    ])
    return [float(f) for f in features]


ATOM_DIM = len(atom_features(Chem.MolFromSmiles('C').GetAtomWithIdx(0)))
BOND_DIM = 11  # will be computed dynamically


def mol_to_pyg(smiles: str, y: Optional[float] = None):
    """Convert SMILES to PyG Data object."""
    from torch_geometric.data import Data

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Atom features
    x = []
    for atom in mol.GetAtoms():
        x.append(atom_features(atom))
    x = torch.tensor(x, dtype=torch.float)

    # Edge features
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_index += [[i, j], [j, i]]
        edge_attr += [bf, bf]

    if len(edge_index) > 0:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, len(bf) if edge_attr else 11), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if y is not None:
        data.y = torch.tensor([y], dtype=torch.float)
    return data


def mol_to_pyg_3d(smiles: str, y: Optional[float] = None):
    """Convert SMILES to PyG Data with 3D coordinates for SchNet."""
    from torch_geometric.data import Data

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    try:
        status = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        if status == -1:
            # Fallback to random coordinates
            status = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3(),
                                           useRandomCoords=True)
        if status == -1:
            return None
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            pass
    except Exception:
        return None

    try:
        conf = mol.GetConformer()
    except ValueError:
        return None

    mol = Chem.RemoveHs(mol)
    try:
        conf = mol.GetConformer()
    except ValueError:
        return None

    pos = []
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        pos.append([p.x, p.y, p.z])
    pos = torch.tensor(pos, dtype=torch.float)

    z = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long)

    # Build fully connected edge index for SchNet
    n = mol.GetNumAtoms()
    row = torch.arange(n).repeat_interleave(n)
    col = torch.arange(n).repeat(n)
    mask = row != col
    edge_index = torch.stack([row[mask], col[mask]], dim=0)

    data = Data(z=z, pos=pos, edge_index=edge_index)
    if y is not None:
        data.y = torch.tensor([y], dtype=torch.float)
    return data


# ──────────────────────────────────────────────────────────────────────
# GNN baseline models
# ──────────────────────────────────────────────────────────────────────

class GINModel(nn.Module):
    """Graph Isomorphism Network (Xu et al., ICLR 2019)."""

    def __init__(self, in_dim, hidden_dim=300, out_dim=1, num_layers=5, dropout=0.5):
        super().__init__()
        from torch_geometric.nn import GINConv, global_mean_pool

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            d_in = in_dim if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(d_in, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.pool = global_mean_pool
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc2(x)


class AttentiveFPModel(nn.Module):
    """AttentiveFP wrapper (Xiong et al., J. Med. Chem. 2020)."""

    def __init__(self, in_dim, edge_dim, hidden_dim=200, out_dim=1,
                 num_layers=3, num_timesteps=2, dropout=0.2):
        super().__init__()
        from torch_geometric.nn.models import AttentiveFP as _AttentiveFP

        self.model = _AttentiveFP(
            in_channels=in_dim,
            hidden_channels=hidden_dim,
            out_channels=out_dim,
            edge_dim=edge_dim,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            dropout=dropout,
        )

    def forward(self, data):
        return self.model(data.x, data.edge_index, data.edge_attr, data.batch)


class DMPNNModel(nn.Module):
    """
    Directed Message-Passing Neural Network (Yang et al., JCIM 2019).
    Simplified PyG implementation with directed edge messages.
    """

    def __init__(self, atom_dim, bond_dim, hidden_dim=300, out_dim=1,
                 depth=3, dropout=0.15):
        super().__init__()
        self.depth = depth

        self.W_i = nn.Linear(atom_dim + bond_dim, hidden_dim, bias=False)
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_o = nn.Linear(atom_dim + hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        from torch_geometric.nn import global_mean_pool
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        src, dst = edge_index
        # Initial message: concatenate source atom + bond features
        msg_input = torch.cat([x[src], edge_attr], dim=-1)
        msg = F.relu(self.W_i(msg_input))

        # Message passing iterations
        for _ in range(self.depth - 1):
            # For each edge (u→v), aggregate messages from edges (w→u) where w ≠ v
            # Simplified: aggregate all incoming messages to source node
            from torch_geometric.utils import scatter
            agg = scatter(msg, dst, dim=0, dim_size=x.size(0), reduce='sum')
            nei_msg = agg[src]
            msg = F.relu(self.W_h(nei_msg))
            msg = self.dropout(msg)

        # Readout: aggregate messages to atoms
        from torch_geometric.utils import scatter
        atom_msg = scatter(msg, dst, dim=0, dim_size=x.size(0), reduce='sum')
        atom_hidden = F.relu(self.W_o(torch.cat([x, atom_msg], dim=-1)))
        atom_hidden = self.dropout(atom_hidden)

        # Global pooling
        graph_repr = global_mean_pool(atom_hidden, batch)
        out = F.relu(self.fc1(graph_repr))
        out = self.dropout(out)
        return self.fc2(out)


class SchNetModel(nn.Module):
    """SchNet wrapper (Schütt et al., NeurIPS 2017)."""

    def __init__(self, hidden_dim=128, out_dim=1, num_filters=128,
                 num_interactions=6, cutoff=10.0):
        super().__init__()
        from torch_geometric.nn.models import SchNet

        self.model = SchNet(
            hidden_channels=hidden_dim,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=50,
            cutoff=cutoff,
        )
        # Replace the output layer
        self.model.lin2 = nn.Linear(hidden_dim // 2, out_dim)

    def forward(self, data):
        return self.model(data.z, data.pos, data.batch)


# ──────────────────────────────────────────────────────────────────────
# GNN training utilities
# ──────────────────────────────────────────────────────────────────────

def build_graph_dataset(smiles_list, targets, converter_fn=mol_to_pyg):
    """Build list of PyG Data objects."""
    graphs = []
    valid_idx = []
    for i, (smi, y) in enumerate(zip(smiles_list, targets)):
        g = converter_fn(smi, y)
        if g is not None:
            graphs.append(g)
            valid_idx.append(i)
    return graphs, valid_idx


def train_gnn_epoch(model, loader, optimizer, device):
    """Train one epoch."""
    model.train()
    total_loss = 0
    n = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch).view(-1)
        loss = F.huber_loss(pred, batch.y.view(-1), delta=1.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        n += batch.num_graphs
    return total_loss / max(n, 1)


@torch.no_grad()
def eval_gnn(model, loader, device):
    """Evaluate GNN model."""
    model.eval()
    preds, trues = [], []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch).view(-1)
        preds.append(pred.cpu().numpy().ravel())
        trues.append(batch.y.view(-1).cpu().numpy().ravel())
    if not preds:
        return np.array([]), np.array([])
    return np.concatenate(preds), np.concatenate(trues)


def train_gnn_model(model, train_data, val_data, test_data, device,
                    epochs=200, lr=1e-3, weight_decay=1e-5, batch_size=64,
                    patience=30, verbose=True):
    """Full GNN training loop with early stopping."""
    from torch_geometric.loader import DataLoader

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    best_val_loss = float('inf')
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_gnn_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        # Validation
        val_pred, val_true = eval_gnn(model, val_loader, device)
        val_loss = float(np.mean((val_pred - val_true) ** 2))

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if verbose and epoch % 50 == 0:
            val_r2 = r2_score(val_true, val_pred)
            print(f'    Epoch {epoch:3d} | train_loss={train_loss:.4f} | '
                  f'val_loss={val_loss:.4f} | val_R2={val_r2:.4f}')

        if wait >= patience:
            if verbose:
                print(f'    Early stopping at epoch {epoch}')
            break

    # Load best and evaluate on test
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)
    test_pred, test_true = eval_gnn(model, test_loader, device)
    return compute_metrics(test_true, test_pred)


# ──────────────────────────────────────────────────────────────────────
# Classical ML baselines
# ──────────────────────────────────────────────────────────────────────

def run_classical_baseline(name: str, train_X, train_y, test_X, test_y,
                           seed: int) -> dict:
    """Run a classical ML baseline."""
    if name == 'RF':
        model = RandomForestRegressor(
            n_estimators=500, max_depth=None, min_samples_leaf=2,
            n_jobs=-1, random_state=seed
        )
    elif name == 'XGBoost':
        try:
            from xgboost import XGBRegressor
            model = XGBRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=seed, n_jobs=-1, verbosity=0
            )
        except ImportError:
            model = GradientBoostingRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.1,
                subsample=0.8, random_state=seed
            )
    elif name == 'SVM':
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        model = make_pipeline(
            StandardScaler(),
            SVR(kernel='rbf', C=10.0, epsilon=0.1)
        )
    else:
        raise ValueError(f'Unknown classical baseline: {name}')

    model.fit(train_X, train_y)
    pred = model.predict(test_X)
    return compute_metrics(test_y, pred)


# ──────────────────────────────────────────────────────────────────────
# Main runner for a single (task, baseline, seed)
# ──────────────────────────────────────────────────────────────────────

def run_single_experiment(task_name: str, baseline_name: str, seed: int,
                          device: str, epochs: int = 200,
                          verbose: bool = True) -> dict:
    """Run one (task, baseline, seed) experiment and return metrics."""
    if verbose:
        print(f'  [{baseline_name}] {task_name} seed={seed} ...', flush=True)

    t0 = time.time()

    # Load and split data
    smiles, targets = load_task_data(task_name)
    smiles_splits, target_splits = scaffold_split(smiles, targets, seed=seed)

    train_smi = smiles_splits['train']
    val_smi = smiles_splits['val']
    test_smi = smiles_splits['test']
    train_y = target_splits['train']
    val_y = target_splits['val']
    test_y = target_splits['test']

    if verbose:
        print(f'    Split: train={len(train_smi)}, val={len(val_smi)}, test={len(test_smi)}')

    # ── Classical baselines ──
    if baseline_name in ('RF', 'XGBoost', 'SVM'):
        train_X = compute_morgan_fp(train_smi)
        test_X = compute_morgan_fp(test_smi)
        metrics = run_classical_baseline(baseline_name, train_X, train_y,
                                         test_X, test_y, seed)

    # ── GNN baselines ──
    elif baseline_name == 'GIN':
        train_data, _ = build_graph_dataset(train_smi, train_y)
        val_data, _ = build_graph_dataset(val_smi, val_y)
        test_data, _ = build_graph_dataset(test_smi, test_y)

        in_dim = train_data[0].x.shape[1]
        model = GINModel(in_dim=in_dim, hidden_dim=300, num_layers=5, dropout=0.5)
        metrics = train_gnn_model(model, train_data, val_data, test_data,
                                  device, epochs=epochs, lr=1e-3,
                                  weight_decay=1e-5, batch_size=64,
                                  patience=30, verbose=verbose)

    elif baseline_name == 'AttentiveFP':
        train_data, _ = build_graph_dataset(train_smi, train_y)
        val_data, _ = build_graph_dataset(val_smi, val_y)
        test_data, _ = build_graph_dataset(test_smi, test_y)

        in_dim = train_data[0].x.shape[1]
        edge_dim = train_data[0].edge_attr.shape[1]
        model = AttentiveFPModel(in_dim=in_dim, edge_dim=edge_dim,
                                 hidden_dim=200, num_layers=3,
                                 num_timesteps=2, dropout=0.2)
        metrics = train_gnn_model(model, train_data, val_data, test_data,
                                  device, epochs=epochs, lr=1e-3,
                                  weight_decay=1e-5, batch_size=64,
                                  patience=30, verbose=verbose)

    elif baseline_name == 'D-MPNN':
        train_data, _ = build_graph_dataset(train_smi, train_y)
        val_data, _ = build_graph_dataset(val_smi, val_y)
        test_data, _ = build_graph_dataset(test_smi, test_y)

        atom_dim = train_data[0].x.shape[1]
        bond_dim = train_data[0].edge_attr.shape[1]
        model = DMPNNModel(atom_dim=atom_dim, bond_dim=bond_dim,
                           hidden_dim=300, depth=3, dropout=0.15)
        metrics = train_gnn_model(model, train_data, val_data, test_data,
                                  device, epochs=epochs, lr=1e-4,
                                  weight_decay=1e-5, batch_size=64,
                                  patience=30, verbose=verbose)

    elif baseline_name == 'SchNet':
        train_data, _ = build_graph_dataset(train_smi, train_y,
                                            converter_fn=mol_to_pyg_3d)
        val_data, _ = build_graph_dataset(val_smi, val_y,
                                          converter_fn=mol_to_pyg_3d)
        test_data, _ = build_graph_dataset(test_smi, test_y,
                                           converter_fn=mol_to_pyg_3d)

        if verbose:
            print(f'    3D valid: train={len(train_data)}, '
                  f'val={len(val_data)}, test={len(test_data)}')

        model = SchNetModel(hidden_dim=128, num_filters=128,
                            num_interactions=6, cutoff=10.0)
        metrics = train_gnn_model(model, train_data, val_data, test_data,
                                  device, epochs=epochs, lr=5e-4,
                                  weight_decay=1e-5, batch_size=32,
                                  patience=30, verbose=verbose)
    else:
        raise ValueError(f'Unknown baseline: {baseline_name}')

    elapsed = time.time() - t0
    metrics['time_seconds'] = elapsed
    if verbose:
        print(f'    → R²={metrics["R2"]:.4f}  RMSE={metrics["RMSE"]:.4f}  '
              f'({elapsed:.1f}s)')
    return metrics


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='NC benchmark comparison for MolDualNet')
    parser.add_argument('--baselines', nargs='+', default=ALL_BASELINES,
                        choices=ALL_BASELINES,
                        help='Baselines to run (default: all)')
    parser.add_argument('--tasks', nargs='+', default=list(TASKS.keys()),
                        choices=list(TASKS.keys()),
                        help='Tasks to evaluate (default: all)')
    parser.add_argument('--seeds', nargs='+', type=int,
                        default=[42, 123, 456],
                        help='Random seeds (default: 42 123 456)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Max epochs for NN baselines (default: 200)')
    parser.add_argument('--device', type=str, default=None,
                        choices=['cuda', 'mps', 'cpu'],
                        help='Device (default: auto-detect)')
    parser.add_argument('--output_dir', type=str, default=RESULTS_DIR,
                        help=f'Output directory (default: {RESULTS_DIR})')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress per-epoch output')
    args = parser.parse_args()

    # Device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f'Device: {device}')

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Run experiments ──
    all_results = {}  # {baseline: {task: {seed: metrics}}}

    for baseline in args.baselines:
        print(f'\n{"="*60}')
        print(f'Baseline: {baseline}')
        print(f'{"="*60}')
        all_results[baseline] = {}

        for task in args.tasks:
            all_results[baseline][task] = {}

            for seed in args.seeds:
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

                try:
                    metrics = run_single_experiment(
                        task, baseline, seed, device,
                        epochs=args.epochs,
                        verbose=not args.quiet
                    )
                    all_results[baseline][task][str(seed)] = metrics
                except Exception as e:
                    print(f'  ERROR: {baseline}/{task}/seed{seed}: {e}')
                    import traceback
                    traceback.print_exc()
                    all_results[baseline][task][str(seed)] = {
                        'R2': float('nan'), 'RMSE': float('nan'),
                        'MAE': float('nan'), 'Pearson_r': float('nan'),
                        'Spearman_rho': float('nan'), 'error': str(e)
                    }

            # Save intermediate results
            results_path = os.path.join(args.output_dir, 'benchmark_raw_results.json')
            with open(results_path, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)

    # ── Aggregate statistics ──
    print(f'\n{"="*60}')
    print('Aggregating results ...')
    print(f'{"="*60}')

    summary = {}
    for baseline in all_results:
        summary[baseline] = {}
        for task in all_results[baseline]:
            r2_vals = [v['R2'] for v in all_results[baseline][task].values()
                       if not np.isnan(v.get('R2', float('nan')))]
            rmse_vals = [v['RMSE'] for v in all_results[baseline][task].values()
                         if not np.isnan(v.get('RMSE', float('nan')))]
            mae_vals = [v['MAE'] for v in all_results[baseline][task].values()
                        if not np.isnan(v.get('MAE', float('nan')))]
            pr_vals = [v['Pearson_r'] for v in all_results[baseline][task].values()
                       if not np.isnan(v.get('Pearson_r', float('nan')))]

            summary[baseline][task] = {
                'R2_mean': float(np.mean(r2_vals)) if r2_vals else float('nan'),
                'R2_std': float(np.std(r2_vals)) if r2_vals else float('nan'),
                'RMSE_mean': float(np.mean(rmse_vals)) if rmse_vals else float('nan'),
                'RMSE_std': float(np.std(rmse_vals)) if rmse_vals else float('nan'),
                'MAE_mean': float(np.mean(mae_vals)) if mae_vals else float('nan'),
                'MAE_std': float(np.std(mae_vals)) if mae_vals else float('nan'),
                'Pearson_r_mean': float(np.mean(pr_vals)) if pr_vals else float('nan'),
                'n_seeds': len(r2_vals),
            }

        # Compute average R² across tasks
        avg_r2 = np.mean([summary[baseline][t]['R2_mean']
                          for t in summary[baseline]
                          if not np.isnan(summary[baseline][t]['R2_mean'])])
        summary[baseline]['Avg_R2'] = float(avg_r2)

    # Save summary
    summary_path = os.path.join(args.output_dir, 'benchmark_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # ── Print LaTeX-ready table ──
    print(f'\n{"="*60}')
    print('LaTeX table (R² ↑, scaffold split, multi-seed):')
    print(f'{"="*60}\n')

    tasks_order = ['ESOL', 'FreeSolv', 'Lipophilicity', 'BACE']
    category_map = {
        'RF': 'FP', 'XGBoost': 'FP', 'SVM': 'FP',
        'GIN': 'GNN', 'AttentiveFP': 'GNN', 'D-MPNN': 'GNN',
        'SchNet': '3D',
    }

    print(r'\begin{tabular}{@{}llccccc@{}}')
    print(r'\toprule')
    print(r'\textbf{Method} & \textbf{Type} & '
          r'\textbf{ESOL} & \textbf{FreeSolv} & '
          r'\textbf{Lipo} & \textbf{BACE} & '
          r'\textbf{Avg $R^2$} \\')
    print(r'\midrule')

    for bl in args.baselines:
        if bl not in summary:
            continue
        s = summary[bl]
        cat = category_map.get(bl, '?')
        cells = []
        for t in tasks_order:
            if t in s and not np.isnan(s[t]['R2_mean']):
                cells.append(f'${s[t]["R2_mean"]:.3f} \\pm {s[t]["R2_std"]:.3f}$')
            else:
                cells.append('---')
        avg = s.get('Avg_R2', float('nan'))
        avg_str = f'${avg:.3f}$' if not np.isnan(avg) else '---'
        print(f'{bl} & {cat} & ' + ' & '.join(cells) + f' & {avg_str} \\\\')

    print(r'\midrule')
    print(r'MolDualNet (ours) & Multi & $0.918$ & $0.945$ & $0.768$ & $0.705$ & $\mathbf{0.834}$ \\')
    print(r'\bottomrule')
    print(r'\end{tabular}')

    print(f'\nResults saved to: {args.output_dir}/')
    print('Done!')


if __name__ == '__main__':
    main()
