# MolDualNet

A multimodal dual-stream network for molecular property prediction.

MolDualNet fuses graph neural networks, SMILES-level transformers, expert molecular descriptors and lightweight 3D geometry through gated bidirectional cross-attention for robust multi-task molecular property prediction.

## Architecture

MolDualNet integrates four complementary molecular representations:

1. **GNN branch** (GATv2) — encodes molecular graph topology with 82-dim atom features and 12/20-dim bond features
2. **Transformer branch** — character-level SMILES tokenizer with learnable positional encoding and [CLS] pooling
3. **Expert descriptor branch** — 1024-bit Morgan fingerprints + 9 RDKit physicochemical descriptors
4. **3D geometry** — RBF-encoded pairwise distances, bond angles and dihedrals from ETKDG conformers

Representations are fused via bidirectional cross-attention with gated fusion before task-specific prediction heads.

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:**
- Python >= 3.8
- PyTorch >= 2.0.0
- PyTorch Geometric >= 2.4.0
- RDKit >= 2023.3.1
- pandas, numpy, scikit-learn, matplotlib, seaborn, pyyaml, tqdm

## Quick Start

### Training

```bash
# Train with default config
python train.py --config configs/config_107k.yaml

# Custom hyperparameters
python train.py --config configs/config_107k.yaml --epochs 50 --batch_size 16 --learning_rate 0.0005

# Resume from checkpoint
python train.py --config configs/config_107k.yaml --checkpoint checkpoints/best_model.pt
```

### Feature Precomputation (for large datasets >10K molecules)

```bash
python precompute_3d.py --input data/merged_dataset.csv --output data/3d_coords_cache.pkl
python precompute_expert_features.py --input data/merged_dataset.csv --output data/expert_features.pkl
```

## Datasets

MolDualNet is evaluated on four MoleculeNet regression benchmarks:

| Dataset | Property | Molecules |
|---------|----------|-----------|
| ESOL | Aqueous solubility (log S) | 1,117 |
| FreeSolv | Hydration free energy (kcal/mol) | 642 |
| Lipophilicity | Octanol-water partition (log D) | 4,200 |
| BACE | Beta-secretase 1 inhibition (pIC50) | 1,513 |

Place raw CSV files in `data/raw/`. Datasets are available from [MoleculeNet](https://moleculenet.org).

## Reproducing Paper Results

### Benchmark comparison (Table 3 / Figure 3)

```bash
python scripts/run_nc_benchmarks.py
python scripts/analyze_nc_benchmarks.py
```

### Multi-seed ablation (Table 4)

```bash
python scripts/run_multiseed_ablation.py
```

### Cross-dataset generalization (Table 5 / Figure 4)

```bash
python scripts/cross_dataset_validation.py
```

### Additional experiments

```bash
python scripts/run_3d_sensitivity.py            # 3D feature sensitivity
python scripts/visualize_cross_attention.py      # Attention visualization
python scripts/compute_attention_entropy.py      # Attention entropy analysis
python scripts/external_validation.py            # External validation
```

## Configuration

All training behavior is driven by YAML config files in `configs/`:

- `config_107k.yaml` — full multimodal model (default)
- `config_gnn_only.yaml` — GNN-only baseline
- `config_transformer_only.yaml` — Transformer-only baseline

Key config sections: `data`, `tasks`, `model` (with sub-sections per modality), `training`, `pretrain`, `save`.

## Project Structure

```
MolDualNet/
├── train.py                    # Main training entry point
├── precompute_3d.py            # 3D coordinate precomputation
├── precompute_expert_features.py  # Expert feature precomputation
├── requirements.txt
├── configs/
│   ├── config_107k.yaml        # Full model config
│   ├── config_gnn_only.yaml    # GNN-only config
│   └── config_transformer_only.yaml
├── src/
│   ├── model.py                # MoleculePropertyPredictor, TaskHead, MultiTaskLoss
│   ├── gnn.py                  # GATv2-based graph encoder
│   ├── transformer.py          # SMILES transformer encoder
│   ├── cross_attention.py      # Bidirectional cross-attention fusion
│   ├── features.py             # Atom/bond/3D featurizers
│   ├── expert_features.py      # Morgan FP + RDKit descriptors
│   ├── dataset.py              # MoleculeDataset with PyG batching
│   ├── tokenizer.py            # Character-level SMILES tokenizer
│   ├── trainer.py              # Training loop with early stopping
│   ├── evaluator.py            # Metrics and visualization
│   ├── pretrain.py             # Self-supervised pretraining
│   └── utils.py                # Utility functions
├── scripts/                    # Experiment reproduction scripts
├── data/raw/                   # Place MoleculeNet CSVs here
├── checkpoints/                # Saved model weights
└── results/                    # Output figures and metrics
```

## Platform Notes

- **Device auto-detection**: CUDA > MPS > CPU. Override with `--device cuda|mps|cpu`.
- **Windows**: Set `num_workers: 0` and `pin_memory: false` in config.

## Citation

If you use MolDualNet in your research, please cite:

```bibtex
@article{moldualnet2026,
  title={MolDualNet: a multimodal dual-stream network for molecular property prediction},
  author={[Authors]},
  journal={Nature Communications},
  year={2026}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
