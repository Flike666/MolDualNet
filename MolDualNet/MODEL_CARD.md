# Model Card for MolDualNet

MolDualNet is a multimodal deep learning model for molecular property prediction that fuses molecular graphs, SMILES strings, expert descriptors, and lightweight bond-level geometry through bidirectional gated cross-attention. It is developed for task-dependent multimodal molecular property prediction under distribution shift.

## Model Details

### Model Description

- **Developed by:** Zihan Zhang, Xuezhou Zhao, Dan Wu
- **Affiliations:** International College of Pharmaceutical Innovation, Soochow University; School of Pharmacy and Biomolecular Sciences, RCSI; Department of Chemistry, RCSI
- **Contact:** danwu@suda.edu.cn
- **Model type:** Multimodal neural network for multi-task molecular property regression
- **Language(s):** SMILES (character-level tokenization)
- **License:** MIT (see LICENSE file in repository)
- **Paper:** *MolDualNet for task-dependent multimodal molecular property prediction under distribution shift*
- **Repository:** https://github.com/Flike666/MolDualNet

### Architecture Summary

MolDualNet contains three parallel encoding branches and a fusion module:

| Branch | Encoder | Input features | Output dim |
|--------|---------|----------------|------------|
| Graph | 6-layer edge-aware GATv2 (8 heads, d=256) | 82-d atom features, 20-d edge features (12 base + 8 RBF 3D) | 256 |
| String | 6-layer Transformer (8 heads, d=256) | Character-level SMILES tokens, max_len=256, learnable positional embeddings | 256 |
| Expert | 3-layer MLP (1033→1024→512→256) | 1024-bit Morgan fingerprint + 9 RDKit physicochemical descriptors | 256 |

**Fusion:** Two layers of bidirectional cross-attention (pre-LayerNorm, gated residual) between graph and string branches, followed by a learnable gated fusion and concatenation with the expert embedding (total 512-d), before task-specific prediction heads.

**Parameters:** ~15.0 M (full model), ~4.8 M (graph-only variant).

**Training:** Two-stage schedule (15 epochs warm-up without active cross-attention, then full fusion), AdamW with cosine decay, gradient clipping, independent modality dropout (p=0.2), Huber loss (δ=1.0) with multi-task label masking.

## Intended Use

### Intended Users

- Computational chemists and medicinal chemists performing early-stage virtual screening
- ML-for-chemistry researchers studying multimodal fusion and distribution-shift robustness
- Researchers benchmarking graph / sequence / descriptor-based baselines on MoleculeNet tasks

### Intended Tasks

MolDualNet is trained to predict four molecular properties jointly:

| Task | Property | Units | Dataset |
|------|----------|-------|---------|
| ESOL | Aqueous solubility | log S (mol/L) | MoleculeNet ESOL |
| FreeSolv | Hydration free energy | ΔG_hyd (kcal/mol) | MoleculeNet FreeSolv |
| Lipophilicity | Octanol–water partition coefficient | log D | MoleculeNet Lipophilicity |
| BACE | β-secretase 1 inhibitory potency | pIC₅₀ | MoleculeNet BACE |

Typical deployment settings:
- Ranking compounds for further triage in early discovery pipelines
- Complementing (not replacing) low-cost reference models such as the Delaney ESOL equation or Crippen log P
- Serving as a reference implementation for studying cross-attention in multimodal molecular learning

### Out-of-Scope Use

MolDualNet **should not** be used for:

- Regulatory, safety, or clinical decisions
- Toxicity, ADMET endpoints outside the four trained properties, or any property the model was not trained on
- Protein–ligand binding affinity prediction beyond the specific BACE1 ligand series in the training set; the model is strictly **ligand-only** and does not incorporate target structure
- Molecules outside drug-like chemical space (e.g., metal complexes, polymers, peptides, materials) — the training distribution is MoleculeNet-like small organics
- Sole basis for go/no-go decisions in drug development
- Large-scale generative design (the model was not trained as a generator and is not calibrated for scoring novel distributions)

## Training Data

- **Total:** 7,068 unique molecules (5,976 train / 747 validation / 749 test) from the merged union of ESOL, FreeSolv, Lipophilicity, and BACE benchmarks in MoleculeNet.
- **Per-task counts:** ESOL (1,117), FreeSolv (642), Lipophilicity (4,200), BACE (1,513) after deduplication.
- **Overlap:** 398 molecules carry labels for more than one task.
- **Preprocessing:** RDKit canonicalization, InChIKey-based deduplication across the merged union.
- **Splitting:** Deterministic Bemis–Murcko scaffold splitting with an 80/10/10 train/validation/test partition per task.
- **3D geometry:** Single ETKDGv3 conformer with MMFF94 optimization for bond-length RBF encoding.

## Evaluation

### Evaluation Data

Three evaluation protocols were used:

1. **Scaffold-split benchmark** — held-out test partitions of the four MoleculeNet tasks (in-distribution).
2. **Scaffold-held-out extrapolation** — for FreeSolv (185 molecules) and Lipophilicity (630 molecules), where test scaffolds are entirely disjoint from training scaffolds.
3. **External cross-dataset validation** — ESOL evaluated on AqSolDB (8,192 molecules); BACE evaluated on ChEMBL BACE1 (7,722 molecules); both external sets deduplicated against the training union.

### Metrics

R², RMSE, MAE, Pearson r, Spearman ρ. Confidence intervals estimated via 1,000 bootstrap resamples for external validation; ablation significance assessed with paired t-tests across five random seeds.

### Quantitative Results

**Scaffold-split (in-distribution):**

| Task | R² | RMSE | MAE | Pearson r |
|------|----|----|-----|-----------|
| ESOL | 0.918 | 0.565 | 0.412 | 0.961 |
| FreeSolv | 0.945 | 0.813 | 0.571 | 0.973 |
| Lipophilicity | 0.768 | 0.584 | 0.447 | 0.876 |
| BACE | 0.705 | 0.734 | 0.549 | 0.841 |
| **Average** | **0.834** | — | — | — |

**Scaffold-held-out and external validation:**

| Task | Protocol | n | R² | RMSE |
|------|----------|---|----|----|
| FreeSolv | Scaffold-held-out | 185 | 0.976 | 0.608 |
| Lipophilicity | Scaffold-held-out | 630 | 0.808 | 0.511 |
| ESOL | AqSolDB (external) | 8,192 | 0.542 | 1.637 |
| BACE | ChEMBL (external) | 7,722 | 0.261 | 1.103 |

**Comparison with strongest baseline (AttentiveFP):** MolDualNet averages R² = 0.834 vs. AttentiveFP R² = 0.661 under scaffold split.

**Ablation (5-seed):** Removing cross-attention is the largest single contributor to average R² (−0.035, paired t-test P < 0.01); expert descriptors and 3D geometry each contribute smaller but statistically significant gains. A graph-only variant retains ≈98% of full-model average R² at ≈18% of the training cost.

## Limitations

1. **Single conformer.** The geometric branch uses a single ETKDG conformer and therefore captures only low-cost spatial bias, not full conformational reasoning. Conformer ensembles or equivariant 3D encoders are expected to improve settings dominated by conformational diversity.
2. **Ligand-only bioactivity.** BACE prediction is entirely ligand-based. The model cannot directly model protein–ligand complementarity, and external BACE R² (0.261 on ChEMBL) reflects this limitation.
3. **Modest supervised corpus.** 7,068 labeled molecules across four tasks is small relative to modern pretrained chemical foundation models; benefits of high-capacity multimodal fusion may be constrained at this scale.
4. **Task-dependent benefit.** Multimodal fusion is not a universal advantage — on FreeSolv (n=642) the full model slightly underperforms simpler variants, consistent with data-limited overfitting of additional cross-attention parameters.
5. **Distribution-shift gap.** Performance degrades from in-distribution benchmarks (R² = 0.83 average) to external evaluation (R² = 0.54 for AqSolDB, 0.26 for ChEMBL BACE1). Users should not assume benchmark performance transfers unchanged to their own chemical space.
6. **Scaffold coverage.** Bemis–Murcko splitting preserves some near-scaffold leakage; the scaffold-held-out protocol is stricter but still within the MoleculeNet chemotype distribution.

## Ethical Considerations and Risks

- Predictions may inherit biases from MoleculeNet curation (e.g., over-representation of certain scaffold families or assay conditions).
- False-positive or false-negative predictions on bioactivity endpoints could influence downstream experimental prioritization. We recommend orthogonal experimental validation before advancing any compound.
- The model should not be used to screen for dual-use chemical hazards; it is not trained or calibrated for that purpose.

## Recommendations

- **For physicochemical tasks (ESOL, FreeSolv, Lipophilicity)** on molecules close to the training distribution, MolDualNet gives substantial gains over fingerprint-based and single-modality GNN baselines.
- **For resource-constrained or rapid-iteration settings**, the graph-only variant is defensible (98% of full-model average R² at 18% of training cost).
- **For bioactivity prediction**, interpret external R² cautiously; combine with structure-based methods whenever target information is available.
- **For deployment on new chemical space**, perform scaffold-held-out or external validation on a representative sample before trusting predictions.

## How to Use

```bash
# Install
git clone https://github.com/Flike666/MolDualNet.git
cd MolDualNet
pip install -r requirements.txt

# Train from scratch
python train.py --config configs/config_107k.yaml

# Resume from checkpoint
python train.py --config configs/config_107k.yaml --checkpoint checkpoints/best_model.pt
```

See the repository README for precomputation scripts (3D conformers, expert features) and experiment scripts (multi-seed ablation, cross-dataset validation, attention visualization).

## Citation

```bibtex
@article{zhang2026moldualnet,
  title   = {MolDualNet for task-dependent multimodal molecular property prediction under distribution shift},
  author  = {Zhang, Zihan and Zhao, Xuezhou and Wu, Dan},
  journal = {Communications Chemistry (under review)},
  year    = {2026}
}
```

## Model Card Authors

Zihan Zhang, Xuezhou Zhao, Dan Wu.

## Model Card Contact

Dan Wu — danwu@suda.edu.cn
