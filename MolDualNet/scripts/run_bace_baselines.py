"""
BACE ChEMBL External Validation — Strong Baseline Comparison
=============================================================
Trains Morgan-FP + RF/XGBoost on MoleculeNet BACE scaffold-split training set,
then evaluates on the same 7,722 ChEMBL BACE1 molecules used for MolDualNet.

Output:
  results/cross_dataset_validation/bace_baseline_comparison.json
  results/cross_dataset_validation/bace_baseline_scatter.png

Usage:
  python scripts/run_bace_baselines.py
"""

import os, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# ── Optional: XGBoost ────────────────────────────────────────────────────────
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[INFO] xgboost not installed; skipping XGBoost baseline.")

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW      = ROOT / "data" / "raw" / "BACE.csv"
CHEMBL_RAW    = ROOT / "results" / "cross_dataset_validation" / "data_cache" / "chembl_bace1_raw.csv"
MOLDUALNET_PRED = ROOT / "results" / "cross_dataset_validation" / "per_task_predictions" / "BACE_pIC50_predictions.csv"
OUT_DIR       = ROOT / "results" / "cross_dataset_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Feature extraction ─────────────────────────────────────────────────────────
def morgan_fp(smiles_list, radius=2, n_bits=2048):
    """Morgan fingerprints (bit-vector). Returns (n_valid, n_bits) array + valid indices."""
    fps, valid_idx = [], []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fps.append(np.array(fp))
            valid_idx.append(i)
    return np.array(fps, dtype=np.float32), valid_idx

def rdkit_descriptors(smiles_list):
    """9 classic RDKit descriptors matching MolDualNet's expert features subset."""
    desc_fns = [
        Descriptors.MolWt, Descriptors.MolLogP, Descriptors.NumHDonors,
        Descriptors.NumHAcceptors, Descriptors.TPSA,
        Descriptors.NumRotatableBonds, Descriptors.RingCount,
        Descriptors.NumAromaticRings, Descriptors.FractionCSP3,
    ]
    rows, valid_idx = [], []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            rows.append([fn(mol) for fn in desc_fns])
            valid_idx.append(i)
    return np.array(rows, dtype=np.float32), valid_idx

# ── Scaffold split (reproduce MolDualNet Protocol-1) ──────────────────────────
def scaffold_split(df, train_frac=0.8, val_frac=0.1, seed=42):
    from rdkit.Chem.Scaffolds import MurckoScaffold
    scaffolds = {}
    for idx, smi in enumerate(df['smiles']):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            scaffold = ''
        else:
            scaffold = MurckoScaffold.MakeScaffoldGeneric(mol)
            scaffold = Chem.MolToSmiles(scaffold) if scaffold else ''
        scaffolds.setdefault(scaffold, []).append(idx)

    rng = np.random.default_rng(seed)
    scaffold_list = sorted(scaffolds.items(), key=lambda x: len(x[1]), reverse=True)
    rng.shuffle(scaffold_list)  # deterministic shuffle

    n = len(df)
    train_cut = int(n * train_frac)
    val_cut   = int(n * (train_frac + val_frac))

    train_idx, val_idx, test_idx = [], [], []
    for _, idxs in scaffold_list:
        if len(train_idx) < train_cut:
            train_idx.extend(idxs)
        elif len(train_idx) + len(val_idx) < val_cut:
            val_idx.extend(idxs)
        else:
            test_idx.extend(idxs)

    return sorted(train_idx), sorted(val_idx), sorted(test_idx)

# ── Metrics ───────────────────────────────────────────────────────────────────
def bootstrap_ci(y_true, y_pred, metric_fn, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        try:
            vals.append(metric_fn(y_true[idx], y_pred[idx]))
        except Exception:
            pass
    return np.percentile(vals, [2.5, 97.5])

def compute_metrics(y_true, y_pred, name="model"):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = np.mean(np.abs(y_true - y_pred))
    r, _ = stats.pearsonr(y_true, y_pred)
    r2_ci   = bootstrap_ci(y_true, y_pred, r2_score)
    rmse_ci = bootstrap_ci(y_true, y_pred,
                           lambda a, b: np.sqrt(mean_squared_error(a, b)))
    print(f"  {name:30s}  R2={r2:.3f} [{r2_ci[0]:.3f},{r2_ci[1]:.3f}]  "
          f"RMSE={rmse:.3f} [{rmse_ci[0]:.3f},{rmse_ci[1]:.3f}]  r={r:.3f}")
    return dict(r2=r2, r2_ci=r2_ci.tolist(), rmse=rmse, rmse_ci=rmse_ci.tolist(),
                mae=mae, pearson_r=r)

# ── Load training data ────────────────────────────────────────────────────────
print("=" * 60)
print("Loading MoleculeNet BACE training data …")
bace_df = pd.read_csv(DATA_RAW).dropna(subset=['smiles', 'BACE_pIC50'])
train_idx, val_idx, test_idx = scaffold_split(bace_df, seed=42)
train_df = bace_df.iloc[train_idx].reset_index(drop=True)
print(f"  Train: {len(train_df)} | Val: {len(bace_df.iloc[val_idx])} "
      f"| In-dist test: {len(bace_df.iloc[test_idx])}")

# Morgan FP for training
X_train_fp, valid_tr = morgan_fp(train_df['smiles'].tolist())
y_train = train_df['BACE_pIC50'].values[valid_tr]
print(f"  Valid training molecules (Morgan FP): {len(X_train_fp)}")

# ── Load ChEMBL test set ──────────────────────────────────────────────────────
print("\nLoading ChEMBL BACE1 test data …")
# Use MolDualNet's already-filtered 7722 molecules for exact comparability
moldualnet_df = pd.read_csv(MOLDUALNET_PRED)
chembl_smiles = moldualnet_df['smiles'].tolist()
chembl_labels = moldualnet_df['experimental'].values
print(f"  ChEMBL test molecules: {len(chembl_smiles)}")

X_test_fp, valid_te = morgan_fp(chembl_smiles)
y_test_aligned = chembl_labels[valid_te]
print(f"  Valid test molecules (Morgan FP): {len(X_test_fp)}")

# ── MolDualNet reference (same valid subset) ──────────────────────────────────
moldualnet_pred = moldualnet_df['predicted'].values[valid_te]
print("\n── Reference: MolDualNet (all 7722 molecules) ──────────────────────")
_ = compute_metrics(chembl_labels, moldualnet_df['predicted'].values, "MolDualNet (7722)")
print("── Reference: MolDualNet (FP-valid subset) ──────────────────────────")
_ = compute_metrics(y_test_aligned, moldualnet_pred, "MolDualNet (FP-valid)")

# ── Baseline 1: Ridge Regression ─────────────────────────────────────────────
print("\n── Baseline experiments ──────────────────────────────────────────────")
scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_train_fp)
X_te_sc = scaler.transform(X_test_fp)

ridge = Ridge(alpha=10.0)
ridge.fit(X_tr_sc, y_train)
y_pred_ridge = ridge.predict(X_te_sc)
ridge_metrics = compute_metrics(y_test_aligned, y_pred_ridge, "Ridge (Morgan FP)")

# ── Baseline 2: Random Forest ─────────────────────────────────────────────────
rf = RandomForestRegressor(n_estimators=500, max_features='sqrt',
                            n_jobs=-1, random_state=42)
rf.fit(X_train_fp, y_train)
y_pred_rf = rf.predict(X_test_fp)
rf_metrics = compute_metrics(y_test_aligned, y_pred_rf, "Random Forest (Morgan FP)")

# ── Baseline 3: Gradient Boosting ────────────────────────────────────────────
gbr = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05,
                                 max_depth=5, random_state=42)
gbr.fit(X_train_fp, y_train)
y_pred_gbr = gbr.predict(X_test_fp)
gbr_metrics = compute_metrics(y_test_aligned, y_pred_gbr, "GBT (Morgan FP)")

# ── Baseline 4: XGBoost ──────────────────────────────────────────────────────
if HAS_XGB:
    xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5,
                        subsample=0.8, colsample_bytree=0.8,
                        random_state=42, n_jobs=-1, verbosity=0)
    xgb.fit(X_train_fp, y_train)
    y_pred_xgb = xgb.predict(X_test_fp)
    xgb_metrics = compute_metrics(y_test_aligned, y_pred_xgb, "XGBoost (Morgan FP)")

# ── In-distribution RF (for comparison) ──────────────────────────────────────
X_indist_fp, valid_id = morgan_fp(bace_df.iloc[test_idx]['smiles'].tolist())
y_indist = bace_df.iloc[test_idx]['BACE_pIC50'].values[valid_id]
rf_indist = RandomForestRegressor(n_estimators=500, max_features='sqrt',
                                   n_jobs=-1, random_state=42)
rf_indist.fit(X_train_fp, y_train)
y_pred_indist = rf_indist.predict(X_indist_fp)
print("\n── In-distribution test (same scaffold split as MolDualNet Table 5) ─")
_ = compute_metrics(y_indist, y_pred_indist, "RF in-distribution (MoleculeNet)")

# ── Save results ──────────────────────────────────────────────────────────────
results = {
    "dataset": "ChEMBL BACE1 (7722 molecules, same as MolDualNet evaluation)",
    "training_set": "MoleculeNet BACE scaffold split (seed=42), Protocol 1",
    "n_train": int(len(X_train_fp)),
    "n_test": int(len(X_test_fp)),
    "baselines": {
        "Ridge_MorganFP": ridge_metrics,
        "RandomForest_MorganFP": rf_metrics,
        "GBT_MorganFP": gbr_metrics,
    }
}
if HAS_XGB:
    results["baselines"]["XGBoost_MorganFP"] = xgb_metrics

out_json = OUT_DIR / "bace_baseline_comparison.json"
with open(out_json, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n[OK] Results saved to {out_json}")

# ── Scatter plot ──────────────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    pairs = [
        ("Ridge\n(Morgan FP)", y_pred_ridge, ridge_metrics),
        ("Random Forest\n(Morgan FP)", y_pred_rf, rf_metrics),
        ("Gradient Boosting\n(Morgan FP)", y_pred_gbr, gbr_metrics),
    ]
    for ax, (label, y_pred, m) in zip(axes, pairs):
        ax.scatter(y_test_aligned, y_pred, alpha=0.3, s=8, c='steelblue')
        mn = min(y_test_aligned.min(), y_pred.min())
        mx = max(y_test_aligned.max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], 'r--', lw=1.5, label='Ideal')
        ax.set_xlabel("Experimental pIC₅₀")
        ax.set_ylabel("Predicted pIC₅₀")
        ax.set_title(f"{label}\nR²={m['r2']:.3f}, RMSE={m['rmse']:.3f}, r={m['pearson_r']:.3f}")
        ax.legend(fontsize=8)
    plt.tight_layout()
    out_fig = OUT_DIR / "bace_baseline_scatter.png"
    plt.savefig(out_fig, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Scatter plot saved to {out_fig}")
except Exception as e:
    print(f"[WARN] Plot failed: {e}")

print("\n" + "=" * 60)
print("Summary: copy these into LaTeX Table 9 after confirming numbers.")
print("=" * 60)
print(f"{'Method':<32} {'R2':>6}  {'RMSE':>6}  {'r':>6}")
print("-" * 52)
for name, m in results["baselines"].items():
    print(f"  {name:<30} {m['r2']:>6.3f}  {m['rmse']:>6.3f}  {m['pearson_r']:>6.3f}")
print(f"  {'MolDualNet (reference)':<30} {'0.261':>6}  {'1.103':>6}  {'0.540':>6}")
