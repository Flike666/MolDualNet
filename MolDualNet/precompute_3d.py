#!/usr/bin/env python3
"""
Precompute 3D coordinates cache for a merged dataset CSV.

Example:
  python precompute_3d.py --dataset data/merged_dataset.csv --output data/3d_coords_cache_107k.pkl
"""

import argparse
import os
import pickle
import time
from typing import Dict

import pandas as pd
from rdkit import Chem

from src.features import Geometry3DFeaturizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute 3D coordinates cache")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to CSV dataset (must include a smiles column)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output pickle path for 3D coordinates cache",
    )
    parser.add_argument(
        "--smiles_col",
        type=str,
        default="smiles",
        help="SMILES column name (default: smiles)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output if it already exists",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=1000,
        help="Log progress every N molecules (default: 1000)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if os.path.exists(args.output) and not args.force:
        raise SystemExit(f"Output already exists: {args.output} (use --force to overwrite)")

    df = pd.read_csv(args.dataset)
    if args.smiles_col not in df.columns:
        raise SystemExit(f"SMILES column not found: {args.smiles_col}")

    smiles_list = df[args.smiles_col].tolist()
    total = len(smiles_list)
    print(f"Dataset: {args.dataset}")
    print(f"Total molecules: {total}")
    print(f"Output file: {args.output}")

    featurizer = Geometry3DFeaturizer()
    cache: Dict[str, object] = {}
    num_success = 0
    num_failed = 0
    start = time.time()

    for idx, smiles in enumerate(smiles_list, 1):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            num_failed += 1
            continue
        coords = featurizer.generate_3d_coords(mol)
        if coords is not None:
            cache[smiles] = coords
            num_success += 1
        else:
            num_failed += 1

        if args.log_every > 0 and idx % args.log_every == 0:
            elapsed = time.time() - start
            print(
                f"[{idx}/{total}] success={num_success} failed={num_failed} elapsed={int(elapsed)}s"
            )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    print(f"Saving cache to: {args.output}")
    with open(args.output, "wb") as f:
        pickle.dump(cache, f, protocol=4)

    elapsed = time.time() - start
    print(
        f"Done. Entries={len(cache)} success={num_success} failed={num_failed} elapsed={int(elapsed)}s"
    )


if __name__ == "__main__":
    main()
