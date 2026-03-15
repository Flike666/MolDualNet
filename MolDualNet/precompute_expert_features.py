"""
专家特征预计算脚本
Expert Features Precomputation Script

为107K数据集预计算Morgan指纹 + RDKit描述符,加速训练
"""

import os
import argparse
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from src.expert_features import get_expert_features


def precompute_expert_features(
    input_csv: str,
    output_pkl: str,
    fp_radius: int = 2,
    fp_bits: int = 1024
):
    """
    预计算专家特征 (Morgan指纹 + RDKit描述符)

    Args:
        input_csv: 输入CSV文件路径
        output_pkl: 输出pkl文件路径
        fp_radius: Morgan指纹半径 (默认2对应ECFP4)
        fp_bits: Morgan指纹位数 (默认1024)
    """
    print(f"Loading dataset from {input_csv}...")
    df = pd.read_csv(input_csv)

    if 'smiles' not in df.columns and 'SMILES' not in df.columns:
        raise ValueError("CSV文件必须包含'smiles'或'SMILES'列")

    # 尝试两种列名
    smiles_col = 'smiles' if 'smiles' in df.columns else 'SMILES'
    smiles_list = df[smiles_col].tolist()
    print(f"Total molecules: {len(smiles_list)}")

    # 预计算专家特征
    all_features = []
    failed_count = 0

    print(f"\nPrecomputing expert features...")
    print(f"  - Morgan Fingerprint: radius={fp_radius}, nBits={fp_bits}")
    print(f"  - RDKit Descriptors: 9 properties")
    print(f"  - Total dimension: {fp_bits + 9}")

    for smi in tqdm(smiles_list, desc="Computing"):
        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            # 无效SMILES,使用零向量
            expert_feat = np.zeros(fp_bits + 9, dtype=np.float32)
            failed_count += 1
        else:
            expert_feat = get_expert_features(mol, fp_radius=fp_radius, fp_bits=fp_bits)

        all_features.append(expert_feat)

    # 转换为numpy数组
    all_features = np.stack(all_features)  # [N, 1033]
    print(f"\nPrecomputed features shape: {all_features.shape}")
    print(f"Failed molecules: {failed_count} / {len(smiles_list)} ({failed_count/len(smiles_list)*100:.2f}%)")

    # 计算统计信息
    print(f"\nFeature statistics:")
    print(f"  - Mean: {all_features.mean():.4f}")
    print(f"  - Std:  {all_features.std():.4f}")
    print(f"  - Min:  {all_features.min():.4f}")
    print(f"  - Max:  {all_features.max():.4f}")

    # Morgan指纹稀疏度
    morgan_fp = all_features[:, :fp_bits]
    sparsity = (morgan_fp == 0).sum() / morgan_fp.size
    print(f"  - Morgan FP sparsity: {sparsity*100:.2f}%")

    # 保存
    print(f"\nSaving to {output_pkl}...")
    os.makedirs(os.path.dirname(output_pkl) or '.', exist_ok=True)

    with open(output_pkl, 'wb') as f:
        pickle.dump({
            'features': all_features,
            'smiles': smiles_list,
            'fp_radius': fp_radius,
            'fp_bits': fp_bits,
            'descriptor_dim': 9,
            'total_dim': fp_bits + 9,
        }, f)

    file_size_mb = os.path.getsize(output_pkl) / 1024 / 1024
    print(f"Done! Saved {len(all_features)} expert features")
    print(f"File size: {file_size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="预计算专家特征 (Morgan FP + RDKit Descriptors)")
    parser.add_argument(
        '--input',
        type=str,
        default='data/merged_dataset.csv',
        help='输入CSV文件路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/expert_features_107k.pkl',
        help='输出pkl文件路径'
    )
    parser.add_argument(
        '--fp_radius',
        type=int,
        default=2,
        help='Morgan指纹半径 (默认2对应ECFP4)'
    )
    parser.add_argument(
        '--fp_bits',
        type=int,
        default=1024,
        help='Morgan指纹位数 (默认1024)'
    )

    args = parser.parse_args()

    precompute_expert_features(
        input_csv=args.input,
        output_pkl=args.output,
        fp_radius=args.fp_radius,
        fp_bits=args.fp_bits,
    )


if __name__ == "__main__":
    main()
