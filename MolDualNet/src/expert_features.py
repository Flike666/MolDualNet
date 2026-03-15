"""
Expert Features Module - Morgan Fingerprints + RDKit Descriptors
专家特征模块 - 摩根指纹 + RDKit描述符

基于AIDD 101 Part 3的All-In-One架构
"""

import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import numpy as np
from typing import Optional, List


def get_morgan_fingerprint(mol: Chem.Mol, radius: int = 2, nBits: int = 1024) -> np.ndarray:
    """
    计算Morgan指纹 (ECFP)

    Args:
        mol: RDKit分子对象
        radius: 指纹半径 (默认2对应ECFP4)
        nBits: 指纹位数 (默认1024)

    Returns:
        np.ndarray: Morgan指纹向量 [nBits]
    """
    if mol is None:
        return np.zeros(nBits, dtype=np.float32)

    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        return np.array(fp, dtype=np.float32)
    except:
        return np.zeros(nBits, dtype=np.float32)


def get_rdkit_descriptors(mol: Chem.Mol) -> np.ndarray:
    """
    计算RDKit分子描述符

    Args:
        mol: RDKit分子对象

    Returns:
        np.ndarray: 描述符向量 [9维]
            - MolWt: 分子量
            - MolLogP: LogP (亲脂性)
            - NumHDonors: 氢键供体数
            - NumHAcceptors: 氢键受体数
            - TPSA: 拓扑极性表面积
            - NumRotatableBonds: 可旋转键数
            - NumAromaticRings: 芳香环数
            - NumSaturatedRings: 饱和环数
            - NumAliphaticRings: 脂肪环数
    """
    if mol is None:
        return np.zeros(9, dtype=np.float32)

    try:
        descriptors = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.NumSaturatedRings(mol),
            Descriptors.NumAliphaticRings(mol),
        ]
        return np.array(descriptors, dtype=np.float32)
    except:
        return np.zeros(9, dtype=np.float32)


def get_expert_features(mol: Chem.Mol, fp_radius: int = 2, fp_bits: int = 1024) -> np.ndarray:
    """
    获取完整的专家特征 (Morgan指纹 + RDKit描述符)

    Args:
        mol: RDKit分子对象
        fp_radius: Morgan指纹半径
        fp_bits: Morgan指纹位数

    Returns:
        np.ndarray: 专家特征向量 [fp_bits + 9]
    """
    morgan_fp = get_morgan_fingerprint(mol, radius=fp_radius, nBits=fp_bits)
    descriptors = get_rdkit_descriptors(mol)

    # 拼接: Morgan FP (1024) + Descriptors (9) = 1033维
    expert_features = np.concatenate([morgan_fp, descriptors])
    return expert_features


class ExpertFeatureEncoder(nn.Module):
    """
    专家特征编码器

    将Morgan指纹 + RDKit描述符映射到统一的嵌入空间
    """

    def __init__(
        self,
        input_dim: int = 1033,  # 1024 (Morgan FP) + 9 (RDKit Descriptors)
        output_dim: int = 256,
        dropout: float = 0.1,
        use_precomputed: bool = True
    ):
        """
        Args:
            input_dim: 输入特征维度 (默认1033)
            output_dim: 输出嵌入维度
            dropout: Dropout概率
            use_precomputed: 是否使用预计算特征
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_precomputed = use_precomputed

        # 投影层: 1033 -> output_dim
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim * 4),
            nn.LayerNorm(output_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(output_dim * 4, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        smiles: Optional[List[str]] = None,
        precomputed_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            smiles: SMILES字符串列表 (如果不使用预计算)
            precomputed_features: 预计算的专家特征 [batch, 1033]

        Returns:
            torch.Tensor: 编码后的特征 [batch, output_dim]
        """
        if self.use_precomputed:
            if precomputed_features is None:
                raise ValueError("use_precomputed=True但未提供precomputed_features")

            # 使用预计算特征
            features = precomputed_features
        else:
            if smiles is None:
                raise ValueError("use_precomputed=False但未提供smiles")

            # 实时计算专家特征
            batch_features = []
            for smi in smiles:
                mol = Chem.MolFromSmiles(smi)
                expert_feat = get_expert_features(mol)
                batch_features.append(expert_feat)

            features = torch.tensor(
                np.stack(batch_features),
                dtype=torch.float32,
                device=self.projection[0].weight.device
            )

        # 投影到嵌入空间
        output = self.projection(features)

        return output

    def get_feature_dim(self) -> int:
        """获取输出特征维度"""
        return self.output_dim
