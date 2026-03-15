"""
原子和化学键特征提取模块
Atom and Bond Feature Extraction Module
包含3D几何特征提取功能
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from typing import List, Tuple, Optional, Dict


# 支持的原子类型列表
ATOM_TYPES = [
    'C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'Si',
    'B', 'Na', 'K', 'Ca', 'Mg', 'Zn', 'Fe', 'Cu', 'Mn', 'Se',
    'H', 'Unknown'
]

# 杂化类型
HYBRIDIZATION_TYPES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
    Chem.rdchem.HybridizationType.UNSPECIFIED,
]

# 键类型
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

# 立体化学类型
STEREO_TYPES = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS,
]


def one_hot_encoding(value, choices: List) -> List[int]:
    """将值编码为one-hot向量"""
    encoding = [0] * (len(choices) + 1)  # +1 for unknown
    if value in choices:
        encoding[choices.index(value)] = 1
    else:
        encoding[-1] = 1  # unknown
    return encoding


class AtomFeaturizer:
    """原子特征提取器"""

    def __init__(self):
        self.atom_types = ATOM_TYPES
        self.hybridization_types = HYBRIDIZATION_TYPES

    @property
    def feature_dim(self) -> int:
        """返回原子特征维度"""
        # one_hot_encoding adds +1 for unknown, so actual dims are:
        # atom_type: 21+1=22, degree: 11+1=12, formal_charge: 11+1=12,
        # num_h: 9+1=10, hybridization: 6+1=7, aromatic: 1, in_ring: 1,
        # num_radical: 5+1=6, chirality: 5+1=6, mass: 1, ring_sizes: 4
        return 22 + 12 + 12 + 10 + 7 + 1 + 1 + 6 + 6 + 1 + 4  # = 82

    def __call__(self, atom: Chem.Atom) -> np.ndarray:
        """提取单个原子的特征"""
        features = []

        # 原子类型 one-hot (22 + 1 = 23)
        atom_symbol = atom.GetSymbol()
        features.extend(one_hot_encoding(atom_symbol, self.atom_types[:-1]))

        # 原子度数 (0-10, 11维)
        degree = min(atom.GetDegree(), 10)
        features.extend(one_hot_encoding(degree, list(range(11))))

        # 形式电荷 (-5 到 5, 11维)
        formal_charge = max(-5, min(atom.GetFormalCharge(), 5))
        features.extend(one_hot_encoding(formal_charge + 5, list(range(11))))

        # 连接氢原子数 (0-8, 9维)
        num_h = min(atom.GetTotalNumHs(), 8)
        features.extend(one_hot_encoding(num_h, list(range(9))))

        # 杂化类型 (6 + 1 = 7)
        hybridization = atom.GetHybridization()
        features.extend(one_hot_encoding(hybridization, self.hybridization_types))

        # 是否芳香 (1维)
        features.append(1 if atom.GetIsAromatic() else 0)

        # 是否在环中 (1维)
        features.append(1 if atom.IsInRing() else 0)

        # 自由基电子数 (0-4, 5维)
        num_radical = min(atom.GetNumRadicalElectrons(), 4)
        features.extend(one_hot_encoding(num_radical, list(range(5))))

        # 手性标签 (5维)
        chiral_tag = int(atom.GetChiralTag())
        features.extend(one_hot_encoding(chiral_tag, list(range(5))))

        # 原子质量 (归一化, 1维)
        mass = atom.GetMass() / 100.0  # 归一化
        features.append(mass)

        # 所在环的大小特征 (4维: 3环, 4环, 5环, 6+环)
        ring_info = atom.GetOwningMol().GetRingInfo()
        ring_sizes = [0, 0, 0, 0]  # 3, 4, 5, 6+
        for ring in ring_info.AtomRings():
            if atom.GetIdx() in ring:
                size = len(ring)
                if size == 3:
                    ring_sizes[0] = 1
                elif size == 4:
                    ring_sizes[1] = 1
                elif size == 5:
                    ring_sizes[2] = 1
                else:
                    ring_sizes[3] = 1
        features.extend(ring_sizes)

        return np.array(features, dtype=np.float32)


class BondFeaturizer:
    """化学键特征提取器"""

    def __init__(self):
        self.bond_types = BOND_TYPES
        self.stereo_types = STEREO_TYPES

    @property
    def feature_dim(self) -> int:
        """返回键特征维度"""
        # bond_type(4+1) + conjugated(1) + in_ring(1) + stereo(6+1) = 12
        return 5 + 1 + 1 + 5  # = 12

    def __call__(self, bond: Chem.Bond) -> np.ndarray:
        """提取单个键的特征"""
        features = []

        # 键类型 (4 + 1 = 5)
        bond_type = bond.GetBondType()
        features.extend(one_hot_encoding(bond_type, self.bond_types))

        # 是否共轭 (1维)
        features.append(1 if bond.GetIsConjugated() else 0)

        # 是否在环中 (1维)
        features.append(1 if bond.IsInRing() else 0)

        # 立体化学 (6 + 1 = 7) - 简化为5维
        stereo = bond.GetStereo()
        stereo_feat = [0] * 5
        if stereo == Chem.rdchem.BondStereo.STEREONONE:
            stereo_feat[0] = 1
        elif stereo == Chem.rdchem.BondStereo.STEREOZ or stereo == Chem.rdchem.BondStereo.STEREOCIS:
            stereo_feat[1] = 1
        elif stereo == Chem.rdchem.BondStereo.STEREOE or stereo == Chem.rdchem.BondStereo.STEREOTRANS:
            stereo_feat[2] = 1
        elif stereo == Chem.rdchem.BondStereo.STEREOANY:
            stereo_feat[3] = 1
        else:
            stereo_feat[4] = 1
        features.extend(stereo_feat)

        return np.array(features, dtype=np.float32)


class Geometry3DFeaturizer:
    """
    3D几何特征提取器
    提取分子的3D距离和几何特征作为边的附加特征
    """

    def __init__(self,
                 rbf_centers: List[float] = [0.5, 1.0, 1.5, 2.0],
                 rbf_sigma: float = 0.5,
                 max_distance: float = 10.0,
                 force_field: str = "MMFF"):
        """
        初始化3D几何特征提取器

        Args:
            rbf_centers: RBF编码的中心点 (Angstrom)
            rbf_sigma: RBF编码的标准差
            max_distance: 最大距离阈值用于归一化
            force_field: 力场类型 ("MMFF" 或 "UFF")
        """
        self.rbf_centers = np.array(rbf_centers)
        self.rbf_sigma = rbf_sigma
        self.max_distance = max_distance
        self.force_field = force_field

    @property
    def feature_dim(self) -> int:
        """返回3D边特征维度: 1(距离) + len(rbf_centers)(RBF) + 3(距离bin)"""
        return 1 + len(self.rbf_centers) + 3  # = 8 by default

    def generate_3d_coords(self, mol: Chem.Mol, num_conformers: int = 1,
                          random_seed: int = 42) -> Optional[np.ndarray]:
        """
        生成分子的3D坐标

        Args:
            mol: RDKit分子对象
            num_conformers: 生成的构象数量
            random_seed: 随机种子

        Returns:
            3D坐标数组 shape: (num_atoms, 3) 或 None（如果失败）
        """
        mol_with_h = Chem.AddHs(mol)

        # 尝试方法1: ETKDG + MMFF优化
        try:
            result = AllChem.EmbedMolecule(mol_with_h,
                                           AllChem.ETKDGv3(),
                                           randomSeed=random_seed)
            if result == 0:  # 成功
                if self.force_field == "MMFF":
                    try:
                        AllChem.MMFFOptimizeMolecule(mol_with_h)
                    except:
                        pass
                elif self.force_field == "UFF":
                    try:
                        AllChem.UFFOptimizeMolecule(mol_with_h)
                    except:
                        pass

                # 获取非氢原子的坐标
                conf = mol_with_h.GetConformer()
                positions = []
                for atom in mol_with_h.GetAtoms():
                    if atom.GetAtomicNum() != 1:  # 非氢原子
                        pos = conf.GetAtomPosition(atom.GetIdx())
                        positions.append([pos.x, pos.y, pos.z])
                return np.array(positions, dtype=np.float32)
        except:
            pass

        # 尝试方法2: ETKDG + UFF优化 (如果MMFF失败)
        try:
            mol_with_h = Chem.AddHs(mol)
            result = AllChem.EmbedMolecule(mol_with_h, randomSeed=random_seed)
            if result == 0:
                try:
                    AllChem.UFFOptimizeMolecule(mol_with_h)
                except:
                    pass

                conf = mol_with_h.GetConformer()
                positions = []
                for atom in mol_with_h.GetAtoms():
                    if atom.GetAtomicNum() != 1:
                        pos = conf.GetAtomPosition(atom.GetIdx())
                        positions.append([pos.x, pos.y, pos.z])
                return np.array(positions, dtype=np.float32)
        except:
            pass

        # 尝试方法3: 仅ETKDG (无优化)
        try:
            mol_with_h = Chem.AddHs(mol)
            result = AllChem.EmbedMolecule(mol_with_h, randomSeed=random_seed)
            if result == 0:
                conf = mol_with_h.GetConformer()
                positions = []
                for atom in mol_with_h.GetAtoms():
                    if atom.GetAtomicNum() != 1:
                        pos = conf.GetAtomPosition(atom.GetIdx())
                        positions.append([pos.x, pos.y, pos.z])
                return np.array(positions, dtype=np.float32)
        except:
            pass

        # 所有3D方法都失败，尝试2D fallback
        try:
            mol_copy = Chem.Mol(mol)  # 复制分子（不添加氢）
            AllChem.Compute2DCoords(mol_copy)
            conf = mol_copy.GetConformer()
            positions = []
            for atom in mol_copy.GetAtoms():
                pos = conf.GetAtomPosition(atom.GetIdx())
                # 2D坐标，z坐标设为0
                positions.append([pos.x, pos.y, 0.0])
            # 成功生成2D坐标
            return np.array(positions, dtype=np.float32)
        except Exception as e:
            # 2D也失败了，返回None
            return None

    def compute_rbf_encoding(self, distance: float) -> np.ndarray:
        """
        使用径向基函数(RBF)编码距离

        Args:
            distance: 原子间3D距离 (Angstrom)

        Returns:
            RBF编码向量
        """
        return np.exp(-((distance - self.rbf_centers) ** 2) / (2 * self.rbf_sigma ** 2))

    def compute_distance_bin(self, distance: float) -> List[int]:
        """
        计算距离的bin编码 (短/中/长距离)

        Args:
            distance: 原子间3D距离 (Angstrom)

        Returns:
            3维one-hot向量 [短距离(<1.6Å), 中距离(1.6-2.5Å), 长距离(>2.5Å)]
        """
        if distance < 1.6:
            return [1, 0, 0]
        elif distance < 2.5:
            return [0, 1, 0]
        else:
            return [0, 0, 1]

    def get_3d_edge_features(self, positions: np.ndarray,
                              edge_index: np.ndarray) -> np.ndarray:
        """
        计算边的3D几何特征

        Args:
            positions: 原子3D坐标 shape: (num_atoms, 3)
            edge_index: 边索引 shape: (2, num_edges)

        Returns:
            3D边特征 shape: (num_edges, feature_dim)
        """
        num_edges = edge_index.shape[1]
        num_atoms = positions.shape[0]
        features_3d = np.zeros((num_edges, self.feature_dim), dtype=np.float32)

        for i in range(num_edges):
            src, dst = edge_index[0, i], edge_index[1, i]

            # 边界检查：确保索引在有效范围内
            if src >= num_atoms or dst >= num_atoms:
                # 如果索引越界，使用零填充（表示无效的3D特征）
                # 这种情况可能发生在3D坐标生成不完整时
                continue

            # 计算3D距离
            distance = np.linalg.norm(positions[src] - positions[dst])

            # 归一化距离
            norm_distance = min(distance / self.max_distance, 1.0)

            # RBF编码
            rbf_features = self.compute_rbf_encoding(distance)

            # 距离bin编码
            distance_bin = self.compute_distance_bin(distance)

            # 组合特征: [归一化距离, RBF编码, 距离bin]
            features_3d[i, 0] = norm_distance
            features_3d[i, 1:1+len(self.rbf_centers)] = rbf_features
            features_3d[i, 1+len(self.rbf_centers):] = distance_bin

        return features_3d

    def get_zero_features(self, num_edges: int) -> np.ndarray:
        """
        返回零填充的3D特征（用于3D生成失败的情况）

        Args:
            num_edges: 边的数量

        Returns:
            零填充的3D边特征 shape: (num_edges, feature_dim)
        """
        return np.zeros((num_edges, self.feature_dim), dtype=np.float32)


def mol_to_graph_data(smiles: str,
                      atom_featurizer: Optional[AtomFeaturizer] = None,
                      bond_featurizer: Optional[BondFeaturizer] = None,
                      geom_featurizer: Optional[Geometry3DFeaturizer] = None,
                      use_3d_features: bool = True) -> Optional[Tuple]:
    """
    将SMILES转换为图数据

    Args:
        smiles: 分子的SMILES表示
        atom_featurizer: 原子特征提取器
        bond_featurizer: 键特征提取器
        geom_featurizer: 3D几何特征提取器
        use_3d_features: 是否使用3D特征

    Returns:
        tuple: (node_features, edge_index, edge_features, positions, has_3d)
               或 None（如果解析失败）
               - node_features: 原子特征 shape: (num_atoms, 82)
               - edge_index: 边索引 shape: (2, num_edges)
               - edge_features: 边特征 shape: (num_edges, 12+8=20) 如果use_3d_features=True
               - positions: 3D坐标 shape: (num_atoms, 3) 或 None
               - has_3d: 是否成功生成3D坐标
    """
    if atom_featurizer is None:
        atom_featurizer = AtomFeaturizer()
    if bond_featurizer is None:
        bond_featurizer = BondFeaturizer()
    if geom_featurizer is None and use_3d_features:
        geom_featurizer = Geometry3DFeaturizer()

    # 解析SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.RemoveHs(mol)

    # 提取原子特征
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        return None

    node_features = []
    for atom in mol.GetAtoms():
        node_features.append(atom_featurizer(atom))
    node_features = np.stack(node_features, axis=0)

    # 提取边信息（双向边）
    edge_index = []
    edge_features = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_feat = bond_featurizer(bond)

        # 添加双向边
        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_features.append(bond_feat)
        edge_features.append(bond_feat)

    if len(edge_index) == 0:
        # 处理单原子分子
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_features = np.zeros((0, bond_featurizer.feature_dim), dtype=np.float32)
        positions = None
        has_3d = False
    else:
        edge_index = np.array(edge_index, dtype=np.int64).T
        edge_features = np.stack(edge_features, axis=0)

        # 生成3D坐标和特征
        positions = None
        has_3d = False
        if use_3d_features and geom_featurizer is not None:
            positions = geom_featurizer.generate_3d_coords(mol)
            if positions is not None:
                has_3d = True
                # 计算3D边特征
                features_3d = geom_featurizer.get_3d_edge_features(positions, edge_index)
            else:
                # 3D生成失败，使用零填充
                features_3d = geom_featurizer.get_zero_features(edge_features.shape[0])

            # 将3D特征拼接到边特征
            edge_features = np.concatenate([edge_features, features_3d], axis=1)

    return node_features, edge_index, edge_features, positions, has_3d


def get_feature_dims(use_3d_features: bool = True) -> Tuple[int, int]:
    """
    获取原子和键特征的维度

    Args:
        use_3d_features: 是否包含3D特征

    Returns:
        tuple: (原子特征维度, 边特征维度)
    """
    atom_featurizer = AtomFeaturizer()
    bond_featurizer = BondFeaturizer()
    bond_dim = bond_featurizer.feature_dim

    if use_3d_features:
        geom_featurizer = Geometry3DFeaturizer()
        bond_dim += geom_featurizer.feature_dim  # 12 + 8 = 20

    return atom_featurizer.feature_dim, bond_dim


if __name__ == "__main__":
    # 测试特征提取
    test_smiles = "CCO"  # 乙醇
    print("=" * 50)
    print("测试3D几何特征提取")
    print("=" * 50)

    # 测试不使用3D特征
    result = mol_to_graph_data(test_smiles, use_3d_features=False)
    if result:
        node_feat, edge_idx, edge_feat, positions, has_3d = result
        print(f"\n[不使用3D特征]")
        print(f"SMILES: {test_smiles}")
        print(f"Node features shape: {node_feat.shape}")
        print(f"Edge index shape: {edge_idx.shape}")
        print(f"Edge features shape: {edge_feat.shape}")
        print(f"Has 3D coords: {has_3d}")

    # 测试使用3D特征
    result = mol_to_graph_data(test_smiles, use_3d_features=True)
    if result:
        node_feat, edge_idx, edge_feat, positions, has_3d = result
        print(f"\n[使用3D特征]")
        print(f"SMILES: {test_smiles}")
        print(f"Node features shape: {node_feat.shape}")
        print(f"Edge index shape: {edge_idx.shape}")
        print(f"Edge features shape: {edge_feat.shape} (12 + 8 = 20维)")
        print(f"Has 3D coords: {has_3d}")
        if positions is not None:
            print(f"3D positions shape: {positions.shape}")
            print(f"3D positions:\n{positions}")

    print(f"\n[特征维度]")
    print(f"Atom feature dim: {AtomFeaturizer().feature_dim}")
    print(f"Bond feature dim (2D only): {BondFeaturizer().feature_dim}")
    print(f"Geometry 3D feature dim: {Geometry3DFeaturizer().feature_dim}")
    print(f"Total edge dim with 3D: {get_feature_dims(use_3d_features=True)[1]}")
