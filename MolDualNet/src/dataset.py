"""
数据集模块
Dataset Module for Molecular Property Prediction
"""

import os
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader as TorchDataLoader
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import train_test_split

from rdkit import Chem
from .features import mol_to_graph_data, AtomFeaturizer, BondFeaturizer, Geometry3DFeaturizer
from .tokenizer import SmilesTokenizer


def _ensure_numpy_core_alias() -> None:
    try:
        import numpy._core  # type: ignore
    except Exception:
        import numpy.core as _np_core
        import sys
        sys.modules.setdefault("numpy._core", _np_core)


class MoleculeDataset(Dataset):
    """
    分子性质预测数据集

    同时提供分子图数据和SMILES序列数据
    """

    def __init__(self,
                 smiles_list: List[str],
                 targets: Dict[str, np.ndarray],
                 tokenizer: SmilesTokenizer,
                 task_config: Dict[str, Any],
                 transform=None,
                 use_3d_features: bool = True,
                 geometry_3d_config: Optional[Dict] = None,
                 coords_cache_path: Optional[str] = None,
                 expert_features_path: Optional[str] = None,
                 random_smiles_prob: float = 0.0,
                 is_train: bool = False):
        """
        Args:
            smiles_list: SMILES字符串列表
            targets: 目标值字典 {task_name: values} (可以包含NaN)
            tokenizer: SMILES分词器
            task_config: 任务配置
            transform: 数据变换
            use_3d_features: 是否使用3D几何特征
            geometry_3d_config: 3D几何特征配置
            coords_cache_path: 3D坐标缓存文件路径
            expert_features_path: 专家特征缓存文件路径
            random_smiles_prob: Random SMILES增强概率（仅训练集）
            is_train: 是否训练集
        """
        self.smiles_list = smiles_list
        self.targets = targets
        self.tokenizer = tokenizer
        self.task_config = task_config
        self.transform = transform
        self.use_3d_features = use_3d_features
        self.random_smiles_prob = max(0.0, float(random_smiles_prob))
        self.is_train = is_train

        # 任务可用性掩码：NaN标记为False，有效值标记为True
        self.task_masks = {}
        for task_name, values in targets.items():
            self.task_masks[task_name] = ~np.isnan(values)

        # 特征提取器
        self.atom_featurizer = AtomFeaturizer()
        self.bond_featurizer = BondFeaturizer()

        # 3D几何特征提取器
        if use_3d_features:
            geom_config = geometry_3d_config or {}
            self.geom_featurizer = Geometry3DFeaturizer(
                rbf_centers=geom_config.get('distance_rbf_centers', [0.5, 1.0, 1.5, 2.0]),
                rbf_sigma=geom_config.get('distance_rbf_sigma', 0.5),
                max_distance=geom_config.get('max_distance', 10.0),
                force_field=geom_config.get('force_field', 'MMFF')
            )
        else:
            self.geom_featurizer = None

        # 加载3D坐标缓存
        self.coords_cache = None
        if coords_cache_path and os.path.exists(coords_cache_path):
            print(f"Loading 3D coordinates cache from: {coords_cache_path}")
            import pickle
            _ensure_numpy_core_alias()
            with open(coords_cache_path, 'rb') as f:
                self.coords_cache = pickle.load(f)
            print(f"Loaded cache with {len(self.coords_cache)} entries")
        elif coords_cache_path:
            print(f"Warning: Cache file not found: {coords_cache_path}")
            print("Will compute 3D coordinates on-the-fly (slower)")

        # 加载专家特征缓存 (Morgan FP + RDKit Descriptors)
        self.expert_features = None
        self.expert_smiles_to_idx = None
        if expert_features_path and os.path.exists(expert_features_path):
            print(f"Loading expert features from: {expert_features_path}")
            import pickle
            _ensure_numpy_core_alias()
            with open(expert_features_path, 'rb') as f:
                expert_data = pickle.load(f)
            self.expert_features = expert_data['features']  # [N, 1033]
            expert_smiles = expert_data['smiles']
            # 创建SMILES到索引的映射
            self.expert_smiles_to_idx = {smi: i for i, smi in enumerate(expert_smiles)}
            print(f"Loaded {len(self.expert_features)} expert features (dim={expert_data.get('total_dim', 1033)})")
        elif expert_features_path:
            print(f"Warning: Expert features file not found: {expert_features_path}")
            print("Expert features will not be used")

        # 3D生成统计
        self.num_3d_success = 0
        self.num_3d_failed = 0

        # 预处理：过滤无效SMILES
        self._preprocess()

    def _maybe_augment_smiles(self, smiles: str) -> str:
        """可选的Random SMILES增强（仅训练集）"""
        if not self.is_train or self.random_smiles_prob <= 0.0:
            return smiles
        if random.random() >= self.random_smiles_prob:
            return smiles
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return smiles
            # 生成随机SMILES
            return Chem.MolToSmiles(mol, doRandom=True, canonical=False)
        except Exception:
            return smiles

    def _preprocess(self):
        """预处理数据，过滤无效SMILES"""
        valid_indices = []

        for i, smiles in enumerate(self.smiles_list):
            # 不使用3D特征进行预处理检查（提高速度）
            graph_data = mol_to_graph_data(
                smiles,
                self.atom_featurizer,
                self.bond_featurizer,
                geom_featurizer=None,
                use_3d_features=False
            )
            if graph_data is not None:
                valid_indices.append(i)

        if len(valid_indices) < len(self.smiles_list):
            print(f"Filtered {len(self.smiles_list) - len(valid_indices)} invalid SMILES")

            self.smiles_list = [self.smiles_list[i] for i in valid_indices]
            for key in self.targets:
                self.targets[key] = self.targets[key][valid_indices]

    def __len__(self) -> int:
        return len(self.smiles_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        smiles = self.smiles_list[idx]
        token_smiles = self._maybe_augment_smiles(smiles)

        # 检查是否有缓存的3D坐标
        cached_positions = None
        if self.coords_cache is not None and smiles in self.coords_cache:
            cached_positions = self.coords_cache[smiles]

        # 如果有缓存，先不使用3D特征获取图数据，然后添加缓存的坐标
        if cached_positions is not None:
            # 不使用3D特征获取基本图数据（更快）
            result = mol_to_graph_data(
                smiles,
                self.atom_featurizer,
                self.bond_featurizer,
                geom_featurizer=None,
                use_3d_features=False
            )
            node_features, edge_index, edge_features, _, _ = result

            # 使用缓存的坐标
            positions = cached_positions
            has_3d = True

            # 如果需要3D特征，计算边的3D特征
            if self.use_3d_features and self.geom_featurizer is not None:
                features_3d = self.geom_featurizer.get_3d_edge_features(positions, edge_index)
                edge_features = np.concatenate([edge_features, features_3d], axis=1)
        else:
            # 没有缓存，实时计算（回退方案）
            result = mol_to_graph_data(
                smiles,
                self.atom_featurizer,
                self.bond_featurizer,
                geom_featurizer=self.geom_featurizer,
                use_3d_features=self.use_3d_features
            )
            node_features, edge_index, edge_features, positions, has_3d = result

        # 更新3D生成统计
        if self.use_3d_features:
            if has_3d:
                self.num_3d_success += 1
            else:
                self.num_3d_failed += 1

        # 创建PyG Data对象
        num_nodes = node_features.shape[0]

        # 处理3D坐标：如果为None，使用零填充以保持批次一致性
        if positions is not None:
            pos_tensor = torch.tensor(positions, dtype=torch.float)
        else:
            # 3D生成失败时使用零坐标
            pos_tensor = torch.zeros((num_nodes, 3), dtype=torch.float)

        graph_data = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_features, dtype=torch.float),
            pos=pos_tensor,
            has_3d=torch.tensor([1 if has_3d else 0], dtype=torch.long),  # 转换为tensor以支持批处理
        )

        # 获取SMILES序列数据
        encoded = self.tokenizer.encode(token_smiles)
        input_ids = torch.tensor(encoded['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(encoded['attention_mask'], dtype=torch.long)

        # 获取目标值和掩码
        target_dict = {}
        mask_dict = {}
        for task_name, values in self.targets.items():
            value = values[idx]
            mask = self.task_masks[task_name][idx]

            # 如果是NaN，用0填充（实际不会用于损失计算）
            if np.isnan(value):
                value = 0.0

            target_dict[f'{task_name}_value'] = torch.tensor(float(value), dtype=torch.float)
            mask_dict[f'{task_name}_mask'] = torch.tensor(bool(mask), dtype=torch.bool)

        if self.transform:
            graph_data = self.transform(graph_data)

        # 获取专家特征(如果可用)
        expert_feature = None
        if self.expert_features is not None and smiles in self.expert_smiles_to_idx:
            expert_idx = self.expert_smiles_to_idx[smiles]
            expert_feature = torch.tensor(self.expert_features[expert_idx], dtype=torch.float)

        return {
            'graph': graph_data,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'expert_feature': expert_feature,  # 新增：专家特征
            'targets': target_dict,
            'masks': mask_dict,  # 新增：任务可用性掩码
            'smiles': smiles,
        }

    def get_3d_stats(self) -> Dict[str, int]:
        """获取3D生成统计信息"""
        return {
            'success': self.num_3d_success,
            'failed': self.num_3d_failed,
            'success_rate': self.num_3d_success / max(1, self.num_3d_success + self.num_3d_failed)
        }


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    自定义collate函数，处理图数据和序列数据的批次化（支持缺失值）
    """
    # 分离图数据和序列数据
    graphs = [item['graph'] for item in batch]
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    smiles = [item['smiles'] for item in batch]

    # 批次化专家特征(如果可用)
    expert_features = None
    if batch[0]['expert_feature'] is not None:
        expert_features = torch.stack([item['expert_feature'] for item in batch])

    # 批次化图数据
    graph_batch = Batch.from_data_list(graphs)

    # 批次化目标值和掩码
    targets = {}
    masks = {}

    # 收集所有可能的键（处理不同样本可能有不同任务的情况）
    all_target_keys = set()
    all_mask_keys = set()
    for item in batch:
        all_target_keys.update(item['targets'].keys())
        all_mask_keys.update(item['masks'].keys())

    # 为每个键创建批次，缺失的用0和False填充
    for key in all_target_keys:
        target_list = []
        for item in batch:
            if key in item['targets']:
                target_list.append(item['targets'][key])
            else:
                # 缺失任务，用0填充（配合mask=False）
                target_list.append(torch.tensor(0.0, dtype=torch.float))
        targets[key] = torch.stack(target_list)

    for key in all_mask_keys:
        mask_list = []
        for item in batch:
            if key in item['masks']:
                mask_list.append(item['masks'][key])
            else:
                mask_list.append(torch.tensor(False, dtype=torch.bool))
        masks[key] = torch.stack(mask_list)

    return {
        'graph': graph_batch,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'expert_features': expert_features,  # 新增：专家特征批次
        'targets': targets,
        'masks': masks,  # 新增：任务可用性掩码批次
        'smiles': smiles,
    }


def load_dataset(config: Dict[str, Any]) -> Tuple[Dict[str, List], Dict[str, np.ndarray], Dict[str, Any]]:
    """
    从配置加载数据集（支持merged_dataset.csv格式）

    Args:
        config: 配置字典

    Returns:
        smiles_dict: {split: smiles_list}
        targets_dict: {split: {task: values}}  # 可以包含NaN
        task_config: 任务配置
    """
    base_path = config['data'].get('base_path', '.')
    tasks = config['tasks']

    # 检查是否使用merged_file格式
    merged_file = config['data'].get('merged_file', None)

    if merged_file:
        # 新格式：从merged_dataset.csv加载
        file_path = os.path.join(base_path, merged_file)
        df = pd.read_csv(file_path)

        print(f"Loading from merged dataset: {file_path}")
        print(f"Total molecules: {len(df)}")

        # 提取SMILES
        all_smiles = df['smiles'].tolist()

        # 提取所有任务的目标值
        all_targets = {}
        task_config = {}

        for task_name, task_cfg in tasks.items():
            if task_name in df.columns:
                # 保留NaN值
                all_targets[task_name] = df[task_name].values.astype(np.float32)
                task_config[task_name] = task_cfg

                # 统计有效值数量
                valid_count = (~np.isnan(all_targets[task_name])).sum()
                coverage = valid_count / len(df) * 100
                print(f"  Task {task_name}: {valid_count}/{len(df)} ({coverage:.1f}%) valid values")
            else:
                print(f"  Warning: Task {task_name} not found in dataset, skipping")

    else:
        # 旧格式：从多个任务文件加载
        all_smiles = None
        all_targets = {}
        task_config = {}

        for task_name, task_cfg in tasks.items():
            file_path = os.path.join(base_path, task_cfg['file'])
            df = pd.read_csv(file_path)

            smiles_col = task_cfg.get('smiles_col', 'Smiles')
            target_col = task_cfg['target_col']

            # 检查列是否存在
            if smiles_col not in df.columns or target_col not in df.columns:
                raise ValueError(f"Required columns not found in {file_path}")

            smiles = df[smiles_col].tolist()
            values = df[target_col].values.astype(np.float32)

            if all_smiles is None:
                all_smiles = smiles
            else:
                # 确保SMILES一致（多任务情况）
                if len(smiles) != len(all_smiles):
                    print(f"Warning: Task {task_name} has different number of samples")

            all_targets[task_name] = values
            task_config[task_name] = task_cfg

    # 数据划分
    train_ratio = config['data'].get('train_ratio', 0.8)
    val_ratio = config['data'].get('val_ratio', 0.1)
    random_seed = config['data'].get('random_seed', 42)

    n_samples = len(all_smiles)
    indices = np.arange(n_samples)

    # 第一次划分：训练+验证 vs 测试
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=1 - train_ratio - val_ratio,
        random_state=random_seed
    )

    # 第二次划分：训练 vs 验证
    val_size = val_ratio / (train_ratio + val_ratio)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size,
        random_state=random_seed
    )

    # 构建划分后的数据
    smiles_dict = {
        'train': [all_smiles[i] for i in train_idx],
        'val': [all_smiles[i] for i in val_idx],
        'test': [all_smiles[i] for i in test_idx],
    }

    targets_dict = {
        'train': {task: values[train_idx] for task, values in all_targets.items()},
        'val': {task: values[val_idx] for task, values in all_targets.items()},
        'test': {task: values[test_idx] for task, values in all_targets.items()},
    }

    print(f"Dataset split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    return smiles_dict, targets_dict, task_config


def create_data_loaders(config: Dict[str, Any],
                       tokenizer: Optional[SmilesTokenizer] = None
                       ) -> Tuple[TorchDataLoader, TorchDataLoader, TorchDataLoader, SmilesTokenizer]:
    """
    创建数据加载器

    Args:
        config: 配置字典
        tokenizer: SMILES分词器（可选）

    Returns:
        train_loader, val_loader, test_loader, tokenizer
    """
    # 加载数据
    smiles_dict, targets_dict, task_config = load_dataset(config)

    # 获取3D几何特征配置
    geometry_3d_config = config.get('model', {}).get('geometry_3d', {})
    use_3d_features = geometry_3d_config.get('enabled', True)

    # 获取3D坐标缓存路径
    coords_cache_path = geometry_3d_config.get('coords_cache_path', None)

    # 获取专家特征配置
    expert_config = config.get('model', {}).get('expert_features', {})
    expert_features_path = expert_config.get('features_cache_path', None)

    # Random SMILES增强配置（仅训练集）
    smiles_aug_config = config.get('data', {}).get('smiles_augmentation', {})
    smiles_aug_enabled = smiles_aug_config.get('enabled', False)
    random_smiles_prob = float(smiles_aug_config.get('random_smiles_prob', 0.0))
    if not smiles_aug_enabled:
        random_smiles_prob = 0.0

    if use_3d_features:
        print("3D geometric features enabled")
        if coords_cache_path:
            print(f"3D coordinates cache: {coords_cache_path}")
    else:
        print("3D geometric features disabled")

    if expert_config.get('enabled', False):
        print("Expert features (Morgan FP + RDKit Descriptors) enabled")
        if expert_features_path:
            print(f"Expert features cache: {expert_features_path}")
    if random_smiles_prob > 0:
        print(f"Random SMILES augmentation enabled (p={random_smiles_prob:.2f})")

    # 创建或使用分词器
    if tokenizer is None:
        tokenizer = SmilesTokenizer(
            max_length=config['model']['transformer'].get('max_seq_len', 256)
        )
        # 从训练数据构建词表
        tokenizer.build_vocab_from_data(smiles_dict['train'])

    # 创建数据集
    train_dataset = MoleculeDataset(
        smiles_dict['train'],
        targets_dict['train'],
        tokenizer,
        task_config,
        use_3d_features=use_3d_features,
        geometry_3d_config=geometry_3d_config,
        coords_cache_path=coords_cache_path,
        expert_features_path=expert_features_path,
        random_smiles_prob=random_smiles_prob,
        is_train=True
    )

    val_dataset = MoleculeDataset(
        smiles_dict['val'],
        targets_dict['val'],
        tokenizer,
        task_config,
        use_3d_features=use_3d_features,
        geometry_3d_config=geometry_3d_config,
        coords_cache_path=coords_cache_path,
        expert_features_path=expert_features_path,
        random_smiles_prob=0.0,
        is_train=False
    )

    test_dataset = MoleculeDataset(
        smiles_dict['test'],
        targets_dict['test'],
        tokenizer,
        task_config,
        use_3d_features=use_3d_features,
        geometry_3d_config=geometry_3d_config,
        coords_cache_path=coords_cache_path,
        expert_features_path=expert_features_path,
        random_smiles_prob=0.0,
        is_train=False
    )

    # 创建数据加载器
    batch_size = config['training'].get('batch_size', 32)
    num_workers = config.get('training', {}).get('num_workers')
    if num_workers is None:
        if torch.cuda.is_available():
            cpu_count = os.cpu_count() or 0
            num_workers = max(2, min(8, cpu_count // 2)) if cpu_count else 4
        else:
            num_workers = 0
    pin_memory = config.get('training', {}).get('pin_memory', torch.cuda.is_available())

    train_loader = TorchDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = TorchDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = TorchDataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, tokenizer


if __name__ == "__main__":
    # 测试数据集
    from .utils import load_config

    config = load_config("config.yaml")

    train_loader, val_loader, test_loader, tokenizer = create_data_loaders(config)

    print(f"\nTokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # 测试一个batch
    for batch in train_loader:
        print(f"\nBatch contents:")
        print(f"  Graph: {batch['graph']}")
        print(f"  Input IDs shape: {batch['input_ids'].shape}")
        print(f"  Attention mask shape: {batch['attention_mask'].shape}")
        print(f"  Targets: {list(batch['targets'].keys())}")
        break
