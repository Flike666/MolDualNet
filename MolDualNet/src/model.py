"""
主模型模块
Main Model Module - Combines GNN and Transformer for Molecular Property Prediction
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any

from .gnn import GNNEncoder
from .transformer import TransformerEncoder
from .cross_attention import CrossAttentionFusion
from .expert_features import ExpertFeatureEncoder


class TaskHead(nn.Module):
    """
    任务头模块

    支持回归和/或分类输出
    """

    def __init__(self,
                 input_dim: int,
                 task_type: str = 'regression',
                 hidden_dim: int = 128,
                 dropout: float = 0.2):
        """
        Args:
            input_dim: 输入特征维度
            task_type: 任务类型 ('regression', 'classification', 'regression+classification')
            hidden_dim: 隐藏层维度
            dropout: dropout比率
        """
        super().__init__()

        if 'classification' in task_type:
            raise ValueError("Classification tasks are disabled in this build")
        self.task_type = task_type

        # 共享的MLP层
        self.shared_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 回归头
        if 'regression' in task_type:
            self.regression_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )


    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入特征 [batch_size, input_dim]

        Returns:
            dict: 包含回归值和/或分类logits
        """
        shared = self.shared_mlp(x)
        outputs = {}

        if 'regression' in self.task_type:
            outputs['regression'] = self.regression_head(shared).squeeze(-1)

        return outputs


class MoleculePropertyPredictor(nn.Module):
    """
    分子性质预测主模型

    双流架构：GNN处理分子图 + Transformer处理SMILES序列
    支持多任务预测，支持交叉注意力融合
    """

    def __init__(self, config: Dict[str, Any], tasks: Dict[str, Dict]):
        """
        Args:
            config: 模型配置字典
            tasks: 任务配置字典 {task_name: task_config}
        """
        super().__init__()

        self.config = config
        self.tasks = tasks

        # GNN配置
        gnn_config = config['gnn']
        self.gnn_hidden_dim = gnn_config.get('hidden_dim', 128)
        self.gnn_encoder = GNNEncoder(
            node_input_dim=gnn_config.get('node_input_dim', 78),
            edge_input_dim=gnn_config.get('edge_input_dim', 12),
            hidden_dim=self.gnn_hidden_dim,
            output_dim=gnn_config.get('output_dim', 128),
            num_layers=gnn_config.get('num_layers', 4),
            num_heads=gnn_config.get('num_heads', 4),
            dropout=gnn_config.get('dropout', 0.1),
        )

        # Transformer配置
        trans_config = config['transformer']
        self.trans_embed_dim = trans_config.get('embed_dim', 128)
        self.transformer_encoder = TransformerEncoder(
            vocab_size=trans_config.get('vocab_size', 100),
            max_seq_len=trans_config.get('max_seq_len', 256),
            embed_dim=self.trans_embed_dim,
            num_heads=trans_config.get('num_heads', 4),
            num_layers=trans_config.get('num_layers', 4),
            dropout=trans_config.get('dropout', 0.1),
            output_dim=trans_config.get('output_dim', 128),
        )

        # 专家特征配置 (Morgan FP + RDKit Descriptors)
        expert_config = config.get('expert_features', {})
        self.use_expert_features = expert_config.get('enabled', False)
        if self.use_expert_features:
            self.expert_encoder = ExpertFeatureEncoder(
                input_dim=expert_config.get('input_dim', 1033),  # 1024 + 9
                output_dim=expert_config.get('output_dim', 256),
                dropout=expert_config.get('dropout', 0.1),
                use_precomputed=expert_config.get('use_precomputed', True),
            )
            expert_output_dim = expert_config.get('output_dim', 256)
        else:
            self.expert_encoder = None
            expert_output_dim = 0

        # 融合层配置
        fusion_config = config.get('fusion', {})
        gnn_output_dim = gnn_config.get('output_dim', 128)
        trans_output_dim = trans_config.get('output_dim', 128)
        fusion_hidden_dim = fusion_config.get('hidden_dim', 256)
        fusion_dropout = fusion_config.get('dropout', 0.2)

        # Fusion mode
        self.use_cross_attention = fusion_config.get('use_cross_attention', False)
        self.allow_cross_attention_toggle = fusion_config.get('allow_cross_attention_toggle', False)
        self.modality_dropout_p = float(fusion_config.get('modality_dropout_p', 0.0))
        self.modality_dropout_keep_at_least_one = fusion_config.get(
            'modality_dropout_keep_at_least_one', True
        )

        build_cross_attention = self.use_cross_attention or self.allow_cross_attention_toggle
        build_simple_fusion = (not self.use_cross_attention) or self.allow_cross_attention_toggle

        if build_cross_attention:
            # Cross-attention fusion
            self.cross_attention_fusion = CrossAttentionFusion(
                gnn_dim=self.gnn_hidden_dim,
                trans_dim=self.trans_embed_dim,
                hidden_dim=fusion_hidden_dim,
                num_layers=fusion_config.get('cross_attention_layers', 2),
                num_heads=fusion_config.get('cross_attention_heads', 4),
                dropout=fusion_dropout,
                output_dim=fusion_hidden_dim,
            )
        else:
            self.cross_attention_fusion = None

        if build_simple_fusion:
            # Simple concat fusion (supports Expert Features)
            fusion_input_dim = gnn_output_dim + trans_output_dim + expert_output_dim
            self.fusion_layer = nn.Sequential(
                nn.Linear(fusion_input_dim, fusion_hidden_dim),
                nn.LayerNorm(fusion_hidden_dim),
                nn.GELU(),
                nn.Dropout(fusion_dropout),
                nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
                nn.LayerNorm(fusion_hidden_dim),
                nn.GELU(),
                nn.Dropout(fusion_dropout),
            )
        else:
            self.fusion_layer = None

        self.task_heads = nn.ModuleDict()
        for task_name, task_config in tasks.items():
            task_type = task_config.get('task_type', 'regression')
            self.task_heads[task_name] = TaskHead(
                input_dim=fusion_hidden_dim,
                task_type=task_type,
                hidden_dim=fusion_hidden_dim // 2,
                dropout=fusion_dropout,
            )

    def forward(self,
                graph_batch,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                expert_features: Optional[torch.Tensor] = None,
                smiles: Optional[list] = None) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        前向传播

        Args:
            graph_batch: PyG Batch对象，包含分子图数据
            input_ids: SMILES token索引 [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            expert_features: 预计算的专家特征 [batch, 1033] (可选)
            smiles: SMILES字符串列表 (可选,实时计算时需要)

        Returns:
            dict: {task_name: {output_type: tensor}}
        """
        drop_flags = self._sample_modality_dropout()

        if self.use_cross_attention:
            if self.cross_attention_fusion is None:
                raise ValueError("cross_attention_fusion is not initialized")
            # ?????????
            # ??GNN????
            gnn_output, node_features = self.gnn_encoder.forward_with_node_features(
                x=graph_batch.x,
                edge_index=graph_batch.edge_index,
                edge_attr=graph_batch.edge_attr,
                batch=graph_batch.batch,
            )

            # ??Transformer????
            trans_seq_features = self._get_transformer_sequence_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            if drop_flags.get('gnn'):
                node_features = torch.zeros_like(node_features)
            if drop_flags.get('transformer'):
                trans_seq_features = torch.zeros_like(trans_seq_features)

            # ???????(GNN + Transformer)
            fused = self.cross_attention_fusion(
                gnn_features=node_features,
                trans_features=trans_seq_features,
                gnn_batch=graph_batch.batch,
                trans_mask=attention_mask,
            )
        else:
            if self.fusion_layer is None:
                raise ValueError("fusion_layer is not initialized")
            # ??????
            gnn_output = self.gnn_encoder(
                x=graph_batch.x,
                edge_index=graph_batch.edge_index,
                edge_attr=graph_batch.edge_attr,
                batch=graph_batch.batch,
            )

            trans_output = self.transformer_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            if drop_flags.get('gnn'):
                gnn_output = torch.zeros_like(gnn_output)
            if drop_flags.get('transformer'):
                trans_output = torch.zeros_like(trans_output)

            # ??????: GNN + Transformer + Expert Features
            features_to_fuse = [gnn_output, trans_output]

            if self.use_expert_features:
                expert_output = self.expert_encoder(
                    smiles=smiles,
                    precomputed_features=expert_features
                )
                if drop_flags.get('expert'):
                    expert_output = torch.zeros_like(expert_output)
                features_to_fuse.append(expert_output)

            fused = torch.cat(features_to_fuse, dim=-1)
            fused = self.fusion_layer(fused)

        outputs = {}
        for task_name, task_head in self.task_heads.items():
            outputs[task_name] = task_head(fused)

        return outputs


    def _sample_modality_dropout(self) -> Dict[str, bool]:
        # Sample modality dropout flags for the current batch.
        if (not self.training) or self.modality_dropout_p <= 0:
            return {}
        modalities = ['gnn', 'transformer']
        if (not self.use_cross_attention) and self.use_expert_features:
            modalities.append('expert')
        drop_flags = {m: (torch.rand(1).item() < self.modality_dropout_p) for m in modalities}
        if self.modality_dropout_keep_at_least_one and modalities and all(drop_flags.values()):
            keep = random.choice(modalities)
            drop_flags[keep] = False
        return drop_flags

    def set_use_cross_attention(self, enabled: bool):
        # Enable/disable cross-attention fusion at runtime.
        if enabled and self.cross_attention_fusion is None:
            raise ValueError('cross_attention_fusion is not initialized')
        if (not enabled) and self.fusion_layer is None:
            raise ValueError('fusion_layer is not initialized')
        self.use_cross_attention = enabled

    def _get_transformer_sequence_features(self,
                                           input_ids: torch.Tensor,
                                           attention_mask: torch.Tensor) -> torch.Tensor:
        """
        获取Transformer的序列级特征（不经过最终池化）

        Returns:
            seq_features: [batch, seq_len, embed_dim]
        """
        # Token嵌入
        x = self.transformer_encoder.token_embedding(input_ids)
        x = self.transformer_encoder.positional_encoding(x)

        # 创建key_padding_mask
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None

        # Transformer编码
        for encoder_layer in self.transformer_encoder.encoder_layers:
            x = encoder_layer(x, key_padding_mask=key_padding_mask)

        x = self.transformer_encoder.final_norm(x)

        return x

    def add_task(self, task_name: str, task_config: Dict):
        """
        动态添加新任务

        Args:
            task_name: 任务名称
            task_config: 任务配置
        """
        if task_name in self.task_heads:
            print(f"Warning: Task {task_name} already exists, replacing...")

        fusion_config = self.config.get('fusion', {})
        fusion_hidden_dim = fusion_config.get('hidden_dim', 256)
        fusion_dropout = fusion_config.get('dropout', 0.2)

        task_type = task_config.get('task_type', 'regression')
        self.task_heads[task_name] = TaskHead(
            input_dim=fusion_hidden_dim,
            task_type=task_type,
            hidden_dim=fusion_hidden_dim // 2,
            dropout=fusion_dropout,
        )

        self.tasks[task_name] = task_config
        print(f"Added task: {task_name} ({task_type})")

    def remove_task(self, task_name: str):
        """移除任务"""
        if task_name in self.task_heads:
            del self.task_heads[task_name]
            del self.tasks[task_name]
            print(f"Removed task: {task_name}")

    def freeze_backbone(self):
        """冻结骨干网络（用于微调）"""
        for param in self.gnn_encoder.parameters():
            param.requires_grad = False
        for param in self.transformer_encoder.parameters():
            param.requires_grad = False
        if self.use_cross_attention:
            for param in self.cross_attention_fusion.parameters():
                param.requires_grad = False
        else:
            for param in self.fusion_layer.parameters():
                param.requires_grad = False
        print("Backbone frozen")

    def unfreeze_backbone(self):
        """解冻骨干网络"""
        for param in self.gnn_encoder.parameters():
            param.requires_grad = True
        for param in self.transformer_encoder.parameters():
            param.requires_grad = True
        if self.use_cross_attention:
            for param in self.cross_attention_fusion.parameters():
                param.requires_grad = True
        else:
            for param in self.fusion_layer.parameters():
                param.requires_grad = True
        print("Backbone unfrozen")

    def get_embedding(self,
                      graph_batch,
                      input_ids: torch.Tensor,
                      attention_mask: torch.Tensor) -> torch.Tensor:
        """
        获取分子的融合嵌入表示

        Args:
            graph_batch: PyG Batch对象
            input_ids: SMILES token索引
            attention_mask: 注意力掩码

        Returns:
            embedding: [batch_size, fusion_hidden_dim]
        """
        if self.use_cross_attention:
            gnn_output, node_features = self.gnn_encoder.forward_with_node_features(
                x=graph_batch.x,
                edge_index=graph_batch.edge_index,
                edge_attr=graph_batch.edge_attr,
                batch=graph_batch.batch,
            )

            trans_seq_features = self._get_transformer_sequence_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            embedding = self.cross_attention_fusion(
                gnn_features=node_features,
                trans_features=trans_seq_features,
                gnn_batch=graph_batch.batch,
                trans_mask=attention_mask,
            )
        else:
            gnn_output = self.gnn_encoder(
                x=graph_batch.x,
                edge_index=graph_batch.edge_index,
                edge_attr=graph_batch.edge_attr,
                batch=graph_batch.batch,
            )

            trans_output = self.transformer_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            fused = torch.cat([gnn_output, trans_output], dim=-1)
            embedding = self.fusion_layer(fused)

        return embedding

    def update_vocab_size(self, new_vocab_size: int):
        """更新词表大小（需要重新初始化embedding层）"""
        old_vocab_size = self.transformer_encoder.token_embedding.num_embeddings
        embed_dim = self.transformer_encoder.embed_dim

        if new_vocab_size != old_vocab_size:
            new_embedding = nn.Embedding(new_vocab_size, embed_dim, padding_idx=0)

            # 复制旧的权重
            min_vocab = min(old_vocab_size, new_vocab_size)
            new_embedding.weight.data[:min_vocab] = \
                self.transformer_encoder.token_embedding.weight.data[:min_vocab]

            self.transformer_encoder.token_embedding = new_embedding
            self.config['transformer']['vocab_size'] = new_vocab_size
            print(f"Updated vocab size: {old_vocab_size} -> {new_vocab_size}")


class MultiTaskLoss(nn.Module):
    """
    多任务损失函数

    支持回归(MSE/Huber)和分类(BCE)损失的加权组合
    """

    def __init__(self,
                 tasks: Dict[str, Dict],
                 regression_weight: float = 1.0,
                 use_huber: bool = False):
        """
        Args:
            tasks: 任务配置
            regression_weight: 回归损失权重
            use_huber: 是否使用Huber损失替代MSE
        """
        super().__init__()

        self.tasks = tasks
        self.regression_weight = regression_weight

        if use_huber:
            self.regression_loss_fn = nn.HuberLoss()
        else:
            self.regression_loss_fn = nn.MSELoss()


    def forward(self,
                outputs: Dict[str, Dict[str, torch.Tensor]],
                targets: Dict[str, torch.Tensor],
                masks: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        计算多任务损失（支持缺失值）

        Args:
            outputs: 模型输出 {task_name: {output_type: tensor}}
            targets: 目标值 {task_name_value: tensor}
            masks: 任务可用性掩码 {task_name_mask: bool_tensor}

        Returns:
            dict: 包含总损失和各任务损失
        """
        losses = {}
        total_loss = None

        for task_name, task_config in self.tasks.items():
            task_type = task_config.get('task_type', 'regression')
            task_outputs = outputs.get(task_name, {})
            task_weight = task_config.get('loss_weight', 1.0)

            # 回归损失（带掩码）
            if 'regression' in task_type and 'regression' in task_outputs:
                target_key = f'{task_name}_value'
                mask_key = f'{task_name}_mask'

                if target_key in targets:
                    pred = task_outputs['regression']
                    target = targets[target_key]

                    # 如果提供了掩码，只计算有效样本的损失
                    if masks is not None and mask_key in masks:
                        mask = masks[mask_key]
                        if mask.any():  # 至少有一个有效样本
                            masked_pred = pred[mask]
                            masked_target = target[mask]
                            reg_loss = self.regression_loss_fn(masked_pred, masked_target)
                            losses[f'{task_name}_regression'] = reg_loss
                            weighted = reg_loss * self.regression_weight * task_weight
                            total_loss = weighted if total_loss is None else total_loss + weighted
                    else:
                        # 向后兼容：没有掩码时使用全部样本
                        reg_loss = self.regression_loss_fn(pred, target)
                        losses[f'{task_name}_regression'] = reg_loss
                        weighted = reg_loss * self.regression_weight * task_weight
                        total_loss = weighted if total_loss is None else total_loss + weighted

        if total_loss is None:
            any_out = None
            for task_outputs in outputs.values():
                for value in task_outputs.values():
                    any_out = value
                    break
                if any_out is not None:
                    break
            total_loss = any_out.sum() * 0.0 if any_out is not None else torch.tensor(0.0)

        losses['total'] = total_loss

        return losses


def create_model(config: Dict[str, Any], tasks: Dict[str, Dict]) -> MoleculePropertyPredictor:
    """
    工厂函数：创建模型

    Args:
        config: 完整配置字典
        tasks: 任务配置

    Returns:
        model: MoleculePropertyPredictor实例
    """
    model_config = config['model']
    model = MoleculePropertyPredictor(model_config, tasks)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Tasks: {list(tasks.keys())}")

    return model


if __name__ == "__main__":
    # 测试模型
    import sys
    sys.path.append('..')
    from utils import load_config

    config = load_config("../config.yaml")

    tasks = {
        'sweetness': {
            'task_type': 'regression+classification',
            'loss_weight': 1.0,
        }
    }

    model = create_model(config, tasks)

    # 创建假数据测试
    from torch_geometric.data import Data, Batch

    batch_size = 4
    num_nodes = 10
    num_edges = 20

    # 创建图数据
    graphs = []
    for _ in range(batch_size):
        x = torch.randn(num_nodes, 78)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, 12)
        graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))

    graph_batch = Batch.from_data_list(graphs)

    # 序列数据
    input_ids = torch.randint(0, 100, (batch_size, 128))
    attention_mask = torch.ones(batch_size, 128)

    # 前向传播
    outputs = model(graph_batch, input_ids, attention_mask)

    print("\nOutputs:")
    for task_name, task_output in outputs.items():
        print(f"  {task_name}:")
        for output_type, tensor in task_output.items():
            print(f"    {output_type}: {tensor.shape}")
