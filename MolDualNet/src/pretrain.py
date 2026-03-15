"""
预训练模块
Pretraining Module for Molecular Property Prediction

包含以下预训练任务：
1. Masked Language Model (MLM) - 遮蔽SMILES token预测
2. Masked Atom Prediction (MAP) - 遮蔽原子特征预测
3. Contrastive Learning - GNN和Transformer表示对比学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import random


class MaskedLanguageModel(nn.Module):
    """
    Masked Language Model (MLM) 预训练头

    随机遮蔽SMILES序列中的token，预测被遮蔽的token
    """

    def __init__(self, embed_dim: int, vocab_size: int, hidden_dim: int = 256):
        super().__init__()

        self.mlm_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, sequence_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sequence_features: Transformer序列特征 [batch, seq_len, embed_dim]

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        return self.mlm_head(sequence_features)


class MaskedAtomPrediction(nn.Module):
    """
    Masked Atom Prediction (MAP) 预训练头

    随机遮蔽分子图中的原子特征，预测被遮蔽的原子类型
    """

    def __init__(self, hidden_dim: int, num_atom_types: int = 100):
        super().__init__()

        # 预测原子类型
        self.atom_type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_atom_types),
        )

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: GNN节点特征 [num_nodes, hidden_dim]

        Returns:
            atom_logits: [num_nodes, num_atom_types]
        """
        return self.atom_type_head(node_features)


class ContrastiveLoss(nn.Module):
    """
    对比学习损失 (InfoNCE)

    最大化同一分子的GNN和Transformer表示的相似度，
    最小化不同分子表示的相似度
    """

    def __init__(self, temperature: float = 0.07, hidden_dim: int = 128):
        super().__init__()

        self.temperature = temperature

        # 投影头
        self.gnn_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.trans_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self,
                gnn_embeddings: torch.Tensor,
                trans_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gnn_embeddings: GNN图级表示 [batch, hidden_dim]
            trans_embeddings: Transformer序列表示 [batch, hidden_dim]

        Returns:
            loss: 对比学习损失
        """
        batch_size = gnn_embeddings.shape[0]

        # 投影
        gnn_proj = F.normalize(self.gnn_projector(gnn_embeddings), dim=-1)
        trans_proj = F.normalize(self.trans_projector(trans_embeddings), dim=-1)

        # 计算相似度矩阵
        sim_matrix = torch.matmul(gnn_proj, trans_proj.T) / self.temperature
        # [batch, batch]

        # 对角线是正样本
        labels = torch.arange(batch_size, device=sim_matrix.device)

        # 双向对比损失
        loss_gnn_to_trans = F.cross_entropy(sim_matrix, labels)
        loss_trans_to_gnn = F.cross_entropy(sim_matrix.T, labels)

        return (loss_gnn_to_trans + loss_trans_to_gnn) / 2


class PretrainingModel(nn.Module):
    """
    预训练模型

    结合MLM、MAP和对比学习的预训练框架
    """

    def __init__(self,
                 gnn_encoder: nn.Module,
                 transformer_encoder: nn.Module,
                 gnn_hidden_dim: int = 128,
                 trans_embed_dim: int = 128,
                 vocab_size: int = 100,
                 num_atom_types: int = 100,
                 mlm_weight: float = 1.0,
                 map_weight: float = 1.0,
                 contrastive_weight: float = 0.5,
                 mask_prob: float = 0.15,
                 temperature: float = 0.07):
        """
        Args:
            gnn_encoder: GNN编码器
            transformer_encoder: Transformer编码器
            gnn_hidden_dim: GNN隐藏维度
            trans_embed_dim: Transformer嵌入维度
            vocab_size: SMILES词表大小
            num_atom_types: 原子类型数量
            mlm_weight: MLM损失权重
            map_weight: MAP损失权重
            contrastive_weight: 对比学习损失权重
            mask_prob: 遮蔽概率
            temperature: 对比学习温度参数
        """
        super().__init__()

        self.gnn_encoder = gnn_encoder
        self.transformer_encoder = transformer_encoder
        self.gnn_hidden_dim = gnn_hidden_dim
        self.trans_embed_dim = trans_embed_dim
        self.mask_prob = mask_prob

        self.mlm_weight = mlm_weight
        self.map_weight = map_weight
        self.contrastive_weight = contrastive_weight

        # MLM头
        self.mlm_head = MaskedLanguageModel(
            embed_dim=trans_embed_dim,
            vocab_size=vocab_size,
        )

        # MAP头
        self.map_head = MaskedAtomPrediction(
            hidden_dim=gnn_hidden_dim,
            num_atom_types=num_atom_types,
        )

        # 对比学习
        self.contrastive = ContrastiveLoss(
            temperature=temperature,
            hidden_dim=min(gnn_hidden_dim, trans_embed_dim),
        )

        # GNN图级表示投影
        self.gnn_pool_proj = nn.Linear(gnn_hidden_dim * 2, min(gnn_hidden_dim, trans_embed_dim))

    def mask_tokens(self,
                    input_ids: torch.Tensor,
                    vocab_size: int,
                    mask_token_id: int = 1,
                    special_token_ids: tuple = (0, 2, 3)) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        随机遮蔽SMILES token

        Args:
            input_ids: [batch, seq_len]
            vocab_size: 词表大小
            mask_token_id: [MASK] token的ID
            special_token_ids: 特殊token ID (PAD, CLS, SEP等)

        Returns:
            masked_ids: 遮蔽后的input_ids
            labels: 原始token ID，未遮蔽位置为-100
        """
        labels = input_ids.clone()
        masked_ids = input_ids.clone()

        # 创建遮蔽概率矩阵
        prob_matrix = torch.full(input_ids.shape, self.mask_prob, device=input_ids.device)

        # 特殊token不遮蔽
        for special_id in special_token_ids:
            prob_matrix.masked_fill_(input_ids == special_id, 0.0)

        # 随机选择要遮蔽的位置
        mask_indices = torch.bernoulli(prob_matrix).bool()

        # 未遮蔽位置的label设为-100 (忽略)
        labels[~mask_indices] = -100

        # 80%替换为[MASK], 10%替换为随机token, 10%保持不变
        random_matrix = torch.rand(input_ids.shape, device=input_ids.device)

        # 80% -> [MASK]
        mask_token_indices = mask_indices & (random_matrix < 0.8)
        masked_ids[mask_token_indices] = mask_token_id

        # 10% -> 随机token
        random_token_indices = mask_indices & (random_matrix >= 0.8) & (random_matrix < 0.9)
        random_tokens = torch.randint(
            low=4, high=vocab_size,  # 避免特殊token
            size=input_ids.shape,
            device=input_ids.device
        )
        masked_ids[random_token_indices] = random_tokens[random_token_indices]

        # 10% -> 保持不变 (已经是原始值)

        return masked_ids, labels

    def mask_atoms(self,
                   x: torch.Tensor,
                   batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        随机遮蔽原子特征

        Args:
            x: 原子特征 [num_nodes, node_dim]
            batch: 节点到图的映射 [num_nodes]

        Returns:
            masked_x: 遮蔽后的原子特征
            mask_indices: 被遮蔽的节点索引
            atom_types: 原始原子类型 (用前几个特征位表示)
        """
        num_nodes = x.shape[0]

        # 创建遮蔽mask
        mask_prob = torch.full((num_nodes,), self.mask_prob, device=x.device)
        mask_indices = torch.bernoulli(mask_prob).bool()

        # 提取原子类型 (假设前44位是原子类型的one-hot编码)
        atom_types = x[:, :44].argmax(dim=-1)

        # 遮蔽原子特征 (用0替换)
        masked_x = x.clone()
        masked_x[mask_indices] = 0

        return masked_x, mask_indices, atom_types

    def forward(self,
                graph_batch,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                vocab_size: int = 100) -> Dict[str, torch.Tensor]:
        """
        预训练前向传播

        Args:
            graph_batch: PyG Batch对象
            input_ids: SMILES token索引 [batch, seq_len]
            attention_mask: 注意力掩码 [batch, seq_len]
            vocab_size: 词表大小

        Returns:
            losses: 包含各项损失的字典
        """
        losses = {}
        total_loss = 0.0

        # ============ MLM ============
        if self.mlm_weight > 0:
            # 遮蔽token
            masked_ids, mlm_labels = self.mask_tokens(input_ids, vocab_size)

            # 获取Transformer序列特征
            trans_features = self._get_transformer_features(masked_ids, attention_mask)

            # MLM预测
            mlm_logits = self.mlm_head(trans_features)
            mlm_loss = F.cross_entropy(
                mlm_logits.view(-1, vocab_size),
                mlm_labels.view(-1),
                ignore_index=-100
            )
            losses['mlm_loss'] = mlm_loss
            total_loss += self.mlm_weight * mlm_loss

        # ============ MAP ============
        if self.map_weight > 0:
            # 遮蔽原子
            masked_x, mask_indices, atom_types = self.mask_atoms(
                graph_batch.x, graph_batch.batch
            )

            # 获取GNN节点特征
            _, node_features = self.gnn_encoder.forward_with_node_features(
                x=masked_x,
                edge_index=graph_batch.edge_index,
                edge_attr=graph_batch.edge_attr,
                batch=graph_batch.batch,
            )

            # MAP预测 (只对遮蔽的节点)
            if mask_indices.sum() > 0:
                masked_node_features = node_features[mask_indices]
                masked_atom_types = atom_types[mask_indices]

                map_logits = self.map_head(masked_node_features)
                map_loss = F.cross_entropy(map_logits, masked_atom_types)
            else:
                map_loss = torch.tensor(0.0, device=input_ids.device)

            losses['map_loss'] = map_loss
            total_loss += self.map_weight * map_loss

        # ============ Contrastive Learning ============
        if self.contrastive_weight > 0:
            # 获取GNN图级表示 (使用原始特征)
            gnn_output, node_features_clean = self.gnn_encoder.forward_with_node_features(
                x=graph_batch.x,
                edge_index=graph_batch.edge_index,
                edge_attr=graph_batch.edge_attr,
                batch=graph_batch.batch,
            )

            # 获取Transformer序列表示 (使用原始token)
            trans_output = self.transformer_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # 对比学习损失
            contrastive_loss = self.contrastive(gnn_output, trans_output)
            losses['contrastive_loss'] = contrastive_loss
            total_loss += self.contrastive_weight * contrastive_loss

        losses['total_loss'] = total_loss
        return losses

    def _get_transformer_features(self,
                                   input_ids: torch.Tensor,
                                   attention_mask: torch.Tensor) -> torch.Tensor:
        """获取Transformer序列特征"""
        x = self.transformer_encoder.token_embedding(input_ids)
        x = self.transformer_encoder.positional_encoding(x)

        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None

        for encoder_layer in self.transformer_encoder.encoder_layers:
            x = encoder_layer(x, key_padding_mask=key_padding_mask)

        x = self.transformer_encoder.final_norm(x)
        return x


def create_pretraining_model(config: Dict,
                             gnn_encoder: nn.Module,
                             transformer_encoder: nn.Module,
                             vocab_size: int) -> PretrainingModel:
    """
    创建预训练模型

    Args:
        config: 配置字典
        gnn_encoder: GNN编码器
        transformer_encoder: Transformer编码器
        vocab_size: 词表大小

    Returns:
        model: PretrainingModel实例
    """
    pretrain_config = config.get('pretrain', {})

    model = PretrainingModel(
        gnn_encoder=gnn_encoder,
        transformer_encoder=transformer_encoder,
        gnn_hidden_dim=config['model']['gnn'].get('hidden_dim', 128),
        trans_embed_dim=config['model']['transformer'].get('embed_dim', 128),
        vocab_size=vocab_size,
        num_atom_types=44,  # 常见原子类型数量
        mlm_weight=pretrain_config.get('mlm_weight', 1.0),
        map_weight=pretrain_config.get('map_weight', 1.0),
        contrastive_weight=pretrain_config.get('contrastive_weight', 0.5),
        mask_prob=pretrain_config.get('mask_prob', 0.15),
        temperature=pretrain_config.get('temperature', 0.07),
    )

    return model


if __name__ == "__main__":
    # 测试预训练模块
    import sys
    sys.path.append('..')

    from gnn import GNNEncoder
    from transformer import TransformerEncoder
    from torch_geometric.data import Data, Batch

    # 创建编码器
    gnn_encoder = GNNEncoder(
        node_input_dim=82,
        edge_input_dim=12,
        hidden_dim=128,
        output_dim=128,
    )

    transformer_encoder = TransformerEncoder(
        vocab_size=100,
        max_seq_len=256,
        embed_dim=128,
        output_dim=128,
    )

    # 创建预训练模型
    pretrain_model = PretrainingModel(
        gnn_encoder=gnn_encoder,
        transformer_encoder=transformer_encoder,
        gnn_hidden_dim=128,
        trans_embed_dim=128,
        vocab_size=100,
    )

    # 创建测试数据
    batch_size = 4
    num_nodes = 10
    num_edges = 20
    seq_len = 64

    graphs = []
    for _ in range(batch_size):
        x = torch.randn(num_nodes, 82)
        # 将前44位设为one-hot编码
        x[:, :44] = F.one_hot(torch.randint(0, 44, (num_nodes,)), 44).float()
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, 12)
        graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))

    graph_batch = Batch.from_data_list(graphs)

    input_ids = torch.randint(4, 100, (batch_size, seq_len))
    input_ids[:, 0] = 2  # CLS token
    attention_mask = torch.ones(batch_size, seq_len)

    # 前向传播
    losses = pretrain_model(graph_batch, input_ids, attention_mask, vocab_size=100)

    print("Pretraining Losses:")
    for name, value in losses.items():
        print(f"  {name}: {value.item():.4f}")

    print(f"\nModel parameters: {sum(p.numel() for p in pretrain_model.parameters()):,}")
