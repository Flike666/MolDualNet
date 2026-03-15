"""
交叉注意力融合模块
Cross-Attention Fusion Module for GNN and Transformer representations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CrossAttention(nn.Module):
    """
    交叉注意力层

    允许一个模态的表示查询另一个模态的信息
    """

    def __init__(self,
                 query_dim: int,
                 key_dim: int,
                 hidden_dim: int,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        """
        Args:
            query_dim: Query向量维度
            key_dim: Key/Value向量维度
            hidden_dim: 输出隐藏维度
            num_heads: 注意力头数
            dropout: dropout比率
        """
        super().__init__()

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Query, Key, Value投影
        self.q_proj = nn.Linear(query_dim, hidden_dim)
        self.k_proj = nn.Linear(key_dim, hidden_dim)
        self.v_proj = nn.Linear(key_dim, hidden_dim)

        # 输出投影
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: [batch, query_len, query_dim]
            key: [batch, key_len, key_dim]
            value: [batch, key_len, key_dim]
            key_padding_mask: [batch, key_len], True表示需要mask的位置

        Returns:
            output: [batch, query_len, hidden_dim]
        """
        batch_size, query_len, _ = query.shape
        key_len = key.shape[1]

        # 投影
        Q = self.q_proj(query)  # [batch, query_len, hidden_dim]
        K = self.k_proj(key)    # [batch, key_len, hidden_dim]
        V = self.v_proj(value)  # [batch, key_len, hidden_dim]

        # 重塑为多头
        Q = Q.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        # 现在形状: [batch, num_heads, seq_len, head_dim]

        # 注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # [batch, num_heads, query_len, key_len]

        # 应用mask (使用大负数而非-inf，更稳定)
        if key_padding_mask is not None:
            # key_padding_mask: [batch, key_len] -> [batch, 1, 1, key_len]
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                -1e4  # 使用-1e4而非-inf，避免NaN
            )

        # Clamp防止数值溢出
        attn_scores = torch.clamp(attn_scores, min=-1e4, max=1e4)

        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权求和
        output = torch.matmul(attn_weights, V)
        # [batch, num_heads, query_len, head_dim]

        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, query_len, self.hidden_dim)

        # 输出投影
        output = self.out_proj(output)

        return output


class CrossAttentionFusionLayer(nn.Module):
    """
    双向交叉注意力融合层

    GNN表示和Transformer表示相互关注
    """

    def __init__(self,
                 gnn_dim: int,
                 trans_dim: int,
                 hidden_dim: int,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        # 输入投影（如果维度不匹配）
        self.gnn_input_proj = nn.Linear(gnn_dim, hidden_dim) if gnn_dim != hidden_dim else nn.Identity()
        self.trans_input_proj = nn.Linear(trans_dim, hidden_dim) if trans_dim != hidden_dim else nn.Identity()

        # Pre-LN for better stability
        self.gnn_pre_norm = nn.LayerNorm(hidden_dim)
        self.trans_pre_norm = nn.LayerNorm(hidden_dim)

        # GNN -> Transformer的交叉注意力 (GNN查询Transformer)
        # 注意：输入已经通过input_proj投影到hidden_dim
        self.gnn_to_trans = CrossAttention(
            query_dim=hidden_dim,
            key_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # Transformer -> GNN的交叉注意力 (Transformer查询GNN)
        self.trans_to_gnn = CrossAttention(
            query_dim=hidden_dim,
            key_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # Post-attention LayerNorm (保留用于FFN)
        self.gnn_norm = nn.LayerNorm(hidden_dim)
        self.trans_norm = nn.LayerNorm(hidden_dim)

        # Feed Forward Networks
        self.gnn_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

        self.trans_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

        self.gnn_ffn_norm = nn.LayerNorm(hidden_dim)
        self.trans_ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(self,
                gnn_features: torch.Tensor,
                trans_features: torch.Tensor,
                gnn_mask: Optional[torch.Tensor] = None,
                trans_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            gnn_features: GNN节点特征 [batch, num_nodes, gnn_dim]
            trans_features: Transformer序列特征 [batch, seq_len, trans_dim]
            gnn_mask: GNN节点mask [batch, num_nodes]
            trans_mask: Transformer padding mask [batch, seq_len]

        Returns:
            gnn_output: 融合后的GNN特征 [batch, num_nodes, hidden_dim]
            trans_output: 融合后的Transformer特征 [batch, seq_len, hidden_dim]
        """
        # 输入投影
        gnn_h = self.gnn_input_proj(gnn_features)
        trans_h = self.trans_input_proj(trans_features)

        # 交叉注意力 (Pre-LN + Residual Scaling)
        # GNN查询Transformer (Pre-LN for stability)
        gnn_cross = self.gnn_to_trans(
            self.gnn_pre_norm(gnn_h),
            self.trans_pre_norm(trans_h),
            self.trans_pre_norm(trans_h),
            trans_mask
        )
        gnn_h = gnn_h + 0.1 * gnn_cross  # 缩放残差连接，增强稳定性

        # Transformer查询GNN (Pre-LN for stability)
        trans_cross = self.trans_to_gnn(
            self.trans_norm(trans_h),
            self.gnn_norm(gnn_h),
            self.gnn_norm(gnn_h),
            gnn_mask
        )
        trans_h = trans_h + 0.1 * trans_cross  # 缩放残差连接

        # FFN with gradient scaling
        gnn_h = gnn_h + 0.1 * self.gnn_ffn(self.gnn_ffn_norm(gnn_h))
        trans_h = trans_h + 0.1 * self.trans_ffn(self.trans_ffn_norm(trans_h))

        return gnn_h, trans_h


class CrossAttentionFusion(nn.Module):
    """
    多层交叉注意力融合模块

    堆叠多层交叉注意力，实现深层交互
    """

    def __init__(self,
                 gnn_dim: int,
                 trans_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 output_dim: int = 256):
        """
        Args:
            gnn_dim: GNN节点特征维度
            trans_dim: Transformer序列特征维度
            hidden_dim: 隐藏层维度
            num_layers: 交叉注意力层数
            num_heads: 注意力头数
            dropout: dropout比率
            output_dim: 输出维度
        """
        super().__init__()

        self.num_layers = num_layers

        # 交叉注意力层
        self.fusion_layers = nn.ModuleList([
            CrossAttentionFusionLayer(
                gnn_dim=gnn_dim if i == 0 else hidden_dim,
                trans_dim=trans_dim if i == 0 else hidden_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for i in range(num_layers)
        ])

        # 全局池化后的融合
        self.gnn_pool_proj = nn.Linear(hidden_dim, hidden_dim)
        self.trans_pool_proj = nn.Linear(hidden_dim, hidden_dim)

        # 门控融合
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        # 最终输出
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self,
                gnn_features: torch.Tensor,
                trans_features: torch.Tensor,
                gnn_batch: Optional[torch.Tensor] = None,
                trans_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            gnn_features: GNN节点特征 [total_nodes, gnn_dim]
            trans_features: Transformer序列特征 [batch, seq_len, trans_dim]
            gnn_batch: 节点到图的映射 [total_nodes]
            trans_mask: Transformer注意力mask [batch, seq_len], 1=有效, 0=padding

        Returns:
            fused: 融合后的表示 [batch, output_dim]
        """
        batch_size = trans_features.shape[0]
        device = trans_features.device

        # 将GNN节点特征按batch分组并padding到相同长度
        if gnn_batch is not None:
            gnn_padded, gnn_mask = self._pad_gnn_features(gnn_features, gnn_batch, batch_size)
        else:
            gnn_padded = gnn_features.unsqueeze(0)
            gnn_mask = None

        # 转换mask格式 (True = 需要mask的位置)
        if trans_mask is not None:
            trans_padding_mask = (trans_mask == 0)
        else:
            trans_padding_mask = None

        # 多层交叉注意力
        gnn_h = gnn_padded
        trans_h = trans_features

        for fusion_layer in self.fusion_layers:
            gnn_h, trans_h = fusion_layer(
                gnn_h, trans_h,
                gnn_mask=gnn_mask,
                trans_mask=trans_padding_mask
            )

        # 全局池化 (增强数值稳定性)
        # GNN: masked mean pooling
        if gnn_mask is not None:
            gnn_valid_mask = (~gnn_mask).float().unsqueeze(-1)  # [batch, nodes, 1]
            gnn_pooled = (gnn_h * gnn_valid_mask).sum(dim=1) / gnn_valid_mask.sum(dim=1).clamp(min=1e-6)  # 增大epsilon
        else:
            gnn_pooled = gnn_h.mean(dim=1)

        # Transformer: masked mean pooling (或使用CLS)
        if trans_mask is not None:
            trans_valid_mask = trans_mask.float().unsqueeze(-1)  # [batch, seq_len, 1]
            trans_pooled = (trans_h * trans_valid_mask).sum(dim=1) / trans_valid_mask.sum(dim=1).clamp(min=1e-6)  # 增大epsilon
        else:
            trans_pooled = trans_h.mean(dim=1)

        # 投影
        gnn_pooled = self.gnn_pool_proj(gnn_pooled)
        trans_pooled = self.trans_pool_proj(trans_pooled)

        # 门控融合
        combined = torch.cat([gnn_pooled, trans_pooled], dim=-1)
        gate = self.gate(combined)

        fused = gate * gnn_pooled + (1 - gate) * trans_pooled

        # 拼接并输出
        output = self.output_layer(torch.cat([fused, gnn_pooled + trans_pooled], dim=-1))

        return output

    def _pad_gnn_features(self,
                          node_features: torch.Tensor,
                          batch: torch.Tensor,
                          batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将不同大小的图padding到相同大小

        Args:
            node_features: [total_nodes, dim]
            batch: [total_nodes] 节点到图的映射
            batch_size: batch大小

        Returns:
            padded: [batch_size, max_nodes, dim]
            mask: [batch_size, max_nodes] True表示padding位置
        """
        device = node_features.device
        dim = node_features.shape[-1]

        # 计算每个图的节点数
        num_nodes_per_graph = torch.bincount(batch, minlength=batch_size)
        max_nodes = num_nodes_per_graph.max().item()

        # 初始化
        padded = torch.zeros(batch_size, max_nodes, dim, device=device)
        mask = torch.ones(batch_size, max_nodes, dtype=torch.bool, device=device)

        # 填充
        current_idx = 0
        for i in range(batch_size):
            n = num_nodes_per_graph[i].item()
            padded[i, :n] = node_features[current_idx:current_idx + n]
            mask[i, :n] = False
            current_idx += n

        return padded, mask


if __name__ == "__main__":
    # 测试交叉注意力融合
    batch_size = 4
    num_nodes = 15
    seq_len = 32
    gnn_dim = 128
    trans_dim = 128
    hidden_dim = 256

    # 创建测试数据
    gnn_features = torch.randn(batch_size * num_nodes, gnn_dim)
    trans_features = torch.randn(batch_size, seq_len, trans_dim)

    # batch索引
    gnn_batch = torch.repeat_interleave(torch.arange(batch_size), num_nodes)
    trans_mask = torch.ones(batch_size, seq_len)
    trans_mask[:, -5:] = 0  # 最后5个是padding

    # 创建模型
    fusion = CrossAttentionFusion(
        gnn_dim=gnn_dim,
        trans_dim=trans_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        num_heads=4,
        output_dim=256
    )

    # 前向传播
    output = fusion(gnn_features, trans_features, gnn_batch, trans_mask)
    print(f"GNN features: {gnn_features.shape}")
    print(f"Transformer features: {trans_features.shape}")
    print(f"Output: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in fusion.parameters()):,}")
