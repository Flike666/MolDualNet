"""
Transformer编码器模块
Transformer Encoder Module for SMILES sequences
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    正弦位置编码

    为序列中的每个位置添加位置信息
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]

        Returns:
            x + positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    可学习的位置编码
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.position_embedding(positions)
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """
    单层Transformer编码器
    """

    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            key_padding_mask: [batch_size, seq_len], True表示padding位置

        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # Self-attention with residual
        attn_output, _ = self.self_attn(
            x, x, x,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)

        return x


class TransformerEncoder(nn.Module):
    """
    SMILES序列的Transformer编码器

    将SMILES token序列编码为固定维度的表示
    """

    def __init__(self,
                 vocab_size: int = 100,
                 max_seq_len: int = 256,
                 embed_dim: int = 128,
                 num_heads: int = 4,
                 num_layers: int = 4,
                 dropout: float = 0.1,
                 output_dim: int = 128,
                 use_learnable_pe: bool = True):
        """
        Args:
            vocab_size: 词表大小
            max_seq_len: 最大序列长度
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            num_layers: Transformer层数
            dropout: dropout比率
            output_dim: 输出维度
            use_learnable_pe: 是否使用可学习的位置编码
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Token嵌入
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # 位置编码
        if use_learnable_pe:
            self.positional_encoding = LearnablePositionalEncoding(
                embed_dim, max_seq_len, dropout
            )
        else:
            self.positional_encoding = PositionalEncoding(
                embed_dim, max_seq_len, dropout
            )

        # Transformer层
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=embed_dim,
                num_heads=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embed_dim)

        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, output_dim),
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播

        Args:
            input_ids: token索引 [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len], 1表示有效位置

        Returns:
            output: 序列表示 [batch_size, output_dim]
        """
        # Token嵌入
        x = self.token_embedding(input_ids)  # [batch, seq_len, embed_dim]

        # 位置编码
        x = self.positional_encoding(x)

        # 创建key_padding_mask (True表示padding位置，需要被忽略)
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None

        # Transformer编码
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, key_padding_mask=key_padding_mask)

        x = self.final_norm(x)

        # 使用[CLS] token (第一个位置) 的表示作为序列表示
        cls_output = x[:, 0, :]  # [batch, embed_dim]

        # 输出投影
        output = self.output_projection(cls_output)

        return output

    def forward_with_attention(self,
                               input_ids: torch.Tensor,
                               attention_mask: Optional[torch.Tensor] = None):
        """
        前向传播，同时返回注意力权重（用于可解释性分析）
        """
        x = self.token_embedding(input_ids)
        x = self.positional_encoding(x)

        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None

        attention_weights = []

        for encoder_layer in self.encoder_layers:
            # 获取注意力权重
            attn_output, attn_weight = encoder_layer.self_attn(
                x, x, x,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=True
            )
            attention_weights.append(attn_weight)

            # 继续前向传播
            x = encoder_layer.norm1(x + encoder_layer.dropout(attn_output))
            ff_output = encoder_layer.feed_forward(x)
            x = encoder_layer.norm2(x + ff_output)

        x = self.final_norm(x)
        cls_output = x[:, 0, :]
        output = self.output_projection(cls_output)

        return output, attention_weights


class SmilesTransformerWithPooling(nn.Module):
    """
    带有多种池化策略的SMILES Transformer

    除了CLS token，还可以使用mean/max pooling
    """

    def __init__(self,
                 vocab_size: int = 100,
                 max_seq_len: int = 256,
                 embed_dim: int = 128,
                 num_heads: int = 4,
                 num_layers: int = 4,
                 dropout: float = 0.1,
                 output_dim: int = 128,
                 pooling: str = 'cls'):
        """
        Args:
            pooling: 池化策略 ('cls', 'mean', 'max', 'cls_mean')
        """
        super().__init__()

        self.pooling = pooling
        self.embed_dim = embed_dim

        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.positional_encoding = LearnablePositionalEncoding(embed_dim, max_seq_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.final_norm = nn.LayerNorm(embed_dim)

        # 根据池化策略调整输出投影的输入维度
        if pooling == 'cls_mean':
            proj_input_dim = embed_dim * 2
        else:
            proj_input_dim = embed_dim

        self.output_projection = nn.Sequential(
            nn.Linear(proj_input_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, output_dim),
        )

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        x = self.token_embedding(input_ids)
        x = self.positional_encoding(x)

        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.final_norm(x)

        # 池化
        if self.pooling == 'cls':
            pooled = x[:, 0, :]
        elif self.pooling == 'mean':
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:
                pooled = x.mean(dim=1)
        elif self.pooling == 'max':
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                x = x.masked_fill(mask == 0, -1e9)
            pooled = x.max(dim=1)[0]
        elif self.pooling == 'cls_mean':
            cls_output = x[:, 0, :]
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                mean_output = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:
                mean_output = x.mean(dim=1)
            pooled = torch.cat([cls_output, mean_output], dim=-1)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

        output = self.output_projection(pooled)

        return output


if __name__ == "__main__":
    # 测试Transformer编码器
    batch_size = 4
    seq_len = 128
    vocab_size = 100

    model = TransformerEncoder(
        vocab_size=vocab_size,
        max_seq_len=256,
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        output_dim=128,
    )

    # 创建测试数据
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[:, -20:] = 0  # 最后20个位置是padding

    # 前向传播
    output = model(input_ids, attention_mask)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
