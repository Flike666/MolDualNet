"""
图神经网络编码器模块
Graph Neural Network Encoder Module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GINConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Batch
from typing import Optional


class GNNEncoder(nn.Module):
    """
    图神经网络编码器

    使用多层GATv2Conv (Graph Attention Network v2)进行消息传递，
    支持边特征，最后使用全局池化得到图级表示。
    """

    def __init__(self,
                 node_input_dim: int = 82,
                 edge_input_dim: int = 20,
                 hidden_dim: int = 128,
                 output_dim: int = 128,
                 num_layers: int = 4,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 residual: bool = True):
        """
        Args:
            node_input_dim: 节点特征输入维度 (默认82)
            edge_input_dim: 边特征输入维度 (默认20 = 12维2D特征 + 8维3D特征)
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            num_layers: GNN层数
            num_heads: 注意力头数
            dropout: dropout比率
            residual: 是否使用残差连接
        """
        super().__init__()

        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual

        # 节点特征嵌入层
        self.node_embedding = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 边特征嵌入层
        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # GATv2卷积层
        self.gnn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for i in range(num_layers):
            # GATv2Conv支持边特征
            self.gnn_layers.append(
                GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    edge_dim=hidden_dim,
                    concat=True,
                    add_self_loops=True,
                )
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # 输出投影层
        # 全局池化使用mean + max，所以输入是2*hidden_dim
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 节点特征 [num_nodes, node_input_dim]
            edge_index: 边索引 [2, num_edges]
            edge_attr: 边特征 [num_edges, edge_input_dim]
            batch: 批次索引 [num_nodes]

        Returns:
            graph_embedding: 图级表示 [batch_size, output_dim]
        """
        # 节点特征嵌入
        h = self.node_embedding(x)

        # 边特征嵌入
        if edge_attr is not None and edge_attr.shape[0] > 0:
            edge_emb = self.edge_embedding(edge_attr)
        else:
            edge_emb = None

        # GNN层
        for i, (gnn_layer, layer_norm) in enumerate(zip(self.gnn_layers, self.layer_norms)):
            h_new = gnn_layer(h, edge_index, edge_attr=edge_emb)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)

            # 残差连接
            if self.residual:
                h = layer_norm(h + h_new)
            else:
                h = layer_norm(h_new)

        # 全局池化
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)

        # 使用mean + max池化
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_graph = torch.cat([h_mean, h_max], dim=-1)

        # 输出投影
        output = self.output_projection(h_graph)

        return output

    def forward_with_node_features(self,
                                   x: torch.Tensor,
                                   edge_index: torch.Tensor,
                                   edge_attr: Optional[torch.Tensor] = None,
                                   batch: Optional[torch.Tensor] = None):
        """
        前向传播，同时返回节点特征

        Returns:
            graph_embedding: 图级表示
            node_features: 节点级特征
        """
        # 节点特征嵌入
        h = self.node_embedding(x)

        # 边特征嵌入
        if edge_attr is not None and edge_attr.shape[0] > 0:
            edge_emb = self.edge_embedding(edge_attr)
        else:
            edge_emb = None

        # GNN层
        for i, (gnn_layer, layer_norm) in enumerate(zip(self.gnn_layers, self.layer_norms)):
            h_new = gnn_layer(h, edge_index, edge_attr=edge_emb)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)

            if self.residual:
                h = layer_norm(h + h_new)
            else:
                h = layer_norm(h_new)

        # 全局池化
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)

        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_graph = torch.cat([h_mean, h_max], dim=-1)

        output = self.output_projection(h_graph)

        return output, h


class GINEncoder(nn.Module):
    """
    GIN (Graph Isomorphism Network) 编码器

    作为GATv2的替代方案，更简单但通常也很有效
    注意：GIN不使用边特征
    """

    def __init__(self,
                 node_input_dim: int = 82,
                 hidden_dim: int = 128,
                 output_dim: int = 128,
                 num_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        self.node_embedding = nn.Linear(node_input_dim, hidden_dim)

        self.gnn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.gnn_layers.append(GINConv(mlp, train_eps=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = dropout

        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:

        h = self.node_embedding(x)

        for gnn_layer, batch_norm in zip(self.gnn_layers, self.batch_norms):
            h = gnn_layer(h, edge_index)
            h = batch_norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)

        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_graph = torch.cat([h_mean, h_max], dim=-1)

        output = self.output_projection(h_graph)

        return output


if __name__ == "__main__":
    # 测试GNN编码器
    from features import AtomFeaturizer, BondFeaturizer, Geometry3DFeaturizer, mol_to_graph_data, get_feature_dims

    # 创建测试数据
    test_smiles = "c1ccccc1"  # 苯
    atom_feat = AtomFeaturizer()
    bond_feat = BondFeaturizer()
    geom_feat = Geometry3DFeaturizer()

    # 测试使用3D特征
    result = mol_to_graph_data(
        test_smiles, atom_feat, bond_feat, geom_feat, use_3d_features=True
    )
    node_features, edge_index, edge_features, positions, has_3d = result

    print(f"Node features shape: {node_features.shape}")
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Edge features shape: {edge_features.shape} (12 + 8 = 20 with 3D)")
    print(f"Has 3D coords: {has_3d}")
    if positions is not None:
        print(f"3D positions shape: {positions.shape}")

    # 获取特征维度
    node_dim, edge_dim = get_feature_dims(use_3d_features=True)
    print(f"\nFeature dimensions: node={node_dim}, edge={edge_dim}")

    # 创建模型
    model = GNNEncoder(
        node_input_dim=node_dim,
        edge_input_dim=edge_dim,
        hidden_dim=128,
        output_dim=128,
        num_layers=4,
    )

    # 转换为tensor
    x = torch.tensor(node_features, dtype=torch.float)
    edge_idx = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_features, dtype=torch.float)

    # 前向传播
    output = model(x, edge_idx, edge_attr)
    print(f"\nOutput shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
