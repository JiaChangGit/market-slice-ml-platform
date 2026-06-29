"""Tensor-forward GNN with pure PyTorch attention fallback."""

from __future__ import annotations

import torch
from torch import Tensor, nn

try:
    from torch_geometric.nn import GATConv as _GATConv

    PYG_GAT_CONV = _GATConv
    USE_PYG: bool = True
except ImportError:
    PYG_GAT_CONV = None
    USE_PYG = False


class UniversalTemporalGNNModel(nn.Module):
    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 64,
        heads: int = 4,
        n_symbols: int = 60,
        embed_dim: int = 16,
    ) -> None:
        super().__init__()
        self.asset_embedding = nn.Embedding(n_symbols, embed_dim)
        self.input_projection = nn.Linear(node_feature_dim + embed_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.direction_head = nn.Linear(hidden_dim, 3)
        self.return_head = nn.Linear(hidden_dim, 1)
        self.vol_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())

    def forward_tensor(
        self, node_features: Tensor, symbol_idx: Tensor, adjacency_matrix: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        embedding = self.asset_embedding(symbol_idx)
        projected = self.input_projection(torch.cat([node_features, embedding], dim=-1))
        sequence = projected.unsqueeze(0)
        connected = adjacency_matrix > 0
        attended, _ = self.attn(sequence, sequence, sequence, attn_mask=~connected)
        hidden = self.layer_norm(attended.squeeze(0) + projected)
        return (
            self.direction_head(hidden),
            self.return_head(hidden).squeeze(-1),
            self.vol_head(hidden).squeeze(-1),
        )

    def forward(
        self, node_features: Tensor, symbol_idx: Tensor, adjacency_matrix: Tensor
    ) -> dict[str, Tensor]:
        direction, returns, volatility = self.forward_tensor(
            node_features, symbol_idx, adjacency_matrix
        )
        return {"direction": direction, "return": returns, "volatility": volatility}
