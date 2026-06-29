"""Universal all-asset LSTM with three prediction heads."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class UniversalLSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        n_layers: int = 2,
        dropout: float = 0.2,
        n_symbols: int = 60,
        embed_dim: int = 16,
    ) -> None:
        super().__init__()
        self.asset_embedding = nn.Embedding(n_symbols, embed_dim)
        self.lstm = nn.LSTM(
            input_size=input_size + embed_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.direction_head = nn.Linear(hidden_size, 3)
        self.return_head = nn.Linear(hidden_size, 1)
        self.vol_head = nn.Sequential(nn.Linear(hidden_size, 1), nn.Softplus())

    def forward(self, x: Tensor, symbol_idx: Tensor) -> dict[str, Tensor]:
        embedding = self.asset_embedding(symbol_idx)
        expanded = embedding.unsqueeze(1).expand(-1, x.size(1), -1)
        output, _ = self.lstm(torch.cat([x, expanded], dim=-1))
        hidden = self.layer_norm(output[:, -1, :])
        return {
            "direction": self.direction_head(hidden),
            "return": self.return_head(hidden).squeeze(-1),
            "volatility": self.vol_head(hidden).squeeze(-1),
        }
