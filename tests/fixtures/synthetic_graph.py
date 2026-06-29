"""Small tensor graph fixture factory."""

from __future__ import annotations

import torch


def synthetic_graph() -> dict[str, torch.Tensor]:
    return {
        "node_features": torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10,
        "symbol_idx": torch.arange(3, dtype=torch.long),
        "adjacency_matrix": torch.ones(3, 3, dtype=torch.float32),
    }
