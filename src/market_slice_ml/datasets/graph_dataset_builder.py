"""Synchronized graph snapshots using tensor-only fallback records."""

from __future__ import annotations

from datetime import datetime

import polars as pl
import torch
from torch import Tensor

GraphSnapshot = dict[str, Tensor]
DIRECTION_INDEX = {"bearish": 0, "neutral": 1, "bullish": 2}


def build_graph_dataset(
    frames: dict[str, pl.DataFrame],
    feature_columns: list[str],
    adjacency_matrix: Tensor,
    horizon: str | None = None,
) -> list[GraphSnapshot]:
    symbols = sorted(frames)
    common: set[datetime] | None = None
    for frame in frames.values():
        timestamps = set(frame.get_column("timestamp_utc").to_list())
        common = timestamps if common is None else common & timestamps
    snapshots: list[GraphSnapshot] = []
    for timestamp in sorted(common or set()):
        rows: list[list[float]] = []
        directions: list[int] = []
        returns: list[float] = []
        volatilities: list[float] = []
        target_mask: list[bool] = []
        for symbol in symbols:
            selected = frames[symbol].filter(pl.col("timestamp_utc") == timestamp)
            rows.append([float(selected.item(0, column) or 0.0) for column in feature_columns])
            if horizon and f"direction_{horizon}" in selected.columns:
                direction = selected.item(0, f"direction_{horizon}")
                forward_return = selected.item(0, f"forward_return_{horizon}")
                volatility = selected.item(0, f"forward_volatility_{horizon}")
                valid = (
                    direction is not None and forward_return is not None and volatility is not None
                )
                target_mask.append(valid)
                directions.append(DIRECTION_INDEX.get(str(direction), 1))
                returns.append(float(forward_return or 0.0))
                volatilities.append(float(volatility or 0.0))
            else:
                target_mask.append(False)
                directions.append(1)
                returns.append(0.0)
                volatilities.append(0.0)
        snapshot = {
            "node_features": torch.tensor(rows, dtype=torch.float32),
            "symbol_idx": torch.arange(len(symbols), dtype=torch.long),
            "adjacency_matrix": adjacency_matrix.clone().float(),
            "target_mask": torch.tensor(target_mask, dtype=torch.bool),
            "direction": torch.tensor(directions, dtype=torch.long),
            "forward_return": torch.tensor(returns, dtype=torch.float32),
            "forward_volatility": torch.tensor(volatilities, dtype=torch.float32),
        }
        if horizon is None or snapshot["target_mask"].any():
            snapshots.append(snapshot)
    return snapshots
