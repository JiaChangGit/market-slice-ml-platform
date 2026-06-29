"""Sliding, per-symbol sequence tensor construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
import torch
from torch import Tensor

DIRECTION_INDEX = {"bearish": 0, "neutral": 1, "bullish": 2}


@dataclass(frozen=True)
class SequenceTensors:
    features: Tensor
    symbol_idx: Tensor
    direction: Tensor
    forward_return: Tensor
    forward_volatility: Tensor


def build_sequence_dataset(
    frames: dict[str, pl.DataFrame],
    feature_columns: list[str],
    horizon: str = "h1",
    sequence_length: int = 64,
) -> SequenceTensors:
    sequences: list[np.ndarray] = []
    symbols: list[int] = []
    directions: list[int] = []
    returns: list[float] = []
    volatilities: list[float] = []
    for symbol_index, symbol in enumerate(sorted(frames)):
        frame = frames[symbol].sort("timestamp_utc")
        values = frame.select(feature_columns).fill_null(0.0).to_numpy().astype(np.float32)
        labels = frame.select(
            f"direction_{horizon}",
            f"forward_return_{horizon}",
            f"forward_volatility_{horizon}",
        ).to_dicts()
        for end in range(sequence_length - 1, frame.height):
            label = labels[end]
            if any(value is None for value in label.values()):
                continue
            sequences.append(values[end - sequence_length + 1 : end + 1])
            symbols.append(symbol_index)
            directions.append(DIRECTION_INDEX[str(label[f"direction_{horizon}"])])
            returns.append(float(label[f"forward_return_{horizon}"]))
            volatilities.append(float(label[f"forward_volatility_{horizon}"]))
    shape = (0, sequence_length, len(feature_columns))
    data = np.stack(sequences) if sequences else np.empty(shape, dtype=np.float32)
    return SequenceTensors(
        torch.from_numpy(data),
        torch.tensor(symbols, dtype=torch.long),
        torch.tensor(directions, dtype=torch.long),
        torch.tensor(returns, dtype=torch.float32),
        torch.tensor(volatilities, dtype=torch.float32),
    )
