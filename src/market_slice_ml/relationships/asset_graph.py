"""Tensor-friendly directed asset graph."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AssetGraph:
    symbols: tuple[str, ...]
    adjacency: np.ndarray


def build_asset_graph(symbols: list[str], weights: dict[tuple[str, str], float]) -> AssetGraph:
    index = {symbol: position for position, symbol in enumerate(symbols)}
    adjacency = np.eye(len(symbols), dtype=np.float32)
    for (source, target), weight in weights.items():
        if source in index and target in index and weight > 0:
            adjacency[index[target], index[source]] = weight
    return AssetGraph(tuple(symbols), adjacency)
