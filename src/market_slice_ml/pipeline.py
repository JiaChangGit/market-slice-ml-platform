"""Deterministic full offline pipeline used for diagnostics and regression tests."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from market_slice_ml.datasets.graph_dataset_builder import build_graph_dataset
from market_slice_ml.datasets.sequence_dataset_builder import build_sequence_dataset
from market_slice_ml.datasets.tabular_dataset_builder import build_tabular_dataset
from market_slice_ml.domain.models import PredictionRecord
from market_slice_ml.features.feature_builder import build_features
from market_slice_ml.labels.label_builder import build_labels
from market_slice_ml.ml.baseline.tree_trainer import train_tree_models
from market_slice_ml.ml.ensemble.dynamic_ensemble_weighting import normalize_weights
from market_slice_ml.ml.ensemble.prediction_ensemble import ModelPrediction, ensemble_prediction
from market_slice_ml.ml.graph.gnn_trainer import train_gnn_snapshot
from market_slice_ml.ml.graph.temporal_gnn_model import UniversalTemporalGNNModel
from market_slice_ml.ml.sequence.lstm_model import UniversalLSTMModel
from market_slice_ml.ml.sequence.lstm_trainer import train_lstm
from market_slice_ml.processing.canonical_builder import build_canonical_bars
from market_slice_ml.reporting.html_report import write_html_report
from market_slice_ml.slicing.slice_models import TrainValPair
from market_slice_ml.slicing.synchronized_slice_builder import build_synchronized_slice


def _synthetic_bars(symbol: str, rows: int = 180) -> pl.DataFrame:
    start = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    timestamps = [start + timedelta(minutes=5 * index) for index in range(rows)]
    offset = {"NQ=F": 0.0, "AMD": 0.4, "NVDA": 0.8}[symbol]
    close = [
        100 + offset + 0.01 * index + math.sin(index / 5 + offset) * 0.3 for index in range(rows)
    ]
    return pl.DataFrame(
        {
            "timestamp_utc": timestamps,
            "symbol": [symbol] * rows,
            "open": [value - 0.02 for value in close],
            "high": [value + 0.08 for value in close],
            "low": [value - 0.08 for value in close],
            "close": close,
            "volume": [1000.0 + index % 30 * 5 for index in range(rows)],
            "provider": ["synthetic"] * rows,
        },
        schema_overrides={"timestamp_utc": pl.Datetime("us", "UTC")},
    )


def run_synthetic_smoke(data_root: Path) -> PredictionRecord:
    symbols = ["NQ=F", "AMD", "NVDA"]
    frames: dict[str, pl.DataFrame] = {}
    for symbol in symbols:
        canonical = build_canonical_bars(
            [_synthetic_bars(symbol)], symbol, futures=symbol.endswith("=F")
        )
        featured = build_features(canonical)
        frames[symbol] = build_labels(
            featured,
            horizons={"h1": 12},
            bullish_threshold=0.0001,
            bearish_threshold=-0.0001,
        )

    start = frames["NQ=F"].item(0, "timestamp_utc")
    pairs = [
        TrainValPair(
            pair_id="smoke-1",
            anchor_symbol="NQ=F",
            train_start_utc=start,
            train_end_utc=start + timedelta(minutes=5 * 80),
            val_start_utc=start + timedelta(minutes=5 * 80),
            val_end_utc=start + timedelta(minutes=5 * 120),
            target_symbols=tuple(symbols),
        ),
        TrainValPair(
            pair_id="smoke-2",
            anchor_symbol="NQ=F",
            train_start_utc=start,
            train_end_utc=start + timedelta(minutes=5 * 120),
            val_start_utc=start + timedelta(minutes=5 * 120),
            val_end_utc=start + timedelta(minutes=5 * 179),
            target_symbols=tuple(symbols),
        ),
    ]
    slices = [build_synchronized_slice(frames, pair) for pair in pairs]
    selected = slices[-1]
    feature_columns = [
        "log_return_1",
        "rsi_14",
        "realized_vol_12bar",
        "xasset_weighted_return_1",
    ]
    tabular = build_tabular_dataset(selected.train)
    usable = tabular.filter(pl.col("direction_h1").is_not_null())
    features = usable.select(feature_columns).fill_null(0.0).to_numpy().astype(np.float32)
    direction_map = {"bearish": 0, "neutral": 1, "bullish": 2}
    direction = np.asarray(
        [direction_map[value] for value in usable.get_column("direction_h1")], dtype=np.int64
    )
    returns = usable.get_column("forward_return_h1").to_numpy().astype(np.float32)
    volatility = (
        usable.get_column("forward_volatility_h1").fill_null(0.0).to_numpy().astype(np.float32)
    )
    train_tree_models(features, direction, returns, volatility, n_estimators=3)

    sequence = build_sequence_dataset(
        selected.train, feature_columns, horizon="h1", sequence_length=16
    )
    sequence_dataset = TensorDataset(
        sequence.features,
        sequence.symbol_idx,
        sequence.direction,
        sequence.forward_return,
        sequence.forward_volatility,
    )
    loader = DataLoader(sequence_dataset, batch_size=16, shuffle=False)
    lstm = UniversalLSTMModel(
        len(feature_columns), hidden_size=16, n_layers=1, n_symbols=len(symbols), embed_dim=4
    )
    train_lstm(lstm, loader, epochs=1, device=torch.device("cpu"))

    adjacency = torch.ones(len(symbols), len(symbols), dtype=torch.float32)
    graphs = build_graph_dataset(selected.train, feature_columns, adjacency)
    graph = graphs[-1]
    gnn = UniversalTemporalGNNModel(
        len(feature_columns), hidden_dim=16, heads=4, n_symbols=len(symbols), embed_dim=4
    )
    train_gnn_snapshot(
        gnn,
        graph["node_features"],
        graph["symbol_idx"],
        graph["adjacency_matrix"],
        torch.tensor([0, 1, 2]),
        torch.zeros(3),
        torch.full((3,), 0.1),
        epochs=1,
    )

    weights = normalize_weights({"gbm": 0.61, "lstm": 0.57, "gnn": 0.54})
    predictions = {
        "gbm": ModelPrediction(np.asarray([0.2, 0.3, 0.5]), 0.002, 0.21),
        "lstm": ModelPrediction(np.asarray([0.25, 0.25, 0.5]), 0.0015, 0.22),
        "gnn": ModelPrediction(np.asarray([0.3, 0.2, 0.5]), 0.001, 0.20),
    }
    prediction = ensemble_prediction("NQ=F", "h1", predictions, weights)
    write_html_report(
        data_root / "reports" / "smoke_report.html",
        [prediction],
        weights,
        slice_count=len(slices),
    )
    return prediction
