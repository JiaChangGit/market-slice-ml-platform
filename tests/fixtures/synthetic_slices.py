"""Deterministic non-overlapping slice definitions."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from market_slice_ml.slicing.slice_models import TrainValPair


def synthetic_pair(pair_id: str = "synthetic-pair") -> TrainValPair:
    start = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    return TrainValPair(
        pair_id=pair_id,
        anchor_symbol="NQ=F",
        train_start_utc=start,
        train_end_utc=start + timedelta(hours=12),
        val_start_utc=start + timedelta(hours=12),
        val_end_utc=start + timedelta(hours=24),
        target_symbols=("NQ=F", "AMD", "NVDA"),
    )
