"""Extract every universe symbol over identical temporal boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from market_slice_ml.domain.exceptions import DataLeakError
from market_slice_ml.slicing.slice_models import TrainValPair


@dataclass(frozen=True)
class SynchronizedSlice:
    pair: TrainValPair
    train: dict[str, pl.DataFrame]
    validation: dict[str, pl.DataFrame]


def build_synchronized_slice(
    frames: dict[str, pl.DataFrame], pair: TrainValPair
) -> SynchronizedSlice:
    if pair.val_start_utc < pair.train_end_utc:
        raise DataLeakError("validation must start at or after training ends")
    train: dict[str, pl.DataFrame] = {}
    validation: dict[str, pl.DataFrame] = {}
    for symbol, frame in frames.items():
        train[symbol] = frame.filter(
            pl.col("timestamp_utc").is_between(
                pair.train_start_utc, pair.train_end_utc, closed="left"
            )
        )
        validation[symbol] = frame.filter(
            pl.col("timestamp_utc").is_between(pair.val_start_utc, pair.val_end_utc, closed="both")
        )
    return SynchronizedSlice(pair, train, validation)
