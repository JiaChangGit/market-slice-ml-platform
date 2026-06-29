"""Symbol coverage for synchronized slices."""

from __future__ import annotations

import polars as pl

from market_slice_ml.slicing.synchronized_slice_builder import SynchronizedSlice


def _quality_mean(frame: pl.DataFrame) -> float:
    if "quality_score" not in frame.columns:
        return 1.0
    value = frame.get_column("quality_score").mean()
    return float(value) if isinstance(value, (int, float)) else 0.0


def slice_coverage(sliced: SynchronizedSlice) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": list(sliced.train),
            "train_rows": [frame.height for frame in sliced.train.values()],
            "val_rows": [frame.height for frame in sliced.validation.values()],
            "quality_mean": [_quality_mean(frame) for frame in sliced.train.values()],
        }
    )
