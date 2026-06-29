"""Flag zero and unusually large volume observations."""

from __future__ import annotations

import polars as pl


def flag_volume_anomalies(frame: pl.DataFrame, multiplier: float = 5.0) -> pl.DataFrame:
    rolling = pl.col("volume").rolling_mean(window_size=20, min_samples=1).over("symbol")
    return frame.with_columns(
        ((pl.col("volume") == 0) & pl.col("is_rth").fill_null(False)).alias("zero_volume_rth"),
        (pl.col("volume") > rolling * multiplier).fill_null(False).alias("volume_spike"),
    )
