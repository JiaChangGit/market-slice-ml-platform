"""Flag close-price disagreement between sources."""

from __future__ import annotations

import polars as pl


def disagreement_flags(frame: pl.DataFrame, threshold: float = 0.005) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    keys = ["symbol", "timestamp_utc"]
    spread = pl.col("close").max().over(keys) - pl.col("close").min().over(keys)
    return frame.with_columns(
        (spread / pl.col("close").mean().over(keys) > threshold)
        .fill_null(False)
        .alias("suspicious")
    )
