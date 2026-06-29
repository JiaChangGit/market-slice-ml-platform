"""Historical realized-volatility features."""

from __future__ import annotations

import math

import polars as pl


def add_volatility_features(frame: pl.DataFrame, vix: pl.DataFrame | None = None) -> pl.DataFrame:
    log_return = (pl.col("close") / pl.col("close").shift(1)).log().over("symbol")
    vol_12 = log_return.rolling_std(12, min_samples=2).over("symbol") * math.sqrt(252 * 78)
    vol_78 = log_return.rolling_std(78, min_samples=2).over("symbol") * math.sqrt(252 * 78)
    result = frame.with_columns(
        vol_12.alias("realized_vol_12bar"),
        vol_78.alias("realized_vol_78bar"),
    )
    if vix is None or vix.is_empty():
        return result.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("vix_level"),
        )
    context = vix.select(
        "timestamp_utc",
        pl.col("close").alias("vix_level"),
    )
    return result.join(context, on="timestamp_utc", how="left")
