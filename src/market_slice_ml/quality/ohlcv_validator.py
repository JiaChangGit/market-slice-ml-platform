"""OHLCV invariant validation."""

from __future__ import annotations

import polars as pl


def validate_ohlcv(frame: pl.DataFrame) -> pl.DataFrame:
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing OHLCV columns: {sorted(missing)}")
    valid = (
        (pl.col("high") >= pl.col("low"))
        & pl.col("open").is_between(pl.col("low"), pl.col("high"))
        & pl.col("close").is_between(pl.col("low"), pl.col("high"))
        & (pl.col("volume") >= 0)
        & pl.all_horizontal(pl.col(column).is_not_null() for column in required)
    )
    return frame.with_columns(valid.alias("ohlcv_valid"))
