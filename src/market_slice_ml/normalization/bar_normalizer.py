"""Normalize provider frames into the canonical OHLCV schema."""

from __future__ import annotations

import polars as pl

from market_slice_ml.normalization.timezone_normalizer import normalize_timestamp

CANONICAL_COLUMNS = (
    "timestamp_utc",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "provider",
)


def normalize_bars(frame: pl.DataFrame, symbol: str, provider: str) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    rename = {column: column.lower().replace(" ", "_") for column in frame.columns}
    result = frame.rename(rename)
    if "datetime" in result.columns and "timestamp_utc" not in result.columns:
        result = result.rename({"datetime": "timestamp_utc"})
    if "date" in result.columns and "timestamp_utc" not in result.columns:
        result = result.rename({"date": "timestamp_utc"})
    missing = {"timestamp_utc", "open", "high", "low", "close", "volume"} - set(result.columns)
    if missing:
        raise ValueError(f"bar frame missing columns: {sorted(missing)}")
    if "symbol" not in result.columns:
        result = result.with_columns(pl.lit(symbol).alias("symbol"))
    if "provider" not in result.columns:
        result = result.with_columns(pl.lit(provider).alias("provider"))
    result = normalize_timestamp(result)
    return result.select(
        pl.col("timestamp_utc"),
        pl.col("symbol").cast(pl.String),
        *[pl.col(column).cast(pl.Float64) for column in ("open", "high", "low", "close", "volume")],
        pl.col("provider").cast(pl.String),
    ).sort(["symbol", "timestamp_utc"])
