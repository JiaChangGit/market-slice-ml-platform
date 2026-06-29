"""Deterministic in-memory canonical-like bars; no external access."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import polars as pl


def synthetic_bars(
    symbol: str = "NQ=F", rows: int = 500, provider: str = "synthetic"
) -> pl.DataFrame:
    start = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    timestamps = [start + timedelta(minutes=5 * index) for index in range(rows)]
    close = [100.0 + index * 0.01 + math.sin(index / 7.0) * 0.2 for index in range(rows)]
    return pl.DataFrame(
        {
            "timestamp_utc": timestamps,
            "symbol": [symbol] * rows,
            "open": [value - 0.02 for value in close],
            "high": [value + 0.08 for value in close],
            "low": [value - 0.08 for value in close],
            "close": close,
            "volume": [1000.0 + index % 20 * 10 for index in range(rows)],
            "provider": [provider] * rows,
        },
        schema_overrides={"timestamp_utc": pl.Datetime("us", "UTC")},
    )
