"""Normalize every timestamp column to microsecond UTC."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

UTC_DTYPE = pl.Datetime("us", "UTC")


def normalize_timestamp(frame: pl.DataFrame, column: str = "timestamp_utc") -> pl.DataFrame:
    if column not in frame.columns:
        raise ValueError(f"missing timestamp column: {column}")
    dtype = frame.schema[column]
    expression = pl.col(column)
    if dtype == pl.String:
        expression = expression.str.to_datetime(time_zone="UTC", strict=True)
    elif isinstance(dtype, pl.Datetime) and dtype.time_zone is None:
        expression = expression.dt.replace_time_zone("UTC")
    elif isinstance(dtype, pl.Datetime):
        expression = expression.dt.convert_time_zone("UTC")
    elif dtype == pl.Date:
        expression = expression.cast(pl.Datetime).dt.replace_time_zone("UTC")
    else:
        raise TypeError(f"unsupported timestamp dtype: {dtype}")
    return frame.with_columns(expression.cast(UTC_DTYPE).alias(column))


def ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)
