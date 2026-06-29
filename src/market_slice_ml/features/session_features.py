"""Cyclical session/time features."""

from __future__ import annotations

import math

import polars as pl


def add_session_features(frame: pl.DataFrame) -> pl.DataFrame:
    local = pl.col("timestamp_utc").dt.convert_time_zone("America/New_York")
    minute = local.dt.hour() * 60 + local.dt.minute()
    fraction = minute / (24 * 60)
    session_encoding = pl.col("session_label").replace_strict(
        {"asia_session": 0, "europe_session": 1, "us_session": 2}, default=2
    )
    return frame.with_columns(
        session_encoding.cast(pl.Int8).alias("session_label_enc"),
        pl.int_range(pl.len()).over(["symbol", local.dt.date()]).alias("bar_index_in_session"),
        (fraction * 2 * math.pi).sin().alias("time_sin"),
        (fraction * 2 * math.pi).cos().alias("time_cos"),
        (local.dt.weekday() * 2 * math.pi / 7).sin().alias("day_of_week_sin"),
        (local.dt.weekday() * 2 * math.pi / 7).cos().alias("day_of_week_cos"),
    )
