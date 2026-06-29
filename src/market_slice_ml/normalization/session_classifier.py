"""Classify market sessions using America/New_York wall time."""

from __future__ import annotations

import polars as pl


def with_session_labels(frame: pl.DataFrame, futures: bool = False) -> pl.DataFrame:
    local = pl.col("timestamp_utc").dt.convert_time_zone("America/New_York")
    # Polars may expose hour/minute as UInt8; cast before multiplication to avoid overflow.
    minute = local.dt.hour().cast(pl.Int32) * 60 + local.dt.minute().cast(pl.Int32)
    if futures:
        label = (
            pl.when((minute >= 18 * 60) | (minute < 2 * 60))
            .then(pl.lit("asia_session"))
            .when(minute < 9 * 60 + 30)
            .then(pl.lit("europe_session"))
            .otherwise(pl.lit("us_session"))
        )
        rth = minute.is_between(9 * 60 + 30, 17 * 60, closed="left")
    else:
        label = pl.lit("us_session")
        rth = minute.is_between(9 * 60 + 30, 16 * 60, closed="left")
    return frame.with_columns(label.alias("session_label"), rth.alias("is_rth"))
