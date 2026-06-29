"""Derive 30-minute bars from six clean canonical observations."""

from __future__ import annotations

import polars as pl


def resample_30m(frame: pl.DataFrame) -> pl.DataFrame:
    clean = (~pl.col("missing")) & (~pl.col("interpolated"))
    return (
        frame.sort("timestamp_utc")
        .group_by_dynamic("timestamp_utc", every="30m", period="30m", group_by="symbol")
        .agg(
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum(),
            clean.sum().alias("clean_bar_count"),
        )
        .with_columns((pl.col("clean_bar_count") < 6).alias("incomplete"))
    )
