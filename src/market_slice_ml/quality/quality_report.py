"""Aggregate canonical quality summary."""

from __future__ import annotations

import polars as pl


def build_quality_report(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.group_by("symbol").agg(
        pl.len().alias("bar_count"),
        pl.col("quality_score").mean().alias("quality_mean"),
        (1.0 - pl.col("missing").mean()).alias("coverage_ratio"),
        pl.col("missing").sum().alias("gap_count"),
        pl.col("suspicious").sum().alias("anomaly_count"),
    )
