"""Relationship-weighted context features."""

from __future__ import annotations

import polars as pl


def add_cross_asset_features(
    frame: pl.DataFrame,
    context: dict[str, pl.DataFrame] | None = None,
    weights: dict[str, float] | None = None,
) -> pl.DataFrame:
    if not context:
        return frame.with_columns(
            pl.lit(0.0).alias("xasset_weighted_return_1"),
            pl.lit(0.0).alias("xasset_weighted_return_12"),
        )
    configured = weights or {symbol: 1.0 / len(context) for symbol in context}
    parts: list[pl.DataFrame] = []
    for symbol, source in context.items():
        weight = configured.get(symbol, 0.0)
        if source.is_empty():
            continue
        close = pl.col("close")
        parts.append(
            source.sort("timestamp_utc").select(
                "timestamp_utc",
                (close.pct_change() * weight).alias("weighted_1"),
                (close.pct_change(12) * weight).alias("weighted_12"),
            )
        )
    if not parts:
        return add_cross_asset_features(frame)
    aggregated = (
        pl.concat(parts)
        .group_by("timestamp_utc")
        .agg(
            pl.col("weighted_1").sum().alias("xasset_weighted_return_1"),
            pl.col("weighted_12").sum().alias("xasset_weighted_return_12"),
        )
    )
    return frame.join(aggregated, on="timestamp_utc", how="left")
