"""Select the best available source at each timestamp."""

from __future__ import annotations

import polars as pl


def select_best_bars(
    frames: list[pl.DataFrame], priorities: dict[str, int] | None = None
) -> pl.DataFrame:
    usable = [frame for frame in frames if not frame.is_empty()]
    if not usable:
        return pl.DataFrame()
    priority = priorities or {}
    combined = pl.concat(usable, how="diagonal_relaxed")
    if "quality_score" not in combined.columns:
        combined = combined.with_columns(pl.lit(1.0).alias("quality_score"))
    combined = combined.with_columns(
        pl.col("provider")
        .replace_strict(priority, default=100, return_dtype=pl.Int64)
        .alias("_priority")
    )
    return (
        combined.sort(
            ["symbol", "timestamp_utc", "quality_score", "_priority"],
            descending=[False, False, True, False],
        )
        .unique(["symbol", "timestamp_utc"], keep="first", maintain_order=True)
        .drop("_priority")
    )
