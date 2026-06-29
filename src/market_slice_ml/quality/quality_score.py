"""Deterministic per-bar quality scores."""

from __future__ import annotations

import polars as pl


def add_quality_score(frame: pl.DataFrame) -> pl.DataFrame:
    def boolean(name: str) -> pl.Expr:
        return pl.col(name).fill_null(False) if name in frame.columns else pl.lit(False)

    score = (
        pl.lit(1.0)
        - boolean("missing").cast(pl.Float64) * 1.0
        - boolean("interpolated").cast(pl.Float64) * 0.25
        - boolean("suspicious").cast(pl.Float64) * 0.30
        - boolean("duplicate").cast(pl.Float64) * 0.20
        - boolean("zero_volume_rth").cast(pl.Float64) * 0.10
        - boolean("volume_spike").cast(pl.Float64) * 0.10
        - (~boolean("ohlcv_valid")).cast(pl.Float64) * 0.50
    ).clip(0.0, 1.0)
    flag_names = [
        name
        for name in (
            "missing",
            "interpolated",
            "suspicious",
            "duplicate",
            "zero_volume_rth",
            "volume_spike",
        )
        if name in frame.columns
    ]
    flags = pl.concat_list(
        [pl.when(pl.col(name)).then(pl.lit(name)).otherwise(pl.lit(None)) for name in flag_names]
    ).list.drop_nulls()
    return frame.with_columns(score.alias("quality_score"), flags.alias("quality_flags"))
