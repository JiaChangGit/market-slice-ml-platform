"""Three-head, forward-looking labels built separately from features."""

from __future__ import annotations

import math

import polars as pl

DEFAULT_HORIZONS = {"h1": 12, "h2": 78, "h3": 390}


def build_labels(
    frame: pl.DataFrame,
    horizons: dict[str, int] | None = None,
    bullish_threshold: float = 0.003,
    bearish_threshold: float = -0.003,
) -> pl.DataFrame:
    result = frame.sort(["symbol", "timestamp_utc"])
    configured = horizons or DEFAULT_HORIZONS
    for name, bars in configured.items():
        future_close = pl.col("close").shift(-bars).over("symbol")
        forward = (future_close - pl.col("close")) / pl.col("close")
        log_return = (pl.col("close") / pl.col("close").shift(1)).log().over("symbol")
        future_volatility = log_return.rolling_std(bars, min_samples=bars).shift(-bars).over(
            "symbol"
        ) * math.sqrt(252 * 78 / bars)
        result = result.with_columns(
            forward.alias(f"forward_return_{name}"),
            future_volatility.alias(f"forward_volatility_{name}"),
        )
        result = result.with_columns(
            pl.when(pl.col(f"forward_return_{name}").is_null())
            .then(pl.lit(None, dtype=pl.String))
            .when(pl.col(f"forward_return_{name}") > bullish_threshold)
            .then(pl.lit("bullish"))
            .when(pl.col(f"forward_return_{name}") < bearish_threshold)
            .then(pl.lit("bearish"))
            .otherwise(pl.lit("neutral"))
            .alias(f"direction_{name}")
        )
    return result
