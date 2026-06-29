"""Session and rolling volume features."""

from __future__ import annotations

import polars as pl


def add_volume_features(frame: pl.DataFrame) -> pl.DataFrame:
    result = frame.with_columns(pl.col("timestamp_utc").dt.date().alias("_session_day"))
    typical = (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
    cumulative_value = (typical * pl.col("volume")).cum_sum().over(["symbol", "_session_day"])
    cumulative_volume = pl.col("volume").cum_sum().over(["symbol", "_session_day"])
    vwap = cumulative_value / cumulative_volume.clip(lower_bound=1e-12)
    direction = pl.col("close").diff().over("symbol").sign().fill_null(0.0)
    obv = (direction * pl.col("volume")).cum_sum().over("symbol")
    volume = pl.col("volume")
    relative_20d = volume / volume.rolling_mean(1560, min_samples=1).over("symbol")
    return result.with_columns(
        vwap.alias("vwap"),
        ((pl.col("close") - vwap) / vwap).alias("vwap_dev_pct"),
        obv.alias("obv"),
        relative_20d.alias("relative_volume"),
    ).drop("_session_day")
