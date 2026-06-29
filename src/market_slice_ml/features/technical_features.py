"""Technical features computed only from current and past bars."""

from __future__ import annotations

import polars as pl


def add_technical_features(frame: pl.DataFrame) -> pl.DataFrame:
    close = pl.col("close")
    difference = close.diff().over("symbol")
    gain = difference.clip(lower_bound=0.0)
    loss = (-difference).clip(lower_bound=0.0)
    average_gain = gain.rolling_mean(14, min_samples=1).over("symbol")
    average_loss = loss.rolling_mean(14, min_samples=1).over("symbol")
    rsi = 100.0 - 100.0 / (1.0 + average_gain / average_loss.clip(lower_bound=1e-12))
    sma_5 = close.rolling_mean(5, min_samples=1).over("symbol")
    sma_20 = close.rolling_mean(20, min_samples=1).over("symbol")
    previous_close = close.shift(1).over("symbol")
    true_range = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - previous_close).abs(),
        (pl.col("low") - previous_close).abs(),
    )
    return frame.sort(["symbol", "timestamp_utc"]).with_columns(
        (close / close.shift(1)).log().over("symbol").alias("log_return_1"),
        (close / close.shift(3)).log().over("symbol").alias("log_return_3"),
        (close / close.shift(12)).log().over("symbol").alias("log_return_12"),
        rsi.alias("rsi_14"),
        close.ewm_mean(span=9).over("symbol").alias("ema_9"),
        close.ewm_mean(span=21).over("symbol").alias("ema_21"),
        close.ewm_mean(span=50).over("symbol").alias("ema_50"),
        true_range.rolling_mean(14, min_samples=1).over("symbol").alias("atr_14"),
        ((pl.col("high") - pl.col("low")) / close).alias("high_low_range_pct"),
        (close / sma_5 - 1.0).alias("sma_ratio_5"),
        (close / sma_20 - 1.0).alias("sma_ratio_20"),
    )
