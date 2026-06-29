"""Quality and recency sample weighting."""

from __future__ import annotations

import math

import polars as pl


def add_slice_weights(frame: pl.DataFrame, decay_lambda: float = 0.02) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    maximum = frame.get_column("timestamp_utc").max()
    if maximum is None:
        return frame.with_columns(pl.lit(1.0).alias("final_training_weight"))
    age_days = (pl.lit(maximum) - pl.col("timestamp_utc")).dt.total_days()
    recency = (-decay_lambda * age_days).exp()
    quality = pl.col("quality_score") if "quality_score" in frame.columns else pl.lit(1.0)
    return frame.with_columns((quality * recency / math.e**0).alias("final_training_weight"))
