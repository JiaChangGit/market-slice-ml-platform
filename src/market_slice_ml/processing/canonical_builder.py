"""Construct canonical bars from multiple historical sources."""

from __future__ import annotations

import polars as pl

from market_slice_ml.normalization.bar_normalizer import normalize_bars
from market_slice_ml.normalization.session_classifier import with_session_labels
from market_slice_ml.processing.gap_filler import fill_short_gaps
from market_slice_ml.providers.provider_failover import select_best_bars
from market_slice_ml.quality.cross_provider_comparator import disagreement_flags
from market_slice_ml.quality.duplicate_detector import flag_duplicates
from market_slice_ml.quality.ohlcv_validator import validate_ohlcv
from market_slice_ml.quality.quality_score import add_quality_score
from market_slice_ml.quality.volume_anomaly_detector import flag_volume_anomalies


def build_canonical_bars(
    frames: list[pl.DataFrame],
    symbol: str,
    futures: bool = False,
    priorities: dict[str, int] | None = None,
) -> pl.DataFrame:
    normalized = [
        normalize_bars(frame, symbol, str(frame.item(0, "provider")))
        for frame in frames
        if not frame.is_empty()
    ]
    if not normalized:
        return pl.DataFrame()
    compared = disagreement_flags(pl.concat(normalized, how="diagonal_relaxed"))
    selected = select_best_bars(compared.partition_by("provider", maintain_order=True), priorities)
    if selected.is_empty():
        return selected
    result = with_session_labels(selected, futures=futures)
    if not futures:
        result = result.filter(pl.col("is_rth"))
    if result.is_empty():
        return result
    result = fill_short_gaps(result)
    result = result.with_columns(
        pl.col("suspicious").fill_null(False),
        pl.col("interpolated").fill_null(False),
        pl.col("missing").fill_null(False),
    )
    result = flag_duplicates(result)
    result = validate_ohlcv(result)
    result = flag_volume_anomalies(result)
    return add_quality_score(result).sort("timestamp_utc")
