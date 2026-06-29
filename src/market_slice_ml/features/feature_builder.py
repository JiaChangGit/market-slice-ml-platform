"""Full leakage-safe feature pipeline."""

from __future__ import annotations

import polars as pl

from market_slice_ml.features.cross_asset_features import add_cross_asset_features
from market_slice_ml.features.session_features import add_session_features
from market_slice_ml.features.technical_features import add_technical_features
from market_slice_ml.features.volatility_features import add_volatility_features
from market_slice_ml.features.volume_features import add_volume_features


def build_features(
    frame: pl.DataFrame,
    context: dict[str, pl.DataFrame] | None = None,
    relationship_weights: dict[str, float] | None = None,
    vix: pl.DataFrame | None = None,
) -> pl.DataFrame:
    result = add_technical_features(frame)
    result = add_volume_features(result)
    result = add_volatility_features(result, vix)
    result = add_cross_asset_features(result, context, relationship_weights)
    return add_session_features(result)
