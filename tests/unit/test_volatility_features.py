from market_slice_ml.features.volatility_features import add_volatility_features


def test_volatility_features_are_nonnegative(canonical_bars):
    result = add_volatility_features(canonical_bars)
    assert "realized_vol_12bar" in result.columns
    assert result.get_column("realized_vol_12bar").drop_nulls().min() >= 0
