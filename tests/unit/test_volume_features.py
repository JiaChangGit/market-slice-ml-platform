from market_slice_ml.features.volume_features import add_volume_features


def test_volume_features_include_session_vwap(canonical_bars):
    result = add_volume_features(canonical_bars)
    assert {"vwap", "vwap_dev_pct", "obv", "relative_volume"}.issubset(result.columns)
