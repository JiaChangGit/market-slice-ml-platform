from market_slice_ml.features.feature_builder import build_features


def test_feature_builder_composes_all_groups(canonical_bars):
    result = build_features(canonical_bars)
    assert {
        "rsi_14",
        "ema_21",
        "vwap",
        "realized_vol_12bar",
        "xasset_weighted_return_1",
        "time_sin",
    }.issubset(result.columns)
