from market_slice_ml.features.technical_features import add_technical_features


def test_technical_features_use_historical_bars(canonical_bars):
    result = add_technical_features(canonical_bars)
    assert {
        "log_return_1",
        "log_return_3",
        "log_return_12",
        "rsi_14",
        "ema_9",
        "ema_21",
        "ema_50",
        "atr_14",
        "high_low_range_pct",
        "sma_ratio_20",
    }.issubset(result.columns)
    assert result.item(0, "log_return_1") is None
