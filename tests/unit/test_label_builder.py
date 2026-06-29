from market_slice_ml.labels.label_builder import build_labels


def test_label_builder_produces_all_three_heads_for_each_horizon(bars):
    result = build_labels(bars)
    for horizon in ("h1", "h2", "h3"):
        assert f"direction_{horizon}" in result.columns
        assert f"forward_return_{horizon}" in result.columns
        assert f"forward_volatility_{horizon}" in result.columns
        values = set(result.get_column(f"direction_{horizon}").drop_nulls().unique())
        assert values <= {"bearish", "neutral", "bullish"}


def test_labels_are_null_without_enough_future_observations(bars):
    result = build_labels(bars, horizons={"h1": 12})
    assert result.tail(12).get_column("forward_return_h1").null_count() == 12
    assert result.tail(12).get_column("direction_h1").null_count() == 12
    assert result.tail(12).get_column("forward_volatility_h1").null_count() == 12
