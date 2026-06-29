from market_slice_ml.labels.label_builder import build_labels


def test_forward_volatility_label_is_nonnegative(bars):
    result = build_labels(bars, horizons={"h1": 12})
    values = result.get_column("forward_volatility_h1").drop_nulls()
    assert values.len() == bars.height - 12
    assert values.min() >= 0
