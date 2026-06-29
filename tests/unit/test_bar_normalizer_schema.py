from market_slice_ml.normalization.bar_normalizer import CANONICAL_COLUMNS, normalize_bars


def test_bar_normalizer_schema(bars):
    result = normalize_bars(bars, "NQ=F", "synthetic")
    assert result.columns == list(CANONICAL_COLUMNS)
    assert result.schema["timestamp_utc"].time_zone == "UTC"
