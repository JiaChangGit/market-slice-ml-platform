from market_slice_ml.quality.ohlcv_validator import validate_ohlcv


def test_ohlcv_invariants_are_flagged(bars):
    valid = validate_ohlcv(bars)
    assert valid.get_column("ohlcv_valid").all()
    invalid = validate_ohlcv(bars.head(1).with_columns(high=1.0))
    assert not invalid.item(0, "ohlcv_valid")
