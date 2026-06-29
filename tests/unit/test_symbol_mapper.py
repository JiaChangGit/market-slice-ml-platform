from market_slice_ml.normalization.symbol_mapper import canonical_symbol, to_stooq_symbol


def test_symbol_mapper_handles_us_equities():
    assert to_stooq_symbol("AAPL") == "AAPL.US"
    assert canonical_symbol("aapl.us") == "AAPL"
