from market_slice_ml.features.session_features import add_session_features


def test_session_features_are_cyclical(canonical_bars):
    result = add_session_features(canonical_bars)
    assert result.get_column("time_sin").is_between(-1.0, 1.0).all()
    assert result.get_column("time_cos").is_between(-1.0, 1.0).all()
