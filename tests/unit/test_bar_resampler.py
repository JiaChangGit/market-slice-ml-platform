from market_slice_ml.processing.bar_resampler import resample_30m


def test_resampler_requires_six_clean_bars(canonical_bars):
    complete = resample_30m(canonical_bars.head(6))
    assert not complete.item(0, "incomplete")
    incomplete = resample_30m(canonical_bars.head(5))
    assert incomplete.item(0, "incomplete")
