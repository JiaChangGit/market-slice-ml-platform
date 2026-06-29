from market_slice_ml.processing.gap_filler import fill_short_gaps


def test_short_gap_is_interpolated(bars):
    frame = bars.head(5).slice(0, 2).vstack(bars.head(5).slice(3, 2))
    result = fill_short_gaps(frame)
    assert result.height == 5
    assert result.get_column("interpolated").sum() == 1
