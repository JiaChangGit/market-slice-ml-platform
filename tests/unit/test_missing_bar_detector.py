from datetime import timedelta

from market_slice_ml.quality.missing_bar_detector import missing_timestamps


def test_missing_bar_detector_finds_expected_gap(bars):
    expected = bars.get_column("timestamp_utc").head(3).to_list()
    result = missing_timestamps(bars.head(1), expected)
    assert result == expected[1:]
    assert expected[1] - expected[0] == timedelta(minutes=5)
