import polars as pl

from market_slice_ml.quality.duplicate_detector import flag_duplicates


def test_duplicate_detector_flags_both_observations(bars):
    result = flag_duplicates(pl.concat([bars.head(1), bars.head(1)]))
    assert result.get_column("duplicate").sum() == 2
