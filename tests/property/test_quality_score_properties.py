import polars as pl
from hypothesis import given
from hypothesis import strategies as st

from market_slice_ml.quality.quality_score import add_quality_score


@given(st.booleans(), st.booleans(), st.booleans())
def test_quality_score_is_always_bounded(missing, interpolated, suspicious):
    frame = pl.DataFrame(
        {
            "missing": [missing],
            "interpolated": [interpolated],
            "suspicious": [suspicious],
            "ohlcv_valid": [True],
        }
    )
    score = add_quality_score(frame).item(0, "quality_score")
    assert 0.0 <= score <= 1.0
