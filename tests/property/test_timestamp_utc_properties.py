from datetime import datetime
from zoneinfo import ZoneInfo

import polars as pl
from hypothesis import given
from hypothesis import strategies as st

from market_slice_ml.normalization.timezone_normalizer import normalize_timestamp


@given(
    st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2030, 1, 1),
        timezones=st.sampled_from(
            [ZoneInfo("UTC"), ZoneInfo("America/New_York"), ZoneInfo("Asia/Taipei")]
        ),
    )
)
def test_any_aware_timestamp_normalizes_to_utc(value):
    frame = pl.DataFrame({"timestamp_utc": [value]})
    result = normalize_timestamp(frame)
    assert result.schema["timestamp_utc"] == pl.Datetime("us", "UTC")
