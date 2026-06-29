import polars as pl

from market_slice_ml.normalization.timezone_normalizer import normalize_timestamp


def test_timezone_normalizer_outputs_microsecond_utc():
    result = normalize_timestamp(pl.DataFrame({"timestamp_utc": ["2024-01-01T00:00:00Z"]}))
    assert result.schema["timestamp_utc"] == pl.Datetime("us", "UTC")
