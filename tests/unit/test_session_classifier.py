from datetime import UTC, datetime

import polars as pl

from market_slice_ml.normalization.session_classifier import with_session_labels


def test_futures_session_classifier(bars):
    result = with_session_labels(bars, futures=True)
    assert set(result.get_column("session_label").unique()) <= {
        "asia_session",
        "europe_session",
        "us_session",
    }


def test_equity_rth_boundary_does_not_overflow_hour_math():
    frame = pl.DataFrame(
        {
            "timestamp_utc": [
                datetime(2024, 1, 2, 14, 29, tzinfo=UTC),
                datetime(2024, 1, 2, 14, 30, tzinfo=UTC),
                datetime(2024, 1, 2, 21, 0, tzinfo=UTC),
            ]
        },
        schema_overrides={"timestamp_utc": pl.Datetime("us", "UTC")},
    )
    assert with_session_labels(frame).get_column("is_rth").to_list() == [False, True, False]
