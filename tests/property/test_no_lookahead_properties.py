from datetime import UTC, datetime, timedelta

import pytest
from hypothesis import given
from hypothesis import strategies as st

from market_slice_ml.domain.exceptions import DataLeakError
from market_slice_ml.slicing.slice_models import TrainValPair


@given(st.integers(min_value=1, max_value=100))
def test_any_validation_overlap_is_rejected(overlap_minutes):
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(days=2)
    with pytest.raises(DataLeakError):
        TrainValPair(
            pair_id="property",
            anchor_symbol="AMD",
            train_start_utc=start,
            train_end_utc=end,
            val_start_utc=end - timedelta(minutes=overlap_minutes),
            val_end_utc=end + timedelta(days=1),
        )
