from datetime import UTC, datetime

import pytest

from market_slice_ml.domain.exceptions import DataLeakError
from market_slice_ml.slicing.slice_models import TrainValPair


def test_overlapping_validation_raises_data_leak_error():
    with pytest.raises(DataLeakError):
        TrainValPair(
            pair_id="bad",
            anchor_symbol="NQ=F",
            train_start_utc=datetime(2024, 1, 1, tzinfo=UTC),
            train_end_utc=datetime(2024, 2, 1, tzinfo=UTC),
            val_start_utc=datetime(2024, 1, 20, tzinfo=UTC),
            val_end_utc=datetime(2024, 2, 20, tzinfo=UTC),
        )
