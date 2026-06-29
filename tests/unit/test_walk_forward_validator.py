from datetime import UTC, datetime, timedelta

from market_slice_ml.ml.evaluation.walk_forward_validator import WalkForwardFold, validate_folds


def test_walk_forward_fold_is_chronological():
    start = datetime(2024, 1, 1, tzinfo=UTC)
    train_end = start + timedelta(days=10)
    fold = WalkForwardFold("f1", start, train_end, train_end, start + timedelta(days=12))
    validate_folds([fold])
