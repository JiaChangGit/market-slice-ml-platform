from hypothesis import given
from hypothesis import strategies as st

from market_slice_ml.ml.ensemble.dynamic_ensemble_weighting import normalize_weights


@given(st.lists(st.floats(min_value=0, max_value=100, allow_nan=False), min_size=3, max_size=3))
def test_normalized_ensemble_weight_sum_is_exact(scores):
    weights = normalize_weights(dict(zip(("gbm", "lstm", "gnn"), scores, strict=True)))
    assert sum(weights.values()) == 1.0
