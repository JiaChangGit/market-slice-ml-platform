from market_slice_ml.ml.ensemble.dynamic_ensemble_weighting import (
    DynamicEnsembleWeighting,
    normalize_weights,
)


def test_ensemble_weights_sum_exactly_to_one():
    weights = normalize_weights({"gbm": 0.7, "lstm": 0.5, "gnn": 0.4})
    assert sum(weights.values()) == 1.0
    assert all(value >= 0.0 for value in weights.values())


def test_unavailable_family_is_zero_and_remainder_is_renormalized():
    weighting = DynamicEnsembleWeighting()
    weights = weighting.update(
        "h1",
        {"gbm": 0.7, "lstm": 0.6, "gnn": 0.9},
        {"gbm": True, "lstm": True, "gnn": False},
    )
    assert weights["gnn"] == 0.0
    assert sum(weights.values()) == 1.0
