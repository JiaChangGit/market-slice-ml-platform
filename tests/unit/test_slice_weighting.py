from market_slice_ml.slicing.slice_weighting import add_slice_weights


def test_slice_weighting_combines_recency_and_quality(canonical_bars):
    result = add_slice_weights(canonical_bars)
    assert result.get_column("final_training_weight").min() >= 0
    assert result.item(-1, "final_training_weight") >= result.item(0, "final_training_weight")
