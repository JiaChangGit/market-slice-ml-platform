from market_slice_ml.versioning.dataset_fingerprint import dataset_fingerprint


def test_fingerprint_is_order_independent_for_mappings():
    assert dataset_fingerprint({"b": 2, "a": 1}) == dataset_fingerprint({"a": 1, "b": 2})
