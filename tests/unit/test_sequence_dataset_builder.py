from market_slice_ml.datasets.sequence_dataset_builder import build_sequence_dataset


def test_sequence_builder_shapes(labeled_bars):
    result = build_sequence_dataset({"NQ=F": labeled_bars}, ["close", "volume"], sequence_length=16)
    assert result.features.ndim == 3
    assert result.features.shape[1:] == (16, 2)
    assert result.symbol_idx.shape[0] == result.features.shape[0]
