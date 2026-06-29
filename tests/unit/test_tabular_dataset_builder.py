from market_slice_ml.datasets.tabular_dataset_builder import build_tabular_dataset


def test_tabular_builder_adds_symbol_index_and_weight(labeled_bars):
    result = build_tabular_dataset({"NQ=F": labeled_bars, "AMD": labeled_bars})
    assert {"symbol_idx", "sample_weight"}.issubset(result.columns)
    assert result.get_column("symbol_idx").n_unique() == 2
