import torch

from market_slice_ml.datasets.graph_dataset_builder import build_graph_dataset


def test_graph_builder_has_symbol_index_per_node(labeled_bars):
    graphs = build_graph_dataset(
        {"NQ=F": labeled_bars, "AMD": labeled_bars}, ["close", "volume"], torch.eye(2)
    )
    assert graphs[0]["node_features"].shape == (2, 2)
    assert graphs[0]["symbol_idx"].tolist() == [0, 1]
