import torch

from market_slice_ml.ml.graph.gnn_trainer import train_gnn_snapshot
from market_slice_ml.ml.graph.temporal_gnn_model import UniversalTemporalGNNModel
from tests.fixtures.synthetic_graph import synthetic_graph


def test_gnn_trainer_runs_one_tiny_epoch():
    graph = synthetic_graph()
    model = UniversalTemporalGNNModel(4, hidden_dim=8, heads=2, n_symbols=3)
    loss = train_gnn_snapshot(
        model,
        graph["node_features"],
        graph["symbol_idx"],
        graph["adjacency_matrix"],
        torch.arange(3),
        torch.zeros(3),
        torch.full((3,), 0.2),
    )
    assert loss >= 0
