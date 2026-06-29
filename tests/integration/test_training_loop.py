import torch

from market_slice_ml.ml.graph.gnn_trainer import train_gnn_snapshot
from market_slice_ml.ml.graph.temporal_gnn_model import UniversalTemporalGNNModel


def test_tiny_training_loop_updates_graph_model():
    model = UniversalTemporalGNNModel(2, hidden_dim=8, heads=2, n_symbols=3)
    before = model.return_head.weight.detach().clone()
    train_gnn_snapshot(
        model,
        torch.randn(3, 2),
        torch.arange(3),
        torch.ones(3, 3),
        torch.arange(3),
        torch.zeros(3),
        torch.full((3,), 0.2),
    )
    assert not torch.equal(before, model.return_head.weight.detach())
