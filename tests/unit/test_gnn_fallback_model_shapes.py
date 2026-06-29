import torch

from market_slice_ml.ml.graph.temporal_gnn_model import UniversalTemporalGNNModel


def test_tensor_gnn_model_shapes():
    model = UniversalTemporalGNNModel(4, hidden_dim=16, heads=4, n_symbols=3)
    direction, returns, volatility = model.forward_tensor(
        torch.randn(3, 4), torch.arange(3), torch.ones(3, 3)
    )
    assert direction.shape == (3, 3)
    assert returns.shape == volatility.shape == (3,)
