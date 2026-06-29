import torch
from torch import nn

from market_slice_ml.ml.graph.temporal_gnn_model import UniversalTemporalGNNModel
from market_slice_ml.ml.sequence.lstm_model import UniversalLSTMModel


def test_lstm_uses_learnable_asset_embedding():
    model = UniversalLSTMModel(input_size=5, hidden_size=16, n_layers=1, n_symbols=3)
    assert isinstance(model.asset_embedding, nn.Embedding)
    output = model(torch.zeros(2, 8, 5), torch.tensor([0, 2]))
    assert output["direction"].shape == (2, 3)
    assert output["return"].shape == (2,)
    assert output["volatility"].shape == (2,)


def test_gnn_uses_learnable_asset_embedding():
    model = UniversalTemporalGNNModel(node_feature_dim=4, hidden_dim=16, heads=4, n_symbols=3)
    assert isinstance(model.asset_embedding, nn.Embedding)
    output = model(torch.zeros(3, 4), torch.arange(3), torch.eye(3))
    assert output["direction"].shape == (3, 3)
    assert output["return"].shape == (3,)
    assert output["volatility"].shape == (3,)
