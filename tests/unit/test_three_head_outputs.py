import torch

from market_slice_ml.ml.graph.temporal_gnn_model import UniversalTemporalGNNModel
from market_slice_ml.ml.sequence.lstm_model import UniversalLSTMModel


def test_neural_models_share_three_head_names():
    lstm = UniversalLSTMModel(2, hidden_size=8, n_layers=1, n_symbols=2)
    gnn = UniversalTemporalGNNModel(2, hidden_dim=8, heads=2, n_symbols=2)
    assert set(lstm(torch.zeros(1, 4, 2), torch.zeros(1, dtype=torch.long))) == {
        "direction",
        "return",
        "volatility",
    }
    assert set(gnn(torch.zeros(2, 2), torch.arange(2), torch.eye(2))) == {
        "direction",
        "return",
        "volatility",
    }
