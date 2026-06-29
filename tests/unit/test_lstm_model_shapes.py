import torch

from market_slice_ml.ml.sequence.lstm_model import UniversalLSTMModel


def test_lstm_model_shapes_for_variable_batch():
    model = UniversalLSTMModel(6, hidden_size=16, n_layers=1, n_symbols=4)
    output = model(torch.randn(3, 12, 6), torch.tensor([0, 1, 3]))
    assert output["direction"].shape == (3, 3)
    assert output["return"].shape == (3,)
    assert output["volatility"].shape == (3,)
