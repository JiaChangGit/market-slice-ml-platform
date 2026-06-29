import torch
from torch.utils.data import DataLoader, TensorDataset

from market_slice_ml.ml.sequence.lstm_model import UniversalLSTMModel
from market_slice_ml.ml.sequence.lstm_trainer import train_lstm


def test_lstm_trainer_runs_one_tiny_epoch():
    dataset = TensorDataset(
        torch.randn(9, 8, 3),
        torch.arange(9) % 3,
        torch.arange(9) % 3,
        torch.randn(9) * 0.01,
        torch.full((9,), 0.2),
    )
    model = UniversalLSTMModel(3, hidden_size=8, n_layers=1, n_symbols=3)
    loss = train_lstm(
        model,
        DataLoader(dataset, batch_size=3),
        epochs=1,
        device=torch.device("cpu"),
    )
    assert loss >= 0
