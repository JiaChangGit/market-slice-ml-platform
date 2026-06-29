"""LSTM training and tracing-friendly ONNX export."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from market_slice_ml.ml.sequence.lstm_model import UniversalLSTMModel


def multi_task_loss(
    outputs: dict[str, Tensor], direction: Tensor, returns: Tensor, volatility: Tensor
) -> Tensor:
    return (
        nn.functional.cross_entropy(outputs["direction"], direction)
        + 0.5 * nn.functional.mse_loss(outputs["return"], returns)
        + 0.5
        * nn.functional.mse_loss(
            torch.log(outputs["volatility"] + 1e-8), torch.log(volatility + 1e-8)
        )
    )


def train_lstm(
    model: UniversalLSTMModel,
    loader: DataLoader[tuple[Tensor, ...]],
    epochs: int = 1,
    device: torch.device | None = None,
) -> float:
    selected = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(selected)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    final_loss = 0.0
    for _ in range(epochs):
        model.train()
        for features, symbol_idx, direction, returns, volatility in loader:
            optimizer.zero_grad()
            outputs = model(features.to(selected), symbol_idx.to(selected))
            loss = multi_task_loss(
                outputs,
                direction.to(selected),
                returns.to(selected),
                volatility.to(selected),
            )
            loss.backward()  # type: ignore[no-untyped-call]  # PyTorch stubs omit backward typing
            optimizer.step()
            final_loss = float(loss.detach().cpu())
        scheduler.step()
    return final_loss


class _OnnxLSTM(nn.Module):
    def __init__(self, model: UniversalLSTMModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: Tensor, symbol_idx: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        output = self.model(x, symbol_idx)
        return output["direction"], output["return"], output["volatility"]


def export_lstm_to_onnx(
    model: UniversalLSTMModel,
    seq_len: int,
    n_features: int,
    out_path: str | Path,
    device: torch.device | None = None,
) -> Path:
    selected = device or torch.device("cpu")
    wrapper = _OnnxLSTM(model.to(selected).eval())
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (
            torch.zeros(1, seq_len, n_features, device=selected),
            torch.zeros(1, dtype=torch.long, device=selected),
        ),
        destination,
        input_names=["x", "symbol_idx"],
        output_names=["direction_logits", "return_pred", "vol_pred"],
        dynamic_axes={
            "x": {0: "batch_size"},
            "symbol_idx": {0: "batch_size"},
            "direction_logits": {0: "batch_size"},
            "return_pred": {0: "batch_size"},
            "vol_pred": {0: "batch_size"},
        },
        opset_version=17,
        dynamo=False,
    )
    return destination
