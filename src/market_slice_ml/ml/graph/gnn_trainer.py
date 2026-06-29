"""Graph model training and tensor-signature ONNX export."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from market_slice_ml.ml.graph.temporal_gnn_model import UniversalTemporalGNNModel


def train_gnn(
    model: UniversalTemporalGNNModel,
    snapshots: list[dict[str, Tensor]],
    epochs: int = 1,
    device: torch.device | None = None,
) -> float:
    if not snapshots:
        raise ValueError("graph training requires at least one labeled snapshot")
    selected = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(selected)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    final_loss = 0.0
    for _ in range(epochs):
        for snapshot in snapshots:
            mask = snapshot["target_mask"].to(selected)
            if not bool(mask.any()):
                continue
            optimizer.zero_grad()
            output = model(
                snapshot["node_features"].to(selected),
                snapshot["symbol_idx"].to(selected),
                snapshot["adjacency_matrix"].to(selected),
            )
            volatility = snapshot["forward_volatility"].to(selected)[mask]
            loss = (
                nn.functional.cross_entropy(
                    output["direction"][mask], snapshot["direction"].to(selected)[mask]
                )
                + 0.5
                * nn.functional.mse_loss(
                    output["return"][mask], snapshot["forward_return"].to(selected)[mask]
                )
                + 0.5
                * nn.functional.mse_loss(
                    torch.log(output["volatility"][mask] + 1e-8),
                    torch.log(volatility + 1e-8),
                )
            )
            loss.backward()  # type: ignore[no-untyped-call]  # PyTorch stubs omit backward typing
            optimizer.step()
            final_loss = float(loss.detach().cpu())
    return final_loss


def train_gnn_snapshot(
    model: UniversalTemporalGNNModel,
    node_features: Tensor,
    symbol_idx: Tensor,
    adjacency_matrix: Tensor,
    direction: Tensor,
    returns: Tensor,
    volatility: Tensor,
    epochs: int = 1,
) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    final_loss = 0.0
    for _ in range(epochs):
        optimizer.zero_grad()
        output = model(node_features.to(device), symbol_idx.to(device), adjacency_matrix.to(device))
        loss = (
            nn.functional.cross_entropy(output["direction"], direction.to(device))
            + 0.5 * nn.functional.mse_loss(output["return"], returns.to(device))
            + 0.5
            * nn.functional.mse_loss(
                torch.log(output["volatility"] + 1e-8),
                torch.log(volatility.to(device) + 1e-8),
            )
        )
        loss.backward()  # type: ignore[no-untyped-call]  # PyTorch stubs omit backward typing
        optimizer.step()
        final_loss = float(loss.detach().cpu())
    return final_loss


class _OnnxGNN(nn.Module):
    def __init__(self, model: UniversalTemporalGNNModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, node_features: Tensor, symbol_idx: Tensor, adjacency_matrix: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        return self.model.forward_tensor(node_features, symbol_idx, adjacency_matrix)


def export_gnn_to_onnx(
    model: UniversalTemporalGNNModel,
    n_nodes: int,
    node_feature_dim: int,
    out_path: str | Path,
    device: torch.device | None = None,
) -> Path:
    selected = device or torch.device("cpu")
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    wrapper = _OnnxGNN(model.to(selected).eval())
    torch.onnx.export(
        wrapper,
        (
            torch.zeros(n_nodes, node_feature_dim, device=selected),
            torch.arange(n_nodes, dtype=torch.long, device=selected),
            torch.eye(n_nodes, device=selected),
        ),
        destination,
        input_names=["node_features", "symbol_idx", "adjacency_matrix"],
        output_names=["direction_logits", "return_pred", "vol_pred"],
        dynamic_axes={
            "node_features": {0: "n_nodes"},
            "symbol_idx": {0: "n_nodes"},
            "adjacency_matrix": {0: "n_nodes", 1: "n_nodes"},
            "direction_logits": {0: "n_nodes"},
            "return_pred": {0: "n_nodes"},
            "vol_pred": {0: "n_nodes"},
        },
        opset_version=17,
        dynamo=False,
    )
    return destination
