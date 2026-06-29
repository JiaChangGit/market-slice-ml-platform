import onnx
import onnxruntime as ort
import torch

from market_slice_ml.ml.graph.gnn_trainer import export_gnn_to_onnx
from market_slice_ml.ml.graph.temporal_gnn_model import UniversalTemporalGNNModel


def test_gnn_tensor_onnx_export_and_runtime(tmp_path):
    model = UniversalTemporalGNNModel(4, hidden_dim=8, heads=2, n_symbols=3)
    path = export_gnn_to_onnx(model, 3, 4, tmp_path / "gnn.onnx", torch.device("cpu"))
    onnx.checker.check_model(onnx.load(path))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert len(session.get_inputs()) == 3
