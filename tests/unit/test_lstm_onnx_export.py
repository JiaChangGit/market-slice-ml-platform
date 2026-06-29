import onnx
import onnxruntime as ort
import torch

from market_slice_ml.ml.sequence.lstm_model import UniversalLSTMModel
from market_slice_ml.ml.sequence.lstm_trainer import export_lstm_to_onnx


def test_lstm_onnx_export_and_runtime(tmp_path):
    model = UniversalLSTMModel(3, hidden_size=8, n_layers=1, n_symbols=3)
    path = export_lstm_to_onnx(model, 6, 3, tmp_path / "lstm.onnx", torch.device("cpu"))
    onnx.checker.check_model(onnx.load(path))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert len(session.get_outputs()) == 3
