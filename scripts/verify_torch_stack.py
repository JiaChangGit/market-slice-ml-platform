#!/usr/bin/env python3
"""Strictly verify the required ML runtime stack."""

from __future__ import annotations

import importlib
import platform
import tempfile
from pathlib import Path


def main() -> int:
    failures: list[str] = []
    if platform.python_version_tuple()[:2] != ("3", "12"):
        failures.append(f"Python 3.12.x required; found {platform.python_version()}")
    try:
        import torch
        from torch import nn
    except Exception as exc:
        failures.append(f"torch import failed: {exc}")
        print("\n".join(failures))
        return 1

    cuda_available = bool(torch.cuda.is_available())
    print(f"torch={torch.__version__} cuda_available={cuda_available}")
    if cuda_available:
        names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        print(f"GPUs: {names}")
        if not names:
            failures.append("CUDA is available but no GPU name was reported")

    try:
        import torch_geometric

        print(f"torch_geometric={torch_geometric.__version__}")
    except Exception:
        print("torch_geometric missing; pure PyTorch GNN fallback will be used")

    for module_name in ("onnx", "onnxruntime", "onnxmltools", "lightgbm"):
        try:
            module = importlib.import_module(module_name)
            print(f"{module_name}={getattr(module, '__version__', 'installed')}")
        except Exception as exc:
            failures.append(f"{module_name} import failed: {exc}")

    device = torch.device("cuda" if cuda_available else "cpu")
    result = (torch.ones(2, device=device) + 1).cpu().tolist()
    if result != [2.0, 2.0]:
        failures.append("tiny tensor operation returned the wrong result")

    if not failures:
        try:
            import onnx
            import onnxruntime as ort

            model = nn.Linear(2, 1).eval()
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "linear.onnx"
                torch.onnx.export(
                    model,
                    torch.zeros(1, 2),
                    path,
                    input_names=["x"],
                    output_names=["y"],
                    opset_version=17,
                )
                onnx.checker.check_model(onnx.load(path))
                session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
                session.run(None, {"x": [[0.0, 0.0]]})
            print("ONNX export/import/runtime smoke test: OK")
        except Exception as exc:
            failures.append(f"ONNX smoke test failed: {exc}")

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        return 1
    print("Torch stack verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
