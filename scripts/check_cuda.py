#!/usr/bin/env python3
"""Print a non-failing diagnostic for the local ML runtime."""

from __future__ import annotations

import importlib
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass

PYG_WHEEL_MATRIX: dict[str, list[str]] = {
    "2.12": ["cpu", "cu126", "cu130", "cu132"],
    "2.11": ["cpu", "cu126", "cu128", "cu130"],
    "2.10": ["cpu", "cu126", "cu128", "cu130"],
    "2.9": ["cpu", "cu126", "cu128", "cu130"],
    "2.8": ["cpu", "cu126", "cu128", "cu129"],
    "2.7": ["cpu", "cu118", "cu126", "cu128"],
    "2.6": ["cpu", "cu118", "cu124", "cu126"],
    "2.5": ["cpu", "cu118", "cu121", "cu124"],
    "2.4": ["cpu", "cu118", "cu121", "cu124"],
    "2.3": ["cpu", "cu118", "cu121"],
}


@dataclass(frozen=True)
class Diagnostic:
    python: str
    python_executable: str
    platform: str
    is_wsl2: bool
    nvidia_smi: str | None
    nvcc: str | None
    torch: str | None
    torch_cuda: str | None
    torch_cuda_available: bool
    cudnn: int | None
    gpu_names: list[str]
    pyg: str | None
    pyg_lib: str | None
    torch_scatter: str | None
    torch_sparse: str | None
    onnxruntime: str | None
    lightgbm: str | None
    recommended_pyg_wheel: str | None


def run_cmd(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def package_version(module_name: str) -> str | None:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    return str(getattr(module, "__version__", "installed"))


def is_wsl2() -> bool:
    release = platform.uname().release.lower()
    return "microsoft" in release or "wsl" in release


def parse_nvcc_release(text: str | None) -> str | None:
    if not text:
        return None
    for line in text.splitlines():
        if "release" in line:
            return line.split("release", 1)[1].split(",", 1)[0].strip()
    return None


def cuda_version_to_tag(cuda_version: str | None) -> str:
    if not cuda_version:
        return "cpu"
    parts = cuda_version.split(".")[:2]
    return f"cu{parts[0]}{parts[1]}" if len(parts) == 2 else "cpu"


def torch_to_pyg_torch_tag(torch_version: str | None) -> tuple[str | None, str | None]:
    if not torch_version:
        return None, None
    parts = torch_version.split("+", 1)[0].split(".")
    if len(parts) < 2:
        return None, None
    minor = f"{parts[0]}.{parts[1]}"
    return minor, f"{minor}.0"


def recommend_pyg_wheel(torch_version: str | None, torch_cuda: str | None) -> str | None:
    torch_minor, torch_wheel = torch_to_pyg_torch_tag(torch_version)
    if not torch_minor or not torch_wheel:
        return None
    supported = PYG_WHEEL_MATRIX.get(torch_minor)
    if not supported:
        return None
    preferred = cuda_version_to_tag(torch_cuda)
    chosen = preferred if preferred in supported else "cpu"
    return f"https://data.pyg.org/whl/torch-{torch_wheel}+{chosen}.html"


def collect_gpu_names() -> list[str]:
    output = run_cmd(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    return [] if not output else [line.strip() for line in output.splitlines() if line.strip()]


def main() -> None:
    torch_version = package_version("torch")
    torch_cuda: str | None = None
    torch_cuda_available = False
    cudnn: int | None = None
    try:
        import torch

        torch_cuda = torch.version.cuda
        torch_cuda_available = bool(torch.cuda.is_available())
        cudnn = torch.backends.cudnn.version()
    except Exception:
        torch_version = None

    diag = Diagnostic(
        python=sys.version.split()[0],
        python_executable=sys.executable,
        platform=f"{platform.system()} {platform.release()}",
        is_wsl2=is_wsl2(),
        nvidia_smi=shutil.which("nvidia-smi"),
        nvcc=parse_nvcc_release(run_cmd(["nvcc", "--version"])),
        torch=torch_version,
        torch_cuda=torch_cuda,
        torch_cuda_available=torch_cuda_available,
        cudnn=cudnn,
        gpu_names=collect_gpu_names(),
        pyg=package_version("torch_geometric"),
        pyg_lib=package_version("pyg_lib"),
        torch_scatter=package_version("torch_scatter"),
        torch_sparse=package_version("torch_sparse"),
        onnxruntime=package_version("onnxruntime"),
        lightgbm=package_version("lightgbm"),
        recommended_pyg_wheel=recommend_pyg_wheel(torch_version, torch_cuda),
    )
    print("=" * 72)
    print("market-slice-ml-platform environment diagnostic")
    print("=" * 72)
    print(json.dumps(asdict(diag), indent=2, ensure_ascii=False))
    if diag.recommended_pyg_wheel:
        print("\nRecommended optional PyG acceleration install:")
        print("  pip install torch_geometric")
        print(f"  pip install pyg_lib torch_scatter torch_sparse -f {diag.recommended_pyg_wheel}")
    else:
        print("\nPyG wheel recommendation unavailable. Use CPU fallback or update matrix.")
    print("\nDiagnostic complete.")


if __name__ == "__main__":
    main()
