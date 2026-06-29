#!/usr/bin/env python3
"""Write a machine-readable local environment report."""

from __future__ import annotations

import importlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REQUIRED = ("polars", "pyarrow", "duckdb", "sqlalchemy", "torch", "onnx", "onnxruntime", "lightgbm")
OPTIONAL = ("torch_geometric", "pyg_lib", "torch_scatter", "torch_sparse", "xgboost")


def version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
        return str(getattr(module, "__version__", "installed"))
    except Exception:
        return None


def command(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    data_root = Path(os.getenv("DATA_ROOT", "data"))
    data_root.mkdir(parents=True, exist_ok=True)
    writable = os.access(data_root, os.W_OK)
    release = platform.uname().release.lower()
    packages = {name: version(name) for name in (*REQUIRED, *OPTIONAL)}
    torch_report: dict[str, object] = {}
    try:
        import torch

        torch_report = {
            "version": torch.__version__,
            "runtime_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cudnn": torch.backends.cudnn.version(),
            "gpus": [
                {
                    "name": torch.cuda.get_device_name(index),
                    "memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                }
                for index in range(torch.cuda.device_count())
            ],
        }
    except Exception as exc:
        torch_report = {"error": str(exc)}
    try:
        from detect_ibkr_host import probe_socket, resolve_ibkr_host

        ibkr_host = resolve_ibkr_host(os.getenv("IBKR_HOST", "auto"))
        ibkr_port = int(os.getenv("IBKR_PORT", "7497"))
        no_network = os.getenv("NO_NETWORK", "1") == "1"
        ibkr = {
            "host": ibkr_host,
            "port": ibkr_port,
            "reachable": None if no_network else probe_socket(ibkr_host, ibkr_port),
            "skipped": no_network,
        }
    except Exception as exc:
        ibkr = {"error": str(exc)}
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
            "venv": sys.prefix != sys.base_prefix,
        },
        "os": {
            "platform": platform.platform(),
            "kernel": platform.release(),
            "wsl2": "microsoft" in release or "wsl" in release,
        },
        "nvidia_smi": command(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]
        ),
        "nvcc": command(["nvcc", "--version"]),
        "torch": torch_report,
        "packages": packages,
        "data_root": {"path": str(data_root.resolve()), "writable": writable},
        "env_file_present": Path(".env").exists(),
        "ibkr_historical_probe_socket": ibkr,
    }
    output = data_root / "diagnostics" / "latest_env_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {output}")
    missing = [name for name in REQUIRED if packages[name] is None]
    if not writable:
        missing.append("writable DATA_ROOT")
    for name in OPTIONAL:
        if packages[name] is None:
            print(f"WARNING: optional dependency missing: {name}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
