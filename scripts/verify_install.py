#!/usr/bin/env python3
"""Strict Phase 0 installation gate."""

from __future__ import annotations

import importlib
import sys

REQUIRED = (
    "polars",
    "pandas",
    "pyarrow",
    "numpy",
    "scipy",
    "duckdb",
    "sqlalchemy",
    "pydantic",
    "pydantic_settings",
    "yaml",
    "typer",
    "rich",
    "lightgbm",
    "xgboost",
    "sklearn",
    "torch",
    "onnx",
    "onnxruntime",
    "onnxmltools",
    "plotly",
    "jinja2",
    "fastapi",
    "uvicorn",
    "yfinance",
    "pandas_datareader",
    "requests",
    "httpx",
    "dotenv",
)
OPTIONAL = (
    "torch_geometric",
    "pyg_lib",
    "torch_scatter",
    "torch_sparse",
    "ib_insync",
    "akshare",
)


def import_status(name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(name)
        return True, str(getattr(module, "__version__", "installed"))
    except Exception as exc:
        return False, str(exc)


def main() -> int:
    failed = sys.version_info[:2] != (3, 12)
    print(f"Python {sys.version.split()[0]} {'OK' if not failed else 'FAIL'}")
    for name in REQUIRED:
        ok, detail = import_status(name)
        failed = failed or not ok
        print(f"{'OK' if ok else 'FAIL':4} required {name:20} {detail}")
    for name in OPTIONAL:
        ok, detail = import_status(name)
        print(f"{'OK' if ok else 'WARN':4} optional {name:20} {detail}")
    if failed:
        return 1
    print("Installation verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
