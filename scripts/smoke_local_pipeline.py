#!/usr/bin/env python3
"""Run the deterministic, network-free end-to-end research smoke flow."""

from __future__ import annotations

from pathlib import Path


def main() -> int:
    from market_slice_ml.pipeline import run_synthetic_smoke

    prediction = run_synthetic_smoke(Path("data"))
    report = Path("data/reports/smoke_report.html")
    if len(prediction.model_dump()) != 6:
        raise RuntimeError("prediction schema must contain exactly six fields")
    if not report.exists():
        raise RuntimeError("smoke report was not created")
    print(f"Smoke pipeline passed: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
