"""Temperature scaling and calibration artifact persistence."""

from __future__ import annotations

import json
import pickle
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


class ProbabilityCalibrator:
    def __init__(self) -> None:
        self.temperature = 1.0

    def fit(self, logits: NDArray[np.float64], targets: NDArray[np.int64]) -> ProbabilityCalibrator:
        best_loss = float("inf")
        for temperature in np.linspace(0.5, 3.0, 101):
            probabilities = self.transform(logits, float(temperature))
            loss = -np.log(probabilities[np.arange(len(targets)), targets] + 1e-12).mean()
            if loss < best_loss:
                best_loss = float(loss)
                self.temperature = float(temperature)
        return self

    def transform(
        self, logits: NDArray[np.float64], temperature: float | None = None
    ) -> NDArray[np.float64]:
        scaled = logits / (temperature or self.temperature)
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        exponential = np.exp(scaled)
        return exponential / exponential.sum(axis=1, keepdims=True)


def save_calibrator(
    calibrator: object,
    path: str | Path,
    *,
    model_id: str,
    calibration_method: str,
    val_metrics: dict[str, float],
    run_id: str,
    dataset_version_id: str,
) -> tuple[Path, Path]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(pickle.dumps(calibrator))
    metadata: dict[str, Any] = {
        "model_id": model_id,
        "created_at": datetime.now(UTC).isoformat(),
        "val_metrics": val_metrics,
        "calibration_method": calibration_method,
        "run_id": run_id,
        "dataset_version_id": dataset_version_id,
    }
    sidecar = destination.with_name(f"{destination.stem}_meta.json")
    sidecar.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return destination, sidecar
