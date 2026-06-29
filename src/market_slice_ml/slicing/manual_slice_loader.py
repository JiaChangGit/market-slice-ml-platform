"""Load user-defined YAML train/validation pairs."""

from __future__ import annotations

from pathlib import Path

from market_slice_ml.config.loader import load_yaml
from market_slice_ml.slicing.slice_models import TrainValPair


def load_train_val_pairs(
    path: str | Path = "configs/train_val_pairs.yaml",
) -> list[TrainValPair]:
    configured = load_yaml(path).get("train_val_pairs", [])
    return [TrainValPair.model_validate(item) for item in configured]
