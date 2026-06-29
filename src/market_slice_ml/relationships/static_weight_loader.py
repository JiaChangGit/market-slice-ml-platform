"""Load frozen relationship definitions from YAML."""

from __future__ import annotations

from pathlib import Path

from market_slice_ml.config.loader import load_yaml
from market_slice_ml.config.schema import RelationshipConfig


def load_relationships(path: str | Path = "configs/relationships.yaml") -> list[RelationshipConfig]:
    configured = load_yaml(path).get("relationships", [])
    return [RelationshipConfig.model_validate(item) for item in configured]
