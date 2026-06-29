"""Safe YAML configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping in {source}")
    return value


def flatten_symbol_config(config: dict[str, Any]) -> tuple[list[str], list[str]]:
    def collect(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            result: list[str] = []
            for item in value:
                result.extend(collect(item))
            return result
        if isinstance(value, dict):
            result = []
            for item in value.values():
                result.extend(collect(item))
            return result
        return []

    return collect(config.get("target_symbols", {})), collect(config.get("context_symbols", {}))
