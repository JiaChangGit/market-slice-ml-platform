"""Deterministic content/config fingerprints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def dataset_fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_fingerprint(paths: list[str | Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(item) for item in paths):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
