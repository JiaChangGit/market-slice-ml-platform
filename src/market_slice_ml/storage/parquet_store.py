"""Append-only Parquet persistence."""

from __future__ import annotations

from pathlib import Path

import polars as pl


class ParquetStore:
    def write_append_only(self, frame: pl.DataFrame, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"append-only destination already exists: {destination}")
        frame.write_parquet(destination)
        return destination

    def read(self, path: str | Path) -> pl.DataFrame:
        return pl.read_parquet(Path(path))

    def scan(self, pattern: str | Path) -> pl.LazyFrame:
        return pl.scan_parquet(str(pattern))

    def write_replace(self, frame: pl.DataFrame, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(f"{destination.suffix}.tmp")
        frame.write_parquet(temporary)
        temporary.replace(destination)
        return destination
