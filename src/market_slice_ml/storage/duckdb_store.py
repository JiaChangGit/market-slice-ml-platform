"""Read-only analytical access to Parquet data."""

from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl


class DuckDBStore:
    def read_parquet(self, paths: str | Path | list[str] | list[Path]) -> pl.DataFrame:
        if isinstance(paths, (str, Path)):
            source: str | list[str] = str(Path(paths))
        else:
            source = [str(Path(path)) for path in paths]
        relation = duckdb.read_parquet(source)
        result = pl.from_arrow(relation.arrow())
        if not isinstance(result, pl.DataFrame):
            raise TypeError("expected a DataFrame from DuckDB Arrow relation")
        return result
