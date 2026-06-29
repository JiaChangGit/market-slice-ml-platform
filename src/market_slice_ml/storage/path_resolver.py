"""Centralized, traversal-safe data paths."""

from __future__ import annotations

from datetime import date
from pathlib import Path


class PathResolver:
    def __init__(self, data_root: str | Path) -> None:
        self.data_root = Path(data_root).resolve()

    @staticmethod
    def safe_component(value: str) -> str:
        return value.replace("^", "IDX_").replace("=", "_").replace("/", "_").replace("..", "_")

    def raw(
        self,
        provider: str,
        symbol: str,
        interval: str,
        day: date,
        batch_id: str = "",
    ) -> Path:
        safe = self.safe_component(symbol)
        suffix = f"_{batch_id}" if batch_id else ""
        filename = f"{safe}_{interval}_{day}{suffix}.parquet"
        return self.data_root / "raw" / provider / safe / f"{day:%Y/%m}" / filename

    def canonical(self, symbol: str, day: date) -> Path:
        safe = self.safe_component(symbol)
        filename = f"{safe}_5m_canonical_{day}.parquet"
        return self.data_root / "canonical" / safe / str(day.year) / filename

    def derived(self, symbol: str, day: date) -> Path:
        safe = self.safe_component(symbol)
        return self.data_root / "derived" / safe / str(day.year) / f"{safe}_30m_{day}.parquet"

    def features(self, symbol: str, day: date) -> Path:
        safe = self.safe_component(symbol)
        return self.data_root / "features" / safe / str(day.year) / f"{safe}_features_{day}.parquet"

    def labels(self, symbol: str, day: date) -> Path:
        safe = self.safe_component(symbol)
        return self.data_root / "labels" / safe / str(day.year) / f"{safe}_labels_{day}.parquet"

    def slice_root(self, pair_id: str) -> Path:
        return self.data_root / "slices" / self.safe_component(pair_id)

    def model_run(self, run_id: str) -> Path:
        return self.data_root / "models" / "runs" / self.safe_component(run_id)

    def report(self, name: str = "latest.html") -> Path:
        return self.data_root / "reports" / self.safe_component(name)
