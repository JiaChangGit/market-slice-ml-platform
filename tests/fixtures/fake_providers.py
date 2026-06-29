"""In-memory provider implementation for integration tests."""

from __future__ import annotations

from datetime import datetime

import polars as pl

from market_slice_ml.domain.models import ProbeResult
from market_slice_ml.providers.base import BaseProvider


class FakeProvider(BaseProvider):
    def __init__(self, frame: pl.DataFrame) -> None:
        self.frame = frame

    @property
    def provider_id(self) -> str:
        return "fake"

    @property
    def is_enabled(self) -> bool:
        return True

    def probe(self, symbol: str, interval: str) -> ProbeResult:
        return ProbeResult(
            provider_id=self.provider_id,
            symbol=symbol,
            interval=interval,
            available=True,
        )

    def fetch_bars(
        self, symbol: str, interval: str, start: datetime, end: datetime
    ) -> pl.DataFrame:
        return self.frame.filter(pl.col("timestamp_utc").is_between(start, end))
