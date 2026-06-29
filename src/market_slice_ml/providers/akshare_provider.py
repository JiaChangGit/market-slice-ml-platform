"""Experimental AKShare source, disabled unless explicitly enabled."""

from __future__ import annotations

import logging
import os
from datetime import datetime

import polars as pl

from market_slice_ml.domain.models import ProbeResult
from market_slice_ml.providers.base import BaseProvider, empty_bars
from market_slice_ml.providers.yfinance_provider import pandas_bars_to_polars

LOGGER = logging.getLogger(__name__)


class AKShareProvider(BaseProvider):
    @property
    def provider_id(self) -> str:
        return "akshare"

    @property
    def is_enabled(self) -> bool:
        return os.getenv("AKSHARE_ENABLED", "false").lower() == "true"

    def probe(self, symbol: str, interval: str) -> ProbeResult:
        return ProbeResult(
            provider_id=self.provider_id,
            symbol=symbol,
            interval=interval,
            available=self.is_enabled,
            message="experimental" if self.is_enabled else "disabled by default",
        )

    def fetch_bars(
        self, symbol: str, interval: str, start: datetime, end: datetime
    ) -> pl.DataFrame:
        self._begin_call()
        if not self.is_enabled:
            return empty_bars()
        try:
            import akshare as ak
        except ImportError:
            LOGGER.info("akshare not installed")
            self._record_failure(
                "akshare is enabled but the package is not installed.",
                'Install the optional dependency with pip install -e ".[akshare]".',
            )
            return empty_bars()
        try:
            data = ak.stock_us_daily(symbol=symbol, adjust="qfq")
            data = data.set_index("date")
            return pandas_bars_to_polars(data, symbol, self.provider_id).filter(
                pl.col("timestamp_utc").is_between(start, end)
            )
        except Exception as exc:
            LOGGER.warning("AKShare historical fetch failed for %s: %s", symbol, exc)
            self._record_failure(
                str(exc), "Check AKShare symbol support and the provider response format."
            )
            return empty_bars()
