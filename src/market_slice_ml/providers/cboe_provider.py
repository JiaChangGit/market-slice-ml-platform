"""Cboe context source with unavailable-data fallback."""

from __future__ import annotations

import logging
from datetime import datetime

import polars as pl

from market_slice_ml.domain.models import ProbeResult
from market_slice_ml.providers.base import BaseProvider, empty_bars
from market_slice_ml.providers.yfinance_provider import YFinanceProvider

LOGGER = logging.getLogger(__name__)


class CboeProvider(BaseProvider):
    SUPPORTED = {"^VIX9D", "^VIX", "^VIX3M", "^VIX6M"}

    @property
    def provider_id(self) -> str:
        return "cboe"

    @property
    def is_enabled(self) -> bool:
        return True

    def probe(self, symbol: str, interval: str) -> ProbeResult:
        return ProbeResult(
            provider_id=self.provider_id,
            symbol=symbol,
            interval=interval,
            available=symbol in self.SUPPORTED,
            message="volatility context only",
        )

    def fetch_bars(
        self, symbol: str, interval: str, start: datetime, end: datetime
    ) -> pl.DataFrame:
        self._begin_call()
        if symbol not in self.SUPPORTED:
            return empty_bars()
        try:
            result = YFinanceProvider().fetch_with_status(symbol, interval, start, end)
            if result.status.value == "failed":
                self._record_failure(result.message, result.suggested_action)
                return empty_bars()
            return result.frame.with_columns(pl.lit(self.provider_id).alias("provider"))
        except Exception as exc:
            LOGGER.warning("Cboe context fetch failed for %s: %s", symbol, exc)
            self._record_failure(str(exc), "Check network access and volatility symbol support.")
            return empty_bars()
