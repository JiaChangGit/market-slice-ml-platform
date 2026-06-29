"""Stooq daily historical source."""

from __future__ import annotations

import logging
from datetime import datetime

import polars as pl

from market_slice_ml.domain.models import ProbeResult
from market_slice_ml.normalization.symbol_mapper import to_stooq_symbol
from market_slice_ml.providers.base import BaseProvider, empty_bars
from market_slice_ml.providers.yfinance_provider import pandas_bars_to_polars

LOGGER = logging.getLogger(__name__)


class StooqProvider(BaseProvider):
    @property
    def provider_id(self) -> str:
        return "stooq"

    @property
    def is_enabled(self) -> bool:
        return True

    def probe(self, symbol: str, interval: str) -> ProbeResult:
        available = interval in {"1d", "d", "daily"}
        return ProbeResult(
            provider_id=self.provider_id,
            symbol=symbol,
            interval=interval,
            available=available,
            message="daily only" if not available else "ready; live request not performed",
        )

    def fetch_bars(
        self, symbol: str, interval: str, start: datetime, end: datetime
    ) -> pl.DataFrame:
        self._begin_call()
        if interval not in {"1d", "d", "daily"}:
            return empty_bars()
        try:
            from pandas_datareader import data as web

            data = web.DataReader(to_stooq_symbol(symbol), "stooq", start, end).sort_index()
            return pandas_bars_to_polars(data, symbol, self.provider_id)
        except Exception as exc:
            LOGGER.warning("Stooq historical fetch failed for %s: %s", symbol, exc)
            self._record_failure(
                str(exc), "Check network access and use a daily interval with a supported symbol."
            )
            return empty_bars()
