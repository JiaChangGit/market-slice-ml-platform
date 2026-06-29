"""yfinance historical source."""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
import polars as pl

from market_slice_ml.domain.models import ProbeResult
from market_slice_ml.providers.base import BaseProvider, empty_bars

LOGGER = logging.getLogger(__name__)


def pandas_bars_to_polars(data: pd.DataFrame, symbol: str, provider: str) -> pl.DataFrame:
    if data.empty:
        return empty_bars()
    frame = data.reset_index()
    frame.columns = [str(column).lower().replace(" ", "_") for column in frame.columns]
    time_column = "datetime" if "datetime" in frame.columns else "date"
    frame[time_column] = pd.to_datetime(frame[time_column], utc=True)
    result = pl.from_pandas(frame)
    return result.select(
        pl.col(time_column).cast(pl.Datetime("us", "UTC")).alias("timestamp_utc"),
        pl.lit(symbol).alias("symbol"),
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Float64),
        pl.lit(provider).alias("provider"),
    )


class YFinanceProvider(BaseProvider):
    @property
    def provider_id(self) -> str:
        return "yfinance"

    @property
    def is_enabled(self) -> bool:
        return True

    def probe(self, symbol: str, interval: str) -> ProbeResult:
        return ProbeResult(
            provider_id=self.provider_id,
            symbol=symbol,
            interval=interval,
            available=True,
            message="ready; live request not performed",
        )

    def fetch_bars(
        self, symbol: str, interval: str, start: datetime, end: datetime
    ) -> pl.DataFrame:
        self._begin_call()
        try:
            import yfinance as yf

            data = yf.Ticker(symbol).history(start=start, end=end, interval=interval)
            return pandas_bars_to_polars(data, symbol, self.provider_id)
        except Exception as exc:
            LOGGER.warning("yfinance historical fetch failed for %s: %s", symbol, exc)
            self._record_failure(
                str(exc),
                "Check network access, interval retention, symbol spelling, and rate limits.",
            )
            return empty_bars()
