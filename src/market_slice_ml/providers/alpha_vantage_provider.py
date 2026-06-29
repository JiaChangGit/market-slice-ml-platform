"""Credential-gated Alpha Vantage intraday source."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any

import polars as pl
import requests

from market_slice_ml.domain.models import ProbeResult
from market_slice_ml.providers.base import BaseProvider, empty_bars

LOGGER = logging.getLogger(__name__)


class AlphaVantageProvider(BaseProvider):
    @property
    def provider_id(self) -> str:
        return "alpha_vantage"

    @property
    def is_enabled(self) -> bool:
        return bool(os.getenv("ALPHA_VANTAGE_API_KEY"))

    def probe(self, symbol: str, interval: str) -> ProbeResult:
        return ProbeResult(
            provider_id=self.provider_id,
            symbol=symbol,
            interval=interval,
            available=self.is_enabled,
            message="credential configured" if self.is_enabled else "missing credential",
        )

    def fetch_bars(
        self, symbol: str, interval: str, start: datetime, end: datetime
    ) -> pl.DataFrame:
        self._begin_call()
        if not self.is_enabled:
            return empty_bars()
        if interval not in {"1min", "5min", "15min", "30min", "60min", "5m"}:
            return empty_bars()
        provider_interval = "5min" if interval == "5m" else interval
        try:
            response = requests.get(
                "https://www.alphavantage.co/query",
                params={
                    "function": "TIME_SERIES_INTRADAY",
                    "symbol": symbol,
                    "interval": provider_interval,
                    "outputsize": "full",
                    "apikey": os.environ["ALPHA_VANTAGE_API_KEY"],
                },
                timeout=30,
            )
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
            key = next((name for name in payload if name.startswith("Time Series")), None)
            if key is None:
                detail = str(payload.get("Note") or payload.get("Error Message") or "no series")
                self._record_failure(detail, "Check the API key, symbol, and provider rate limit.")
                return empty_bars()
            rows = []
            for timestamp, values in payload[key].items():
                rows.append(
                    {
                        "timestamp_utc": timestamp,
                        "symbol": symbol,
                        "open": values["1. open"],
                        "high": values["2. high"],
                        "low": values["3. low"],
                        "close": values["4. close"],
                        "volume": values["5. volume"],
                        "provider": self.provider_id,
                    }
                )
            frame = pl.DataFrame(rows).with_columns(
                pl.col("timestamp_utc")
                .str.to_datetime(time_zone="America/New_York")
                .dt.convert_time_zone("UTC")
                .cast(pl.Datetime("us", "UTC")),
                *[
                    pl.col(column).cast(pl.Float64)
                    for column in ("open", "high", "low", "close", "volume")
                ],
            )
            return frame.filter(pl.col("timestamp_utc").is_between(start, end))
        except Exception as exc:
            LOGGER.warning("Alpha Vantage fetch failed for %s: %s", symbol, exc)
            self._record_failure(
                str(exc), "Check the API key, network access, and provider rate limits."
            )
            return empty_bars()
