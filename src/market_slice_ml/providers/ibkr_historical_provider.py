"""Optional historical-only IBKR source with WSL2 host resolution."""

from __future__ import annotations

import logging
import os
import platform
from datetime import datetime
from pathlib import Path

import polars as pl

from market_slice_ml.domain.models import ProbeResult
from market_slice_ml.providers.base import BaseProvider, empty_bars

LOGGER = logging.getLogger(__name__)


def _resolve_ibkr_host(configured_host: str) -> str:
    if configured_host != "auto":
        return configured_host
    release = platform.uname().release.lower()
    if "microsoft" not in release and "wsl" not in release:
        return "127.0.0.1"
    resolv = Path("/etc/resolv.conf")
    if resolv.exists():
        for line in resolv.read_text(encoding="utf-8").splitlines():
            if line.startswith("nameserver"):
                candidate = line.split()[-1].strip()
                if candidate and not candidate.startswith("127."):
                    return candidate
    return "127.0.0.1"


class IBKRHistoricalProvider(BaseProvider):
    def __init__(self, enabled: bool = False) -> None:
        self._enabled = enabled
        self.host = _resolve_ibkr_host(os.getenv("IBKR_HOST", "auto"))
        self.port = int(os.getenv("IBKR_PORT", "7497"))
        self.client_id = int(os.getenv("IBKR_CLIENT_ID", "1"))

    @property
    def provider_id(self) -> str:
        return "ibkr_historical"

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def probe(self, symbol: str, interval: str) -> ProbeResult:
        if not self.is_enabled:
            return ProbeResult(
                provider_id=self.provider_id,
                symbol=symbol,
                interval=interval,
                available=False,
                message="disabled",
            )
        try:
            import ib_insync  # noqa: F401

            return ProbeResult(
                provider_id=self.provider_id,
                symbol=symbol,
                interval=interval,
                available=True,
                message=f"library ready; endpoint {self.host}:{self.port}",
            )
        except Exception as exc:
            LOGGER.warning("IBKR historical probe unavailable: %s", exc)
            return ProbeResult(
                provider_id=self.provider_id,
                symbol=symbol,
                interval=interval,
                available=False,
                message=str(exc),
            )

    def fetch_bars(
        self, symbol: str, interval: str, start: datetime, end: datetime
    ) -> pl.DataFrame:
        self._begin_call()
        if not self.is_enabled:
            return empty_bars()
        connection = None
        try:
            from ib_insync import IB, ContFuture, Index, Stock, util

            connection = IB()
            connection.connect(
                self.host,
                self.port,
                clientId=self.client_id,
                readonly=True,
                timeout=5,
            )
            if symbol.endswith("=F"):
                contract = ContFuture(symbol.removesuffix("=F"), "CME")
            elif symbol.startswith("^"):
                contract = Index(symbol.removeprefix("^"), "CBOE", "USD")
            else:
                contract = Stock(symbol, "SMART", "USD")
            qualified = connection.qualifyContracts(contract)
            if not qualified:
                return empty_bars()
            duration_days = max(1, (end - start).days + 1)
            bars = connection.reqHistoricalData(
                qualified[0],
                endDateTime=end,
                durationStr=f"{duration_days} D",
                barSizeSetting="5 mins" if interval == "5m" else "1 day",
                whatToShow="TRADES",
                useRTH=not symbol.endswith("=F"),
                formatDate=2,
                keepUpToDate=False,
            )
            data = util.df(bars)
            if data.empty:
                return empty_bars()
            data = data.rename(columns={"date": "timestamp_utc"})
            data["symbol"] = symbol
            data["provider"] = self.provider_id
            return pl.from_pandas(data).select(
                pl.col("timestamp_utc").cast(pl.Datetime("us", "UTC")),
                pl.col("symbol"),
                *[
                    pl.col(column).cast(pl.Float64)
                    for column in ("open", "high", "low", "close", "volume")
                ],
                pl.col("provider"),
            )
        except Exception as exc:
            LOGGER.warning("IBKR historical fetch failed for %s: %s", symbol, exc)
            self._record_failure(
                str(exc),
                "Check Gateway/TWS API access, WSL2 host detection, port, and trusted IPs.",
            )
            return empty_bars()
        finally:
            if connection is not None and connection.isConnected():
                connection.disconnect()
