"""Historical provider contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

import polars as pl

from market_slice_ml.domain.enums import OperationStatus
from market_slice_ml.domain.models import ProbeResult

BAR_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "timestamp_utc": pl.Datetime("us", "UTC"),
    "symbol": pl.String,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "provider": pl.String,
}


def empty_bars() -> pl.DataFrame:
    return pl.DataFrame(schema=BAR_SCHEMA)


@dataclass(frozen=True)
class ProviderFetchResult:
    provider_id: str
    symbol: str
    interval: str
    status: OperationStatus
    frame: pl.DataFrame
    message: str
    suggested_action: str = ""


class BaseProvider(ABC):
    _last_error: str | None = None
    _last_suggested_action: str = ""

    def _begin_call(self) -> None:
        self._last_error = None
        self._last_suggested_action = ""

    def _record_failure(self, message: str, suggested_action: str) -> None:
        self._last_error = message
        self._last_suggested_action = suggested_action

    def fetch_with_status(
        self, symbol: str, interval: str, start: datetime, end: datetime
    ) -> ProviderFetchResult:
        if not self.is_enabled:
            return ProviderFetchResult(
                self.provider_id,
                symbol,
                interval,
                OperationStatus.DISABLED,
                empty_bars(),
                "Provider is disabled or not configured.",
                "Enable the provider and verify its dependency or credentials.",
            )
        self._begin_call()
        frame = self.fetch_bars(symbol, interval, start, end)
        if self._last_error:
            return ProviderFetchResult(
                self.provider_id,
                symbol,
                interval,
                OperationStatus.FAILED,
                frame,
                self._last_error,
                self._last_suggested_action,
            )
        if frame.is_empty():
            return ProviderFetchResult(
                self.provider_id,
                symbol,
                interval,
                OperationStatus.NO_DATA,
                frame,
                "The provider returned no bars for the requested interval.",
                "Check interval support, symbol mapping, and the requested date range.",
            )
        return ProviderFetchResult(
            self.provider_id,
            symbol,
            interval,
            OperationStatus.SUCCESS,
            frame,
            f"Fetched {frame.height} bars.",
        )

    @abstractmethod
    def probe(self, symbol: str, interval: str) -> ProbeResult:
        """Report whether the historical source can serve a symbol."""

    @abstractmethod
    def fetch_bars(
        self, symbol: str, interval: str, start: datetime, end: datetime
    ) -> pl.DataFrame:
        """Fetch historical bars, returning an empty frame on source failure."""

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Stable source identifier."""

    @property
    @abstractmethod
    def is_enabled(self) -> bool:
        """Whether calls to this source are allowed."""
