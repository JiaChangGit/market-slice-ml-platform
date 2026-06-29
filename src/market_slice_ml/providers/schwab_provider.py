"""Disabled Schwab scaffold."""

from __future__ import annotations

import logging
from datetime import datetime

import polars as pl

from market_slice_ml.domain.enums import OperationStatus
from market_slice_ml.domain.models import ProbeResult
from market_slice_ml.providers.base import BaseProvider, empty_bars

LOGGER = logging.getLogger(__name__)


class SchwabProvider(BaseProvider):
    @property
    def is_enabled(self) -> bool:
        return False

    @property
    def provider_id(self) -> str:
        return "schwab"

    def probe(self, symbol: str, interval: str) -> ProbeResult:
        LOGGER.warning("Schwab provider is disabled by default")
        return ProbeResult(
            provider_id=self.provider_id,
            symbol=symbol,
            interval=interval,
            available=False,
            status=OperationStatus.DISABLED,
            message="Schwab Provider 依規格停用。",
            suggested_action="請選擇已啟用的 Historical Provider。",
        )

    def fetch_bars(
        self, symbol: str, interval: str, start: datetime, end: datetime
    ) -> pl.DataFrame:
        LOGGER.warning("Schwab provider is disabled by default")
        return empty_bars()
