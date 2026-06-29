"""Disabled realtime scaffold."""

from __future__ import annotations

import logging
from datetime import datetime

import polars as pl

from market_slice_ml.domain.enums import OperationStatus
from market_slice_ml.domain.models import ProbeResult
from market_slice_ml.providers.base import BaseProvider, empty_bars

LOGGER = logging.getLogger(__name__)


class IBKRRealtimeProvider(BaseProvider):
    @property
    def is_enabled(self) -> bool:
        return False

    @property
    def provider_id(self) -> str:
        return "ibkr_realtime"

    def probe(self, symbol: str, interval: str) -> ProbeResult:
        LOGGER.warning("IBKR realtime is disabled in MVP")
        return ProbeResult(
            provider_id=self.provider_id,
            symbol=symbol,
            interval=interval,
            available=False,
            status=OperationStatus.DISABLED,
            message="Realtime Provider 依規格停用。",
            suggested_action="研究資料請改用 Historical Provider。",
        )

    def fetch_bars(
        self, symbol: str, interval: str, start: datetime, end: datetime
    ) -> pl.DataFrame:
        LOGGER.warning("IBKR realtime is disabled in MVP")
        return empty_bars()
