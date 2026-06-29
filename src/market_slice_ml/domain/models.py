"""Frozen data-transfer objects."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from market_slice_ml.domain.enums import OperationStatus


class ProbeResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider_id: str
    symbol: str
    interval: str
    available: bool
    status: OperationStatus = OperationStatus.SUCCESS
    message: str = ""
    suggested_action: str = ""


class PredictionRecord(BaseModel):
    """Strict six-field research prediction contract."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    horizon: Literal["h1", "h2", "h3"]
    direction: Literal["bullish", "neutral", "bearish"]
    expected_return: float
    expected_volatility: float = Field(ge=0.0)
    confidence_score: float = Field(ge=0.0, le=100.0)
