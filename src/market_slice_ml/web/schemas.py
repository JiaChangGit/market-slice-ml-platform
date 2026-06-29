"""Validated Web API request schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ProviderId = Literal[
    "yfinance",
    "stooq",
    "ibkr_historical",
    "alpha_vantage",
    "cboe",
    "akshare",
]


class FetchJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbols: list[str] = Field(min_length=1, max_length=100)
    provider: ProviderId = "yfinance"
    interval: str = Field(default="5m", pattern=r"^[A-Za-z0-9]+$")
    start_utc: datetime
    end_utc: datetime

    @field_validator("start_utc", "end_utc")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("UTC timestamp 必須包含 timezone")
        return value

    @field_validator("symbols")
    @classmethod
    def normalize_symbols(cls, values: list[str]) -> list[str]:
        normalized = [value.strip().upper() for value in values if value.strip()]
        if not normalized:
            raise ValueError("至少需要一個 Symbol")
        return normalized

    @model_validator(mode="after")
    def validate_range(self) -> FetchJobRequest:
        if self.end_utc <= self.start_utc:
            raise ValueError("結束時間必須晚於開始時間")
        return self


class SymbolsJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbols: list[str] = Field(default_factory=list, max_length=100)

    @field_validator("symbols")
    @classmethod
    def normalize_symbols(cls, values: list[str]) -> list[str]:
        return [value.strip().upper() for value in values if value.strip()]


class TrainJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    horizon: Literal["h1", "h2", "h3", "all"] = "all"
    force: bool = False


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(min_length=1, max_length=24)
    horizon: Literal["h1", "h2", "h3"] = "h1"

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("Symbol 不可為空")
        return normalized


class ApiError(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    message: str
    suggested_action: str
    warnings: tuple[str, ...] = ()
    job_id: str | None = None
