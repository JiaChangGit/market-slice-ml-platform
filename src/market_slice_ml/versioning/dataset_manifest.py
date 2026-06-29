"""Immutable dataset manifest schema."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from market_slice_ml.domain.enums import DatasetType


class DatasetManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset_id: str = Field(default_factory=lambda: str(uuid4()))
    dataset_type: DatasetType
    version: str = "1.0.0"
    created_at_utc: datetime
    config_hash: str
    symbol_universe_hash: str
    row_count: int = Field(ge=0)
    timestamp_min_utc: datetime | None = None
    timestamp_max_utc: datetime | None = None
    quality_summary: dict[str, Any] = Field(default_factory=dict)
    parquet_paths: list[str] = Field(default_factory=list)

    @field_validator("created_at_utc", "timestamp_min_utc", "timestamp_max_utc")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("manifest timestamps must be timezone-aware")
        return value
