"""Immutable synchronized slice definitions."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from market_slice_ml.domain.exceptions import DataLeakError


class TrainValPair(BaseModel):
    model_config = ConfigDict(frozen=True)

    pair_id: str
    description: str = ""
    anchor_symbol: str
    train_start_utc: datetime
    train_end_utc: datetime
    val_start_utc: datetime
    val_end_utc: datetime
    target_symbols: tuple[str, ...] = ()

    @field_validator("train_start_utc", "train_end_utc", "val_start_utc", "val_end_utc")
    @classmethod
    def require_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("slice timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_order(self) -> TrainValPair:
        if self.train_end_utc <= self.train_start_utc:
            raise ValueError("training interval must be positive")
        if self.val_end_utc <= self.val_start_utc:
            raise ValueError("validation interval must be positive")
        if self.val_start_utc < self.train_end_utc:
            raise DataLeakError("validation must start at or after training ends")
        return self


class SliceManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    slice_id: str
    pair_id: str
    symbols: tuple[str, ...]
    train_rows: int
    val_rows: int
    dataset_fingerprint: str
