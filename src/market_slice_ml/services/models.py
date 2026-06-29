"""Serializable operation and job snapshots."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from market_slice_ml.domain.enums import JobStatus, OperationStatus


class OperationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: OperationStatus
    message: str
    suggested_action: str = ""
    warnings: tuple[str, ...] = ()
    result: dict[str, Any] = Field(default_factory=dict)


class JobRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    job_id: str
    job_type: str
    status: JobStatus
    created_at_utc: datetime
    started_at_utc: datetime | None = None
    finished_at_utc: datetime | None = None
    message: str = ""
    suggested_action: str = ""
    warnings: tuple[str, ...] = ()
    parameters: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)
    log_path: str = ""


class ProviderReadiness(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider_id: str
    status: OperationStatus
    credentials: str
    message: str
    suggested_action: str = ""


class PipelineStage(BaseModel):
    model_config = ConfigDict(frozen=True)

    stage_id: str
    label_zh_tw: str
    status: str
    detail: str
