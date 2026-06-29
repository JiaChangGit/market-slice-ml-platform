"""Frozen configuration schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RelationshipConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: str
    target: str
    static_weight: float = Field(ge=0.0)
    relationship_type: str
