"""Domain enumerations."""

from enum import StrEnum


class DatasetType(StrEnum):
    RAW = "raw"
    CANONICAL_5M = "canonical_5m"
    DERIVED_30M = "derived_30m"
    FEATURES = "features"
    LABELS = "labels"
    SLICED_DATASET = "sliced_dataset"
    TRAINING_DATASET = "training_dataset"
    PREDICTION_DATASET = "prediction_dataset"


class Direction(StrEnum):
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    BULLISH = "bullish"


class OperationStatus(StrEnum):
    DISABLED = "disabled"
    NO_DATA = "no_data"
    PARTIAL = "partial"
    FAILED = "failed"
    SUCCESS = "success"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
