"""Bounded research-confidence score."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class ConfidenceCalibrator:
    def __init__(self) -> None:
        self.scale = 1.0

    def fit(
        self, raw_confidence: NDArray[np.float64], correct: NDArray[np.bool_]
    ) -> ConfidenceCalibrator:
        average_confidence = float(np.mean(raw_confidence)) if raw_confidence.size else 0.0
        accuracy = float(np.mean(correct)) if correct.size else 0.0
        self.scale = accuracy / max(average_confidence, 1e-6)
        return self

    def transform(self, raw_confidence: float) -> float:
        return min(1.0, max(0.0, raw_confidence * self.scale))


def confidence_score(
    direction_probability: float,
    model_agreement: float,
    recent_accuracy: float,
    slice_recency: float,
    data_quality: float,
    provider_coverage: float,
) -> float:
    weighted = (
        0.30 * direction_probability
        + 0.25 * model_agreement
        + 0.20 * recent_accuracy
        + 0.10 * slice_recency
        + 0.10 * data_quality
        + 0.05 * provider_coverage
    )
    return min(100.0, max(0.0, 100.0 * weighted))
