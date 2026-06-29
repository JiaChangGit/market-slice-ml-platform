"""Combine calibrated model outputs into the strict six-field contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from market_slice_ml.domain.models import PredictionRecord
from market_slice_ml.ml.calibration.confidence_calibrator import confidence_score


@dataclass(frozen=True)
class ModelPrediction:
    direction_probabilities: NDArray[np.float64]
    expected_return: float
    expected_volatility: float


def ensemble_prediction(
    symbol: str,
    horizon: Literal["h1", "h2", "h3"],
    predictions: dict[str, ModelPrediction],
    weights: dict[str, float],
    recent_accuracy: float = 0.5,
    data_quality: float = 1.0,
    provider_coverage: float = 1.0,
) -> PredictionRecord:
    active = {name: value for name, value in predictions.items() if weights.get(name, 0.0) > 0}
    if not active:
        raise ValueError("no available model predictions")
    probabilities = sum(
        (weights[name] * value.direction_probabilities for name, value in active.items()),
        start=np.zeros(3, dtype=np.float64),
    )
    expected_return = sum(weights[name] * value.expected_return for name, value in active.items())
    volatility = sum(weights[name] * value.expected_volatility for name, value in active.items())
    classes: list[Literal["bearish", "neutral", "bullish"]] = [
        "bearish",
        "neutral",
        "bullish",
    ]
    direction_indices = [int(np.argmax(value.direction_probabilities)) for value in active.values()]
    agreement = max(direction_indices.count(index) for index in set(direction_indices)) / len(
        direction_indices
    )
    confidence = confidence_score(
        float(probabilities.max()),
        agreement,
        recent_accuracy,
        1.0,
        data_quality,
        provider_coverage,
    )
    return PredictionRecord(
        symbol=symbol,
        horizon=horizon,
        direction=classes[int(np.argmax(probabilities))],
        expected_return=float(expected_return),
        expected_volatility=max(0.0, float(volatility)),
        confidence_score=confidence,
    )
