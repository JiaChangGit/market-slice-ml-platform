"""Log-bias volatility calibration."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class VolatilityCalibrator:
    def __init__(self) -> None:
        self.log_bias = 0.0

    def fit(
        self, predicted: NDArray[np.float64], actual: NDArray[np.float64]
    ) -> VolatilityCalibrator:
        safe_predicted = np.maximum(predicted, 1e-12)
        safe_actual = np.maximum(actual, 1e-12)
        self.log_bias = float(np.mean(np.log(safe_predicted) - np.log(safe_actual)))
        return self

    def transform(self, predicted: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(np.maximum(0.0, predicted * np.exp(-self.log_bias)), dtype=np.float64)
