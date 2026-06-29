"""Forward-volatility error metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def volatility_metrics(
    actual: NDArray[np.float64], predicted: NDArray[np.float64]
) -> dict[str, float]:
    return {"volatility_mae": float(np.mean(np.abs(actual - predicted)))}
