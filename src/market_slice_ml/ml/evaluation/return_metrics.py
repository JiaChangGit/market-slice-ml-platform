"""Forward-return error and information coefficient."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def return_metrics(actual: NDArray[np.float64], predicted: NDArray[np.float64]) -> dict[str, float]:
    correlation = float(np.corrcoef(actual, predicted)[0, 1]) if len(actual) > 1 else 0.0
    return {"return_mae": float(np.mean(np.abs(actual - predicted))), "return_ic": correlation}
