"""Direction classification metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, f1_score


def direction_metrics(actual: NDArray[np.int64], predicted: NDArray[np.int64]) -> dict[str, float]:
    return {
        "direction_accuracy": float(accuracy_score(actual, predicted)),
        "direction_f1_weighted": float(f1_score(actual, predicted, average="weighted")),
    }
