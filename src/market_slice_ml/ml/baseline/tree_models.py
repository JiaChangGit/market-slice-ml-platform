"""Compact LightGBM models for the three research prediction heads."""

from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from numpy.typing import NDArray


class TreeDirectionModel:
    def __init__(self, n_estimators: int = 100, random_state: int = 42) -> None:
        self.model = LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=n_estimators,
            learning_rate=0.05,
            num_leaves=31,
            random_state=random_state,
            verbosity=-1,
        )

    def fit(
        self,
        features: NDArray[np.float32],
        target: NDArray[np.int64],
        sample_weight: NDArray[np.float32] | None = None,
    ) -> TreeDirectionModel:
        self.model.fit(features, target, sample_weight=sample_weight)
        return self

    def predict_proba(self, features: NDArray[np.float32]) -> NDArray[np.float64]:
        return np.asarray(self.model.predict_proba(features), dtype=np.float64)


class TreeReturnModel:
    def __init__(self, n_estimators: int = 100, random_state: int = 42) -> None:
        self.model = LGBMRegressor(
            objective="regression",
            n_estimators=n_estimators,
            learning_rate=0.05,
            num_leaves=31,
            random_state=random_state,
            verbosity=-1,
        )

    def fit(
        self,
        features: NDArray[np.float32],
        target: NDArray[np.float32],
        sample_weight: NDArray[np.float32] | None = None,
    ) -> TreeReturnModel:
        self.model.fit(features, target, sample_weight=sample_weight)
        return self

    def predict(self, features: NDArray[np.float32]) -> NDArray[np.float64]:
        return np.asarray(self.model.predict(features), dtype=np.float64)


class TreeVolatilityModel:
    def __init__(self, n_estimators: int = 100, random_state: int = 42) -> None:
        self.model = LGBMRegressor(
            objective="regression",
            n_estimators=n_estimators,
            learning_rate=0.05,
            num_leaves=31,
            random_state=random_state,
            verbosity=-1,
        )

    def fit(
        self,
        features: NDArray[np.float32],
        target: NDArray[np.float32],
        sample_weight: NDArray[np.float32] | None = None,
    ) -> TreeVolatilityModel:
        transformed = np.log1p(np.maximum(target, 0.0))
        self.model.fit(features, transformed, sample_weight=sample_weight)
        return self

    def predict(self, features: NDArray[np.float32]) -> NDArray[np.float64]:
        predicted = self.model.predict(features)
        return np.asarray(np.maximum(0.0, np.expm1(predicted)), dtype=np.float64)
