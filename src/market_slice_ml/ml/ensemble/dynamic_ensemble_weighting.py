"""Availability-aware normalized ensemble weights."""

from __future__ import annotations

MODEL_FAMILIES = ("gbm", "lstm", "gnn")


def normalize_weights(
    scores: dict[str, float], available: dict[str, bool] | None = None
) -> dict[str, float]:
    readiness = available or {name: True for name in MODEL_FAMILIES}
    positive = {
        name: max(0.0, float(scores.get(name, 0.0))) if readiness.get(name, False) else 0.0
        for name in MODEL_FAMILIES
    }
    total = sum(positive.values())
    if total == 0.0:
        ready = [name for name in MODEL_FAMILIES if readiness.get(name, False)]
        if not ready:
            raise ValueError("at least one model family must be available")
        positive = {name: (1.0 if name in ready else 0.0) for name in MODEL_FAMILIES}
        total = float(len(ready))
    weights = {name: positive[name] / total for name in MODEL_FAMILIES}
    correction_key = max(weights, key=weights.__getitem__)
    for _ in range(4):
        residual = 1.0 - sum(weights.values())
        if residual == 0.0:
            break
        weights[correction_key] += residual
    assert sum(weights.values()) == 1.0
    return weights


class DynamicEnsembleWeighting:
    def __init__(self) -> None:
        self.weights: dict[str, dict[str, float]] = {}

    def update(
        self,
        horizon: str,
        validation_scores: dict[str, float],
        available: dict[str, bool] | None = None,
    ) -> dict[str, float]:
        weights = normalize_weights(validation_scores, available)
        self.weights[horizon] = weights
        assert sum(weights.values()) == 1.0
        return weights

    def get(self, horizon: str) -> dict[str, float]:
        return self.weights.get(horizon, normalize_weights({name: 1.0 for name in MODEL_FAMILIES}))
