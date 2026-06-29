"""Tiny-capable training and onnxmltools export for three LightGBM heads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from market_slice_ml.ml.baseline.tree_models import (
    TreeDirectionModel,
    TreeReturnModel,
    TreeVolatilityModel,
)


@dataclass(frozen=True)
class TreeModelBundle:
    direction: TreeDirectionModel
    forward_return: TreeReturnModel
    forward_volatility: TreeVolatilityModel


def train_tree_models(
    features: NDArray[np.float32],
    direction: NDArray[np.int64],
    forward_return: NDArray[np.float32],
    forward_volatility: NDArray[np.float32],
    sample_weight: NDArray[np.float32] | None = None,
    n_estimators: int = 50,
) -> TreeModelBundle:
    return TreeModelBundle(
        TreeDirectionModel(n_estimators).fit(features, direction, sample_weight),
        TreeReturnModel(n_estimators).fit(features, forward_return, sample_weight),
        TreeVolatilityModel(n_estimators).fit(features, forward_volatility, sample_weight),
    )


def export_lgbm_to_onnx(model: object, n_features: int, out_path: str | Path) -> Path:
    import onnx
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType

    initial_types = [("float_input", FloatTensorType([None, n_features]))]
    try:
        converted = convert_lightgbm(model, initial_types=initial_types, target_opset=17)
    except RuntimeError:
        compatible = convert_lightgbm(model, initial_types=initial_types, target_opset=15)
        converted = onnx.version_converter.convert_version(compatible, 17)
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(converted.SerializeToString())
    return destination
