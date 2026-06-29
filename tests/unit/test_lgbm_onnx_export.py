import numpy as np
import onnx
from lightgbm import LGBMRegressor

from market_slice_ml.ml.baseline.tree_trainer import export_lgbm_to_onnx


def test_lightgbm_onnx_export_uses_valid_model(tmp_path):
    features = np.arange(40, dtype=np.float32).reshape(10, 4)
    model = LGBMRegressor(n_estimators=2, verbosity=-1).fit(features, np.arange(10))
    path = export_lgbm_to_onnx(model.booster_, 4, tmp_path / "tree.onnx")
    onnx.checker.check_model(onnx.load(path))
