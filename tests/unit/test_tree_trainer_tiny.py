import numpy as np

from market_slice_ml.ml.baseline.tree_trainer import train_tree_models


def test_tree_trainer_fits_all_heads_on_tiny_data():
    rng = np.random.default_rng(4)
    features = rng.normal(size=(30, 4)).astype(np.float32)
    direction = np.tile(np.arange(3), 10).astype(np.int64)
    returns = rng.normal(scale=0.01, size=30).astype(np.float32)
    volatility = rng.uniform(0.1, 0.3, size=30).astype(np.float32)
    bundle = train_tree_models(features, direction, returns, volatility, n_estimators=2)
    assert bundle.direction.predict_proba(features[:2]).shape == (2, 3)
    assert bundle.forward_return.predict(features[:2]).shape == (2,)
    assert (bundle.forward_volatility.predict(features[:2]) >= 0).all()
