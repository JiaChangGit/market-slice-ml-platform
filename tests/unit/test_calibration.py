import numpy as np

from market_slice_ml.ml.calibration.probability_calibrator import ProbabilityCalibrator
from market_slice_ml.ml.calibration.volatility_calibrator import VolatilityCalibrator


def test_calibrators_fit_and_transform():
    logits = np.asarray([[3.0, 1.0, 0.0], [0.0, 1.0, 3.0]])
    probabilities = ProbabilityCalibrator().fit(logits, np.asarray([0, 2])).transform(logits)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    predicted = np.asarray([0.01, -0.02])
    actual = np.asarray([0.02, -0.01])
    volatility = VolatilityCalibrator().fit(np.abs(predicted), np.abs(actual))
    assert (volatility.transform(np.abs(predicted)) >= 0).all()
