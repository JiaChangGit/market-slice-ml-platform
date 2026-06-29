from market_slice_ml.ml.calibration.confidence_calibrator import confidence_score


def test_confidence_score_is_bounded():
    assert confidence_score(1, 1, 1, 1, 1, 1) == 100.0
    assert confidence_score(0, 0, 0, 0, 0, 0) == 0.0
