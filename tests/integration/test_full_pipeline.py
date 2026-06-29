from market_slice_ml.pipeline import run_synthetic_smoke


def test_full_pipeline_creates_strict_prediction_and_report(tmp_path):
    prediction = run_synthetic_smoke(tmp_path)
    assert len(prediction.model_dump()) == 6
    assert (tmp_path / "reports" / "smoke_report.html").exists()
