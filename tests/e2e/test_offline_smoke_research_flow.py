from market_slice_ml.pipeline import run_synthetic_smoke


def test_offline_research_flow_has_no_execution_fields(tmp_path):
    prediction = run_synthetic_smoke(tmp_path)
    assert len(prediction.model_dump()) == 6
    html = (tmp_path / "reports" / "smoke_report.html").read_text()
    assert "Prediction 摘要" in html
    assert "https://" not in html
