from market_slice_ml.domain.models import PredictionRecord
from market_slice_ml.reporting.html_report import write_html_report


def test_html_report_is_single_self_contained_file(tmp_path):
    prediction = PredictionRecord(
        symbol="AMD",
        horizon="h1",
        direction="neutral",
        expected_return=0.0,
        expected_volatility=0.2,
        confidence_score=50,
    )
    path = write_html_report(tmp_path / "report.html", [prediction], {"gbm": 1.0})
    text = path.read_text()
    assert "<style>" in text
    assert 'type="application/json"' in text
    assert "Prediction 摘要" in text


def test_html_report_escapes_symbol_text(tmp_path):
    prediction = PredictionRecord(
        symbol='<img src=x onerror="alert(1)">',
        horizon="h1",
        direction="neutral",
        expected_return=0.0,
        expected_volatility=0.1,
        confidence_score=50,
    )
    text = write_html_report(tmp_path / "safe.html", [prediction], {}).read_text(encoding="utf-8")
    assert "<img src=x" not in text
    assert "&lt;img src=x" in text
