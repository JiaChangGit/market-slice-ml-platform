from market_slice_ml.reporting.html_report import write_html_report


def test_report_contains_no_external_urls(tmp_path):
    path = write_html_report(tmp_path / "report.html", [], {"gbm": 1.0})
    text = path.read_text().lower()
    assert "http://" not in text
    assert "https://" not in text
    assert "cdn" not in text
