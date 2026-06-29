from market_slice_ml.reporting.html_report import write_html_report


def test_report_generation_is_local(tmp_path):
    weights = {"gbm": 1 / 3, "lstm": 1 / 3, "gnn": 1 / 3}
    path = write_html_report(tmp_path / "report.html", [], weights)
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "環境摘要" in text
    assert "https://" not in text
