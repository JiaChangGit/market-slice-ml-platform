from market_slice_ml.quality.quality_report import build_quality_report


def test_quality_report_has_coverage(canonical_bars):
    report = build_quality_report(canonical_bars)
    assert report.item(0, "bar_count") == canonical_bars.height
    assert 0.0 <= report.item(0, "coverage_ratio") <= 1.0
