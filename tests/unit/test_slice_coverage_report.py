from market_slice_ml.slicing.slice_coverage_report import slice_coverage
from market_slice_ml.slicing.synchronized_slice_builder import build_synchronized_slice
from tests.fixtures.synthetic_data import synthetic_bars
from tests.fixtures.synthetic_slices import synthetic_pair


def test_slice_coverage_reports_every_symbol():
    frames = {symbol: synthetic_bars(symbol) for symbol in ("NQ=F", "AMD", "NVDA")}
    report = slice_coverage(build_synchronized_slice(frames, synthetic_pair()))
    assert report.height == 3
