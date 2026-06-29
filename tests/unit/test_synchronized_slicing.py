from market_slice_ml.slicing.synchronized_slice_builder import build_synchronized_slice
from tests.fixtures.synthetic_data import synthetic_bars
from tests.fixtures.synthetic_slices import synthetic_pair


def test_slice_extracts_all_symbols_over_same_window():
    frames = {symbol: synthetic_bars(symbol) for symbol in ("NQ=F", "AMD", "NVDA")}
    result = build_synchronized_slice(frames, synthetic_pair())
    assert set(result.train) == set(frames)
    starts = {frame.item(0, "timestamp_utc") for frame in result.train.values()}
    assert len(starts) == 1
