from hypothesis import given
from hypothesis import strategies as st

from market_slice_ml.labels.label_builder import build_labels
from tests.fixtures.synthetic_data import synthetic_bars


@given(st.integers(min_value=1, max_value=20))
def test_label_null_tail_equals_horizon(horizon):
    frame = build_labels(synthetic_bars(rows=60), horizons={"hx": horizon})
    assert frame.tail(horizon).get_column("forward_return_hx").null_count() == horizon
