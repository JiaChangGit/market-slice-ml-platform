from market_slice_ml.slicing.dataset_slice_validator import validate_slice
from market_slice_ml.slicing.synchronized_slice_builder import build_synchronized_slice
from tests.fixtures.synthetic_data import synthetic_bars
from tests.fixtures.synthetic_slices import synthetic_pair


def test_dataset_slice_validator_requires_symbol_coverage():
    sliced = build_synchronized_slice({"NQ=F": synthetic_bars()}, synthetic_pair())
    validate_slice(sliced, ["NQ=F"])
