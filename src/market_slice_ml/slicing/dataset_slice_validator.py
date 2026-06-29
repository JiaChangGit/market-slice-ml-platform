"""Validate synchronized symbol coverage and time boundaries."""

from __future__ import annotations

from market_slice_ml.domain.exceptions import DataLeakError
from market_slice_ml.slicing.synchronized_slice_builder import SynchronizedSlice


def validate_slice(sliced: SynchronizedSlice, required_symbols: list[str]) -> None:
    if sliced.pair.val_start_utc < sliced.pair.train_end_utc:
        raise DataLeakError("validation overlaps training")
    missing = set(required_symbols) - set(sliced.train)
    if missing:
        raise ValueError(f"slice missing symbols: {sorted(missing)}")
    if any(sliced.train[symbol].is_empty() for symbol in required_symbols):
        raise ValueError("one or more training symbol frames are empty")
