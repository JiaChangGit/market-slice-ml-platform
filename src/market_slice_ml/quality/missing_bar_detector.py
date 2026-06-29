"""Detect missing expected timestamps."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

import polars as pl


def missing_timestamps(frame: pl.DataFrame, expected: Iterable[datetime]) -> list[datetime]:
    actual = set(frame.get_column("timestamp_utc").to_list())
    return sorted(timestamp for timestamp in expected if timestamp not in actual)
