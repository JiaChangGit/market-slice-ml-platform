"""Flag duplicate symbol/timestamp observations."""

from __future__ import annotations

import polars as pl


def flag_duplicates(frame: pl.DataFrame) -> pl.DataFrame:
    duplicate = pl.struct(["symbol", "timestamp_utc"]).is_duplicated()
    return frame.with_columns(duplicate.alias("duplicate"))
