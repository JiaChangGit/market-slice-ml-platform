"""Short, session-local interpolation for canonical bars."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import polars as pl


def fill_short_gaps(
    frame: pl.DataFrame, max_bars: int = 3, max_missing_bars: int = 78
) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    rows: list[dict[str, Any]] = []
    for symbol_frame in frame.partition_by("symbol", maintain_order=True):
        source = symbol_frame.sort("timestamp_utc").to_dicts()
        for index, current in enumerate(source[:-1]):
            rows.append(current)
            following = source[index + 1]
            delta = following["timestamp_utc"] - current["timestamp_utc"]
            count = int(delta / timedelta(minutes=5)) - 1
            if 0 < count <= max_missing_bars:
                for offset in range(1, count + 1):
                    fraction = offset / (count + 1)
                    interpolated = dict(current)
                    interpolated["timestamp_utc"] = current["timestamp_utc"] + timedelta(
                        minutes=5 * offset
                    )
                    if count <= max_bars:
                        for column in ("open", "high", "low", "close"):
                            interpolated[column] = current[column] + fraction * (
                                following[column] - current[column]
                            )
                        interpolated["volume"] = 0.0
                        interpolated["provider"] = "interpolated"
                        interpolated["interpolated"] = True
                        interpolated["missing"] = False
                    else:
                        for column in ("open", "high", "low", "close", "volume"):
                            interpolated[column] = None
                        interpolated["provider"] = "missing"
                        interpolated["interpolated"] = False
                        interpolated["missing"] = True
                    rows.append(interpolated)
        rows.append(source[-1])
    result = pl.DataFrame(rows, infer_schema_length=None)
    if "interpolated" not in result.columns:
        result = result.with_columns(pl.lit(False).alias("interpolated"))
    if "missing" not in result.columns:
        result = result.with_columns(pl.lit(False).alias("missing"))
    return result.with_columns(
        pl.col("timestamp_utc").cast(pl.Datetime("us", "UTC")),
        pl.col("interpolated").fill_null(False),
        pl.col("missing").fill_null(False),
    ).sort(["symbol", "timestamp_utc"])
