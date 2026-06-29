"""All-symbol tabular model matrix."""

from __future__ import annotations

import polars as pl


def build_tabular_dataset(
    frames: dict[str, pl.DataFrame], symbol_to_index: dict[str, int] | None = None
) -> pl.DataFrame:
    mapping = symbol_to_index or {symbol: index for index, symbol in enumerate(sorted(frames))}
    rows: list[pl.DataFrame] = []
    for symbol, frame in frames.items():
        if frame.is_empty():
            continue
        quality = pl.col("quality_score") if "quality_score" in frame.columns else pl.lit(1.0)
        training = (
            pl.col("final_training_weight")
            if "final_training_weight" in frame.columns
            else pl.lit(1.0)
        )
        rows.append(
            frame.with_columns(
                pl.lit(mapping[symbol], dtype=pl.Int64).alias("symbol_idx"),
                (quality * training).alias("sample_weight"),
            )
        )
    return pl.concat(rows, how="diagonal_relaxed") if rows else pl.DataFrame()
