import polars as pl

from market_slice_ml.quality.cross_provider_comparator import disagreement_flags


def test_cross_provider_disagreement_threshold(bars):
    first = bars.head(1)
    second = first.with_columns(pl.col("close") * 1.02, pl.lit("other").alias("provider"))
    result = disagreement_flags(pl.concat([first, second]))
    assert result.get_column("suspicious").all()
