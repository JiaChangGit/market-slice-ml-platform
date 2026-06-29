import polars as pl

from market_slice_ml.quality.quality_score import add_quality_score


def test_quality_score_penalizes_interpolation(canonical_bars):
    base = canonical_bars.head(2).drop("quality_score", "quality_flags")
    result = add_quality_score(base.with_columns(pl.Series("interpolated", [False, True])))
    assert result.item(1, "quality_score") < result.item(0, "quality_score")
