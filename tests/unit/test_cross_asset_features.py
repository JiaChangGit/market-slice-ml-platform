from market_slice_ml.features.cross_asset_features import add_cross_asset_features


def test_cross_asset_weighted_return(canonical_bars):
    result = add_cross_asset_features(canonical_bars, {"AMD": canonical_bars}, {"AMD": 1.0})
    assert "xasset_weighted_return_1" in result.columns
    assert result.get_column("xasset_weighted_return_1").drop_nulls().len() > 0
