from market_slice_ml.providers.provider_failover import select_best_bars


def test_provider_failover_respects_priority(bars):
    secondary = bars.with_columns(bars.get_column("provider").replace("synthetic", "secondary"))
    result = select_best_bars([secondary, bars], {"synthetic": 1, "secondary": 2})
    assert result.height == bars.height
    assert result.get_column("provider").unique().to_list() == ["synthetic"]
