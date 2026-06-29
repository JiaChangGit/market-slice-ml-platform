from market_slice_ml.processing.canonical_builder import build_canonical_bars


def test_canonical_builder_adds_required_quality_metadata(bars):
    result = build_canonical_bars([bars], "NQ=F", futures=True)
    required = {
        "quality_score",
        "quality_flags",
        "provider",
        "interpolated",
        "suspicious",
        "missing",
    }
    assert required.issubset(result.columns)
    assert result.height == bars.height
    assert result.get_column("quality_score").is_between(0.0, 1.0).all()
    assert result.schema["timestamp_utc"].time_zone == "UTC"


def test_canonical_builder_is_provider_failover_deterministic(bars):
    weaker = bars.with_columns(
        bars.get_column("close").alias("close"),
    ).with_columns(bars.get_column("provider").replace("synthetic", "secondary"))
    result = build_canonical_bars(
        [weaker, bars], "NQ=F", futures=True, priorities={"synthetic": 1, "secondary": 2}
    )
    assert result.get_column("provider").unique().to_list() == ["synthetic"]
