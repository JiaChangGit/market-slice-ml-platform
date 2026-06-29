import polars as pl

from market_slice_ml.quality.volume_anomaly_detector import flag_volume_anomalies


def test_volume_spike_is_flagged(canonical_bars):
    frame = (
        canonical_bars.head(25)
        .with_row_index()
        .with_columns(
            pl.when(pl.col("index") == 24)
            .then(100_000.0)
            .otherwise(pl.col("volume"))
            .alias("volume")
        )
    )
    result = flag_volume_anomalies(frame)
    assert result.item(24, "volume_spike")
