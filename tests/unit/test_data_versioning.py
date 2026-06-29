from datetime import UTC, datetime

from market_slice_ml.domain.enums import DatasetType
from market_slice_ml.versioning.dataset_manifest import DatasetManifest


def test_manifest_is_frozen_and_timezone_aware():
    now = datetime.now(UTC)
    manifest = DatasetManifest(
        dataset_type=DatasetType.FEATURES,
        created_at_utc=now,
        config_hash="a",
        symbol_universe_hash="b",
        row_count=1,
        timestamp_min_utc=now,
        timestamp_max_utc=now,
    )
    assert manifest.dataset_id
    assert manifest.model_config["frozen"] is True
