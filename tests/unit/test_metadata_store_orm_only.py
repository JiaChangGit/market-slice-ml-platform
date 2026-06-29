from datetime import UTC, datetime

from market_slice_ml.domain.enums import DatasetType
from market_slice_ml.storage.metadata_store import MetadataStore
from market_slice_ml.versioning.dataset_manifest import DatasetManifest


def test_metadata_store_round_trip_uses_registry_models(tmp_path):
    now = datetime.now(UTC)
    manifest = DatasetManifest(
        dataset_type=DatasetType.CANONICAL_5M,
        created_at_utc=now,
        config_hash="a",
        symbol_universe_hash="b",
        row_count=2,
        timestamp_min_utc=now,
        timestamp_max_utc=now,
    )
    store = MetadataStore(tmp_path / "meta.db")
    store.register_manifest(manifest)
    assert store.get_manifest(manifest.dataset_id) == manifest
