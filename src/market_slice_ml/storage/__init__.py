"""Local data stores."""

from market_slice_ml.storage.metadata_store import MetadataStore
from market_slice_ml.storage.parquet_store import ParquetStore

__all__ = ["MetadataStore", "ParquetStore"]
