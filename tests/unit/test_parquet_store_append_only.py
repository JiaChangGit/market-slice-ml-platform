import pytest

from market_slice_ml.storage.parquet_store import ParquetStore


def test_parquet_store_never_overwrites(bars, tmp_path):
    path = tmp_path / "bars.parquet"
    store = ParquetStore()
    store.write_append_only(bars, path)
    with pytest.raises(FileExistsError):
        store.write_append_only(bars, path)
    assert store.read(path).height == bars.height
