import polars as pl

from market_slice_ml.storage.duckdb_store import DuckDBStore


def test_duckdb_store_reads_multiple_parquet_files(tmp_path):
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    pl.DataFrame({"value": [1]}).write_parquet(first)
    pl.DataFrame({"value": [2]}).write_parquet(second)
    result = DuckDBStore().read_parquet([first, second])
    assert sorted(result.get_column("value").to_list()) == [1, 2]
