from datetime import date

from market_slice_ml.storage.path_resolver import PathResolver


def test_path_resolver_sanitizes_market_symbols(tmp_path):
    path = PathResolver(tmp_path).raw("test", "NQ=F", "5m", date(2024, 1, 2))
    assert path.is_relative_to(tmp_path)
    assert "NQ_F" in str(path)
