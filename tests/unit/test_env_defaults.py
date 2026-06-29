from datetime import date
from pathlib import Path

from market_slice_ml.config.settings import Settings


def test_environment_defaults(monkeypatch):
    monkeypatch.delenv("DATA_ROOT", raising=False)
    settings = Settings(_env_file=None)
    assert settings.data_root == Path("data")
    assert settings.data_start_date == date(2022, 1, 1)
    assert settings.akshare_enabled is False
