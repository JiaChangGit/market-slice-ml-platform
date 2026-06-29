from datetime import UTC, datetime

from market_slice_ml.providers.akshare_provider import AKShareProvider


def test_akshare_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("AKSHARE_ENABLED", raising=False)
    provider = AKShareProvider()
    assert provider.is_enabled is False
    assert provider.fetch_bars("AMD", "1d", datetime.now(UTC), datetime.now(UTC)).is_empty()
