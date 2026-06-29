from datetime import UTC, datetime

import yfinance

from market_slice_ml.providers.yfinance_provider import YFinanceProvider


def test_yfinance_failure_returns_empty(monkeypatch):
    def fail(_symbol):
        raise RuntimeError("offline")

    monkeypatch.setattr(yfinance, "Ticker", fail)
    result = YFinanceProvider().fetch_bars("AMD", "5m", datetime.now(UTC), datetime.now(UTC))
    assert result.is_empty()
