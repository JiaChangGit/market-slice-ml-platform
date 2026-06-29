from market_slice_ml.providers.ibkr_historical_provider import _resolve_ibkr_host


def test_configured_ibkr_host_is_not_rewritten():
    assert _resolve_ibkr_host("192.0.2.10") == "192.0.2.10"
