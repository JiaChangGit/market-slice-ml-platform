from market_slice_ml.providers.provider_registry import ProviderRegistry


def test_provider_registry_orders_enabled_sources():
    registry = ProviderRegistry(
        {
            "providers": {
                "yfinance": {"enabled": True, "priority": 2},
                "stooq": {"enabled": True, "priority": 1},
                "cboe": {"enabled": False},
            }
        }
    )
    assert [provider.provider_id for provider in registry.providers] == ["stooq", "yfinance"]
