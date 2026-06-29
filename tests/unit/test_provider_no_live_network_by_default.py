from market_slice_ml.providers.provider_registry import ProviderRegistry


def test_default_registry_construction_does_not_fetch_data():
    registry = ProviderRegistry()
    assert all(provider.is_enabled for provider in registry.providers)
