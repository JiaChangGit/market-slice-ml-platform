"""Priority-ordered enabled provider registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from market_slice_ml.config.loader import load_yaml
from market_slice_ml.providers.akshare_provider import AKShareProvider
from market_slice_ml.providers.alpha_vantage_provider import AlphaVantageProvider
from market_slice_ml.providers.base import BaseProvider
from market_slice_ml.providers.cboe_provider import CboeProvider
from market_slice_ml.providers.ibkr_historical_provider import IBKRHistoricalProvider
from market_slice_ml.providers.ibkr_realtime_provider import IBKRRealtimeProvider
from market_slice_ml.providers.schwab_provider import SchwabProvider
from market_slice_ml.providers.stooq_provider import StooqProvider
from market_slice_ml.providers.yfinance_provider import YFinanceProvider


class ProviderRegistry:
    def __init__(self, config: str | Path | dict[str, Any] = "configs/providers.yaml") -> None:
        values = load_yaml(config) if isinstance(config, (str, Path)) else config
        provider_config = values.get("providers", {})
        self.all_providers: dict[str, BaseProvider] = {
            "yfinance": YFinanceProvider(),
            "stooq": StooqProvider(),
            "ibkr_historical": IBKRHistoricalProvider(
                enabled=bool(provider_config.get("ibkr_historical", {}).get("enabled", False))
            ),
            "alpha_vantage": AlphaVantageProvider(),
            "cboe": CboeProvider(),
            "akshare": AKShareProvider(),
            "ibkr_realtime": IBKRRealtimeProvider(),
            "schwab": SchwabProvider(),
        }
        ranked: list[tuple[int, BaseProvider]] = []
        for name, provider in self.all_providers.items():
            settings = provider_config.get(name, {})
            if bool(settings.get("enabled", provider.is_enabled)) and provider.is_enabled:
                ranked.append((int(settings.get("priority", 100)), provider))
        self.providers = [provider for _, provider in sorted(ranked, key=lambda item: item[0])]

    def get_providers_for_symbol(self, symbol: str) -> list[BaseProvider]:
        return [provider for provider in self.providers if provider.probe(symbol, "5m").available]

    def get(self, provider_id: str) -> BaseProvider:
        try:
            return self.all_providers[provider_id]
        except KeyError as exc:
            raise ValueError(f"Unknown provider: {provider_id}") from exc
