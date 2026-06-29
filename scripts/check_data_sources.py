#!/usr/bin/env python3
"""Offline-safe provider readiness matrix; network probes require --live."""

from __future__ import annotations

import argparse
import importlib.util
import os
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ProviderStatus:
    provider: str
    enabled: bool
    credential_status: str
    can_probe_offline: bool
    live_test_skipped: bool


def statuses(live: bool = False) -> list[ProviderStatus]:
    def has(name: str) -> bool:
        return importlib.util.find_spec(name) is not None

    ak_enabled = os.getenv("AKSHARE_ENABLED", "false").lower() == "true"
    return [
        ProviderStatus(
            "ibkr_historical",
            has("ib_insync"),
            "library" if has("ib_insync") else "missing optional library",
            True,
            not live,
        ),
        ProviderStatus("yfinance", has("yfinance"), "no credentials", True, not live),
        ProviderStatus("stooq", has("pandas_datareader"), "no credentials", True, not live),
        ProviderStatus(
            "akshare",
            ak_enabled and has("akshare"),
            "enabled by environment" if ak_enabled else "disabled by default",
            True,
            not live,
        ),
        ProviderStatus(
            "alpha_vantage",
            bool(os.getenv("ALPHA_VANTAGE_API_KEY")),
            "configured" if os.getenv("ALPHA_VANTAGE_API_KEY") else "missing key",
            True,
            not live,
        ),
        ProviderStatus("cboe", has("httpx"), "no credentials", True, not live),
        ProviderStatus("schwab", False, "disabled scaffold", True, True),
        ProviderStatus("ibkr_realtime", False, "disabled scaffold", True, True),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="allow explicit live readiness probes")
    args = parser.parse_args()
    columns = ("provider", "enabled", "credential_status", "can_probe_offline", "live_test_skipped")
    print(" | ".join(columns))
    for status in statuses(args.live):
        row = asdict(status)
        print(" | ".join(str(row[column]) for column in columns))
    if args.live:
        print(
            "Live mode was requested; source-specific fetches are available "
            "through the application CLI."
        )


if __name__ == "__main__":
    main()
