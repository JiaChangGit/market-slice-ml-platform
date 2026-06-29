from datetime import UTC, datetime, timedelta

from market_slice_ml.domain.enums import OperationStatus
from market_slice_ml.providers.ibkr_realtime_provider import IBKRRealtimeProvider
from market_slice_ml.providers.schwab_provider import SchwabProvider


def test_disabled_scaffolds_return_disabled_status():
    now = datetime.now(UTC)
    realtime = IBKRRealtimeProvider()
    schwab = SchwabProvider()
    assert realtime.probe("NQ=F", "5m").status is OperationStatus.DISABLED
    assert schwab.probe("AMD", "5m").status is OperationStatus.DISABLED
    assert realtime.fetch_bars("NQ=F", "5m", now - timedelta(days=1), now).is_empty()
    assert schwab.fetch_bars("AMD", "5m", now - timedelta(days=1), now).is_empty()
