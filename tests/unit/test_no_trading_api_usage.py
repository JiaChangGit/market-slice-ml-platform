from pathlib import Path

from scripts.check_no_trading_api import scan


def test_implementation_passes_no_execution_guard():
    assert scan(Path("src")) == []
    assert scan(Path("tests")) == []
