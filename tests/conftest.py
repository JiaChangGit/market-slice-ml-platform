"""Shared deterministic offline fixtures."""

from __future__ import annotations

import socket

import pytest

from tests.fixtures.synthetic_data import synthetic_bars


@pytest.fixture(autouse=True)
def enforce_offline_tests(monkeypatch):
    """Fail immediately if a test attempts to open a real network socket."""
    monkeypatch.setenv("NO_NETWORK", "1")

    def blocked(*_args, **_kwargs):
        raise AssertionError("Live network access is forbidden in tests (NO_NETWORK=1)")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)


@pytest.fixture
def bars():
    return synthetic_bars("NQ=F", rows=500)


@pytest.fixture
def canonical_bars(bars):
    from market_slice_ml.processing.canonical_builder import build_canonical_bars

    return build_canonical_bars([bars], "NQ=F", futures=True)


@pytest.fixture
def labeled_bars(canonical_bars):
    from market_slice_ml.features.feature_builder import build_features
    from market_slice_ml.labels.label_builder import build_labels

    return build_labels(build_features(canonical_bars))
