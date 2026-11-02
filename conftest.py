"""Repo-wide test setup."""

import pytest


@pytest.fixture(autouse=True)
def _no_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep daemon-booting tests from sending events or touching the user config."""
    monkeypatch.setenv("REACHY_MINI_TELEMETRY", "0")
