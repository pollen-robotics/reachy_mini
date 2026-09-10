"""Fixtures for the over-the-air BLE test suite.

This package is NOT collected by a default ``pytest`` run (it sits outside
``testpaths``). Run it explicitly, from a Linux host with BlueZ and a BLE
adapter in radio range of a powered robot:

    pytest --pyargs reachy_mini_ble_test -v    (installed package)
    pytest src/reachy_mini_ble_test -v         (from a checkout)

Configuration (environment variables):
    REACHY_BLE_NAME              advertised name to match (default "ReachyMini")
    REACHY_BLE_ADDRESS           pin discovery to one MAC address (optional)
    REACHY_BLE_SCAN_TIMEOUT      BLE scan duration in seconds (default 15)
    REACHY_BLE_RESPONSE_TIMEOUT  max wait for a command's notification (default 25)
"""

import os
from collections.abc import Iterator

import pytest

from .ble_link import DEFAULT_DEVICE_NAME, ReachyBleLink


def pytest_configure(config: pytest.Config) -> None:
    """Register the ble marker (needed when running outside the repo)."""
    config.addinivalue_line(
        "markers", "ble: mark test as exercising the BLE service over the air"
    )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Tag every test in this package with the ``ble`` marker."""
    for item in items:
        item.add_marker(pytest.mark.ble)


@pytest.fixture(scope="session")
def link() -> Iterator[ReachyBleLink]:
    """A connected BLE link to the robot, shared by the whole session.

    Fails (rather than skips) when no robot is found: this suite only runs
    when explicitly targeted, so an unreachable robot is a real failure.
    """
    lk = ReachyBleLink(
        name=os.environ.get("REACHY_BLE_NAME", DEFAULT_DEVICE_NAME),
        address=os.environ.get("REACHY_BLE_ADDRESS") or None,
        scan_timeout=float(os.environ.get("REACHY_BLE_SCAN_TIMEOUT", "15")),
        response_timeout=float(os.environ.get("REACHY_BLE_RESPONSE_TIMEOUT", "25")),
    )
    try:
        if lk.discover() is None:
            pytest.fail(
                f"No Reachy Mini found over BLE (name {lk.name!r}, "
                f"address {lk.address or 'any'}, scanned {lk.scan_timeout}s). "
                "Is the robot powered, in range, and not already connected "
                "to another BLE central?"
            )
        lk.connect()
        yield lk
    finally:
        lk.stop()
