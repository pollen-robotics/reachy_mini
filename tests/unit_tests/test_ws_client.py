"""Tests for WSClient's loopback-aware WebSocket compression setting.

permessage-deflate buys nothing when the SDK client and the daemon run on
the same host (no network bandwidth to save) but still costs CPU at the
daemon's 50 Hz state-stream rate. WSClient disables compression when
connecting to a loopback host and leaves the library default (deflate)
untouched for any other host.
"""

from unittest.mock import MagicMock, patch

import pytest

from reachy_mini.io.ws_client import WSClient, _is_loopback_host


@pytest.mark.parametrize(
    "host",
    [
        "localhost",
        "127.0.0.1",
        "127.0.0.2",
        "::1",
        "[::1]",
        "LOCALHOST",
        "Localhost",
        " 127.0.0.1 ",
    ],
)
def test_is_loopback_host_true_for_loopback_addresses(host: str) -> None:
    """Loopback hostnames/IPs are recognized regardless of case/whitespace."""
    assert _is_loopback_host(host) is True


@pytest.mark.parametrize(
    "host",
    ["192.168.1.42", "reachy.local", "example.com", "0.0.0.0", "[not-an-ip]", ""],
)
def test_is_loopback_host_false_for_non_loopback_addresses(host: str) -> None:
    """Non-loopback hosts (including the all-interfaces address) are not matched."""
    assert _is_loopback_host(host) is False


def _make_fake_ws() -> MagicMock:
    """Build a ws_sync.ClientConnection stand-in whose message iterator ends immediately."""
    fake_ws = MagicMock()
    fake_ws.__iter__.return_value = iter([])
    return fake_ws


@patch("reachy_mini.io.ws_client.ws_sync.connect")
def test_connect_disables_compression_for_localhost(mock_connect: MagicMock) -> None:
    """Connecting to localhost passes compression=None to the sync client."""
    mock_connect.return_value = _make_fake_ws()

    client = WSClient(host="localhost", port=1234)
    client.disconnect()

    mock_connect.assert_called_once_with("ws://localhost:1234/ws/sdk", compression=None)


@patch("reachy_mini.io.ws_client.ws_sync.connect")
def test_connect_disables_compression_for_127_0_0_1(mock_connect: MagicMock) -> None:
    """Connecting to 127.0.0.1 also passes compression=None."""
    mock_connect.return_value = _make_fake_ws()

    client = WSClient(host="127.0.0.1", port=1234)
    client.disconnect()

    mock_connect.assert_called_once_with("ws://127.0.0.1:1234/ws/sdk", compression=None)


@patch("reachy_mini.io.ws_client.ws_sync.connect")
def test_connect_keeps_default_compression_for_remote_host(
    mock_connect: MagicMock,
) -> None:
    """Connecting to a non-loopback host leaves the library default (deflate)."""
    mock_connect.return_value = _make_fake_ws()

    client = WSClient(host="192.168.1.50", port=1234)
    client.disconnect()

    mock_connect.assert_called_once_with(
        "ws://192.168.1.50:1234/ws/sdk", compression="deflate"
    )
