"""Unit tests for the pushed pose stream in ``GstMediaServer``.

The daemon pushes the robot pose to subscribed peers over a dedicated
unreliable/unordered ``pose`` data channel. These tests cover the pure
Python logic - subscriber bookkeeping, the periodic ``_push_pose``
broadcast, and channel cleanup - with the GStreamer / GLib edges mocked,
mirroring the approach in ``test_media_server_watchdog``.

Out of scope (would belong to an integration test): real
``create-data-channel`` negotiation, GLib timer firing, and on-wire SCTP
delivery.
"""

from __future__ import annotations

import logging
from typing import Any, List, cast
from unittest.mock import MagicMock

import pytest

from reachy_mini.media.media_server import GstMediaServer


def _make_server() -> GstMediaServer:
    """Build a minimal ``GstMediaServer`` with only the pose attrs wired up.

    Bypasses ``__init__`` (which boots GStreamer) and initialises just the
    attributes the pose code touches; same approach as
    ``test_media_server_watchdog``.
    """
    server = cast(GstMediaServer, object.__new__(GstMediaServer))
    server._logger = logging.getLogger("test_pose")
    server._pose_channels = {}
    server._pose_subscribers = set()
    server._pose_provider = None
    server._pose_push_source_id = None
    # Stubs for `__del__` -> `close()`, which the destructor calls at GC time.
    server._loop = MagicMock()
    server._bus_sender = MagicMock()
    return server


def test_set_pose_subscription_add_and_remove_is_idempotent() -> None:
    """Adding/removing a peer is a set op - safe to repeat."""
    server = _make_server()

    server.set_pose_subscription("peer-1", True)
    assert server._pose_subscribers == {"peer-1"}
    server.set_pose_subscription("peer-1", True)
    assert server._pose_subscribers == {"peer-1"}

    server.set_pose_subscription("peer-1", False)
    assert server._pose_subscribers == set()
    # Removing an unknown peer must not raise.
    server.set_pose_subscription("peer-1", False)
    assert server._pose_subscribers == set()


def test_push_pose_noop_without_subscribers() -> None:
    """No subscribers: the provider isn't even polled, timer stays alive."""
    server = _make_server()
    provider = MagicMock(return_value='{"state": {}, "seq": 1}')
    server._pose_provider = provider

    assert server._push_pose() is True
    provider.assert_not_called()


def test_push_pose_noop_without_provider() -> None:
    """A subscriber but no provider: nothing to send, timer stays alive."""
    server = _make_server()
    channel = MagicMock()
    server._pose_channels = {"peer-1": channel}
    server._pose_subscribers = {"peer-1"}

    assert server._push_pose() is True
    channel.emit.assert_not_called()


def test_push_pose_skips_tick_when_provider_returns_none() -> None:
    """A ``None`` frame (state not ready) skips the tick without sending."""
    server = _make_server()
    channel = MagicMock()
    server._pose_channels = {"peer-1": channel}
    server._pose_subscribers = {"peer-1"}
    server._pose_provider = MagicMock(return_value=None)

    assert server._push_pose() is True
    channel.emit.assert_not_called()


def test_push_pose_broadcasts_to_subscribed_channels() -> None:
    """Every subscribed peer with an open channel gets the frame."""
    server = _make_server()
    message = '{"state": {}, "seq": 7}'
    server._pose_provider = MagicMock(return_value=message)
    channel_a = MagicMock()
    channel_b = MagicMock()
    server._pose_channels = {"peer-a": channel_a, "peer-b": channel_b}
    server._pose_subscribers = {"peer-a", "peer-b"}

    assert server._push_pose() is True
    channel_a.emit.assert_called_once_with("send-string", message)
    channel_b.emit.assert_called_once_with("send-string", message)


def test_push_pose_skips_subscriber_without_open_channel() -> None:
    """A subscribed peer whose channel isn't open yet is skipped, not crashed."""
    server = _make_server()
    server._pose_provider = MagicMock(return_value='{"state": {}, "seq": 1}')
    server._pose_subscribers = {"peer-no-channel"}
    server._pose_channels = {}

    assert server._push_pose() is True  # must not raise


def test_push_pose_survives_a_failing_channel() -> None:
    """One channel raising on send must not stop the broadcast to others."""
    server = _make_server()
    message = '{"state": {}, "seq": 3}'
    server._pose_provider = MagicMock(return_value=message)
    bad = MagicMock()
    bad.emit.side_effect = RuntimeError("channel closed")
    good = MagicMock()
    server._pose_channels = {"bad": bad, "good": good}
    server._pose_subscribers = {"bad", "good"}

    assert server._push_pose() is True
    good.emit.assert_called_once_with("send-string", message)


def test_pose_channel_close_drops_channel_and_subscriber() -> None:
    """Closing a pose channel frees both the channel ref and the subscription."""
    server = _make_server()
    channel = MagicMock()
    server._pose_channels = {"peer-1": channel}
    server._pose_subscribers = {"peer-1"}

    server._on_pose_channel_close(channel, "peer-1")

    assert "peer-1" not in server._pose_channels
    assert "peer-1" not in server._pose_subscribers


def test_ensure_pose_push_started_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The push timer is armed exactly once, even across repeated calls."""
    added: List[Any] = []
    from reachy_mini.media import media_server as ms

    def fake_timeout_add(interval: int, fn: Any) -> int:
        added.append((interval, fn))
        return 4242

    monkeypatch.setattr(ms.GLib, "timeout_add", fake_timeout_add)

    server = _make_server()
    server._ensure_pose_push_started()
    server._ensure_pose_push_started()

    assert len(added) == 1
    assert added[0][0] == GstMediaServer.POSE_PUSH_INTERVAL_MS
    assert server._pose_push_source_id == 4242

    # Clear the fake source id so `__del__` -> `close()` doesn't try to remove
    # a non-existent GLib source at GC time (noisy unraisable warning).
    server._pose_push_source_id = None
