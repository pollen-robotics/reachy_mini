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
from threading import Lock
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
    server._peer_webrtcbins = {}
    server._pose_subscribers = set()
    server._pose_provider = None
    server._pose_push_source_id = None
    server._pose_lock = Lock()
    # Stubs for `__del__` -> `close()`, which the destructor calls at GC time.
    server._loop = MagicMock()
    server._bus_sender = MagicMock()
    return server


def _stub_timeout_add(
    monkeypatch: pytest.MonkeyPatch, source_id: int = 4242
) -> List[Any]:
    """Record ``GLib.timeout_add`` calls instead of arming a real timer."""
    added: List[Any] = []
    from reachy_mini.media import media_server as ms

    def fake_timeout_add(interval: int, fn: Any) -> int:
        added.append((interval, fn))
        return source_id

    monkeypatch.setattr(ms.GLib, "timeout_add", fake_timeout_add)
    return added


def _stub_idle_add(monkeypatch: pytest.MonkeyPatch, run: bool = False) -> List[Any]:
    """Record ``GLib.idle_add`` calls, optionally running them inline.

    ``set_pose_subscription`` defers pose-channel creation to the GLib main
    loop, which no test runs, so the scheduled callback has to be captured
    (and invoked by hand when the test cares about the result).
    """
    scheduled: List[Any] = []
    from reachy_mini.media import media_server as ms

    def fake_idle_add(fn: Any, *args: Any) -> int:
        scheduled.append((fn, args))
        if run:
            fn(*args)
        return 1

    monkeypatch.setattr(ms.GLib, "idle_add", fake_idle_add)
    return scheduled


def test_set_pose_subscription_add_and_remove_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adding/removing a peer is a set op - safe to repeat."""
    _stub_idle_add(monkeypatch)
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


def test_push_pose_disarms_without_subscribers() -> None:
    """No subscribers: the provider isn't polled and the timer is dropped."""
    server = _make_server()
    provider = MagicMock(return_value='{"state": {}, "seq": 1}')
    server._pose_provider = provider
    server._pose_push_source_id = 4242

    assert server._push_pose() is False
    provider.assert_not_called()
    assert server._pose_push_source_id is None


def test_push_pose_disarms_without_provider() -> None:
    """A subscriber but no provider: nothing to push, so the timer is dropped."""
    server = _make_server()
    channel = MagicMock()
    server._pose_channels = {"peer-1": channel}
    server._pose_subscribers = {"peer-1"}
    server._pose_push_source_id = 4242

    assert server._push_pose() is False
    channel.emit.assert_not_called()
    assert server._pose_push_source_id is None


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


def test_arm_pose_push_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """The push timer is armed exactly once, even across repeated calls."""
    added = _stub_timeout_add(monkeypatch)

    server = _make_server()
    with server._pose_lock:
        server._arm_pose_push_locked()
        server._arm_pose_push_locked()

    assert len(added) == 1
    assert added[0][0] == GstMediaServer.POSE_PUSH_INTERVAL_MS
    assert server._pose_push_source_id == 4242

    # Clear the fake source id so `__del__` -> `close()` doesn't try to remove
    # a non-existent GLib source at GC time (noisy unraisable warning).
    server._pose_push_source_id = None


def test_subscribing_arms_the_push_timer(monkeypatch: pytest.MonkeyPatch) -> None:
    """A peer subscribing is what starts the 30 Hz push, not the connection."""
    added = _stub_timeout_add(monkeypatch)
    _stub_idle_add(monkeypatch)

    server = _make_server()
    server._pose_provider = MagicMock(return_value='{"state": {}, "seq": 1}')

    # A connected peer that never subscribes must not cost a periodic wakeup.
    assert added == []

    server.set_pose_subscription("peer-1", True)
    assert len(added) == 1
    assert server._pose_push_source_id == 4242

    # A second subscriber shares the single timer.
    server.set_pose_subscription("peer-2", True)
    assert len(added) == 1

    server._pose_push_source_id = None


def test_timer_rearms_after_disarming(monkeypatch: pytest.MonkeyPatch) -> None:
    """Once the last subscriber leaves, the next subscribe re-arms the timer."""
    added = _stub_timeout_add(monkeypatch)
    _stub_idle_add(monkeypatch)

    server = _make_server()
    server._pose_provider = MagicMock(return_value='{"state": {}, "seq": 1}')

    server.set_pose_subscription("peer-1", True)
    server.set_pose_subscription("peer-1", False)
    # The timer only drops on its next tick, which finds nobody subscribed.
    assert server._push_pose() is False
    assert server._pose_push_source_id is None

    server.set_pose_subscription("peer-1", True)
    assert len(added) == 2
    assert server._pose_push_source_id == 4242

    server._pose_push_source_id = None


def test_set_pose_provider_arms_when_subscribers_already_waiting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A late-wired provider picks up peers that subscribed before it."""
    added = _stub_timeout_add(monkeypatch)
    _stub_idle_add(monkeypatch)

    server = _make_server()
    server.set_pose_subscription("peer-1", True)
    # No provider yet, so nothing to push and nothing armed.
    assert added == []

    server.set_pose_provider(MagicMock(return_value='{"state": {}, "seq": 1}'))
    assert len(added) == 1

    server._pose_push_source_id = None


def test_close_tolerates_a_half_built_server() -> None:
    """`__del__` on an instance whose `__init__` raised must stay silent."""
    server = cast(GstMediaServer, object.__new__(GstMediaServer))
    server._logger = logging.getLogger("test_pose")

    # No pose attrs, no main loop, no bus: must not raise.
    server.close()


def test_close_disarms_an_armed_push_timer(monkeypatch: pytest.MonkeyPatch) -> None:
    """A live timer is removed from the main loop on shutdown."""
    removed: List[Any] = []
    from reachy_mini.media import media_server as ms

    monkeypatch.setattr(ms.GLib, "source_remove", removed.append)

    server = _make_server()
    server._pose_push_source_id = 4242

    server.close()

    assert removed == [4242]
    assert server._pose_push_source_id is None


def test_consumer_removal_drops_pose_state() -> None:
    """A peer vanishing without an `on-close` must not pin the timer on."""
    server = _make_server()
    server._pose_channels = {"peer-1": MagicMock()}
    server._pose_subscribers = {"peer-1"}

    server._drop_pose_peer("peer-1")

    assert server._pose_channels == {}
    assert server._pose_subscribers == set()


def test_subscribing_opens_the_pose_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pose channel is opened on subscribe, never for every peer.

    A client predating label-based routing keeps the last channel it is
    handed as its command channel, so handing it an unsolicited second one
    would silently break its control path.
    """
    _stub_timeout_add(monkeypatch)
    _stub_idle_add(monkeypatch, run=True)

    server = _make_server()
    server._pose_provider = MagicMock(return_value='{"state": {}, "seq": 1}')
    webrtcbin = MagicMock()
    server._peer_webrtcbins = {"peer-1": webrtcbin}

    # Connected but not subscribed: no channel has been created.
    assert webrtcbin.emit.call_count == 0

    server.set_pose_subscription("peer-1", True)

    labels = [c.args[1] for c in webrtcbin.emit.call_args_list]
    assert labels == ["pose"]
    assert "peer-1" in server._pose_channels

    server._pose_push_source_id = None


def test_pose_channel_is_opened_once_per_peer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-subscribing reuses the channel instead of opening another."""
    _stub_timeout_add(monkeypatch)
    _stub_idle_add(monkeypatch, run=True)

    server = _make_server()
    server._pose_provider = MagicMock(return_value='{"state": {}, "seq": 1}')
    webrtcbin = MagicMock()
    server._peer_webrtcbins = {"peer-1": webrtcbin}

    server.set_pose_subscription("peer-1", True)
    server.set_pose_subscription("peer-1", False)
    server.set_pose_subscription("peer-1", True)

    assert webrtcbin.emit.call_count == 1

    server._pose_push_source_id = None


def test_subscribe_without_a_known_peer_drops_the_subscription(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A subscribe for a peer whose webrtcbin is gone is logged, not fatal.

    The peer must not stay subscribed either, or the timer would keep
    building a frame 30 times a second for a peer with no channel.
    """
    _stub_timeout_add(monkeypatch)
    _stub_idle_add(monkeypatch, run=True)

    server = _make_server()
    server._pose_provider = MagicMock(return_value='{"state": {}, "seq": 1}')

    server.set_pose_subscription("ghost-peer", True)

    assert server._pose_channels == {}
    assert server._pose_subscribers == set()
    # Nothing left to push, so the next tick drops the timer.
    assert server._push_pose() is False

    server._pose_push_source_id = None


def test_failed_channel_creation_drops_the_subscription(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`create-data-channel` returning null must not leave a ghost subscriber."""
    _stub_timeout_add(monkeypatch)
    _stub_idle_add(monkeypatch, run=True)

    server = _make_server()
    server._pose_provider = MagicMock(return_value='{"state": {}, "seq": 1}')
    webrtcbin = MagicMock()
    webrtcbin.emit.return_value = None  # GStreamer refused the channel
    server._peer_webrtcbins = {"peer-1": webrtcbin}

    server.set_pose_subscription("peer-1", True)

    assert server._pose_channels == {}
    assert server._pose_subscribers == set()

    server._pose_push_source_id = None
