"""Unit tests for sending on the per-peer data channels.

A channel that is not open yet (or never opened, its peer gone) must be
skipped: emitting on it is a GStreamer assertion per frame, which flooded
the daemon logs while a phone was re-dialing. Open state is tracked from the
channel's signals, never read from the channel at send time.
"""

from __future__ import annotations

import logging
from typing import cast
from unittest.mock import MagicMock

from reachy_mini.media.media_server import GstMediaServer


def _make_server() -> GstMediaServer:
    """Build a minimal ``GstMediaServer`` instance without booting GStreamer."""
    server = cast(GstMediaServer, object.__new__(GstMediaServer))
    server._logger = logging.getLogger("test_data_channel")
    server._data_channels = {}
    server._created_data_channels = {}
    server._data_channel_session = {}
    server._open_data_channels = set()
    # `__del__` -> `close()` touches these.
    server._loop = MagicMock()
    server._bus_sender = MagicMock()
    return server


def test_broadcast_reaches_open_channels_only() -> None:
    """A channel receives broadcasts between its on-open and on-close."""
    server = _make_server()
    opened, negotiating, closed = MagicMock(), MagicMock(), MagicMock()
    server._data_channels = {"a": opened, "b": negotiating, "c": closed}
    server._on_data_channel_open(opened, "a")
    server._on_data_channel_open(closed, "c")
    server._on_data_channel_close(closed, "c")

    server.send_data_message(None, "hello")

    opened.emit.assert_called_once_with("send-string", "hello")
    negotiating.emit.assert_not_called()
    closed.emit.assert_not_called()
    negotiating.get_property.assert_not_called()


def test_targeted_send_skips_a_channel_not_open_yet() -> None:
    """A reply to a peer whose channel is still negotiating is dropped quietly."""
    server = _make_server()
    channel = MagicMock()
    server._data_channels = {"a": channel}

    server.send_data_message("a", "hello")

    channel.emit.assert_not_called()


def _relayed(server: GstMediaServer, session: str) -> MagicMock:
    """Register a relayed session's channel the way `_setup_data_channel` does.

    Every relayed session shares the relay's one consumer peer id ("relay").
    """
    channel = MagicMock()
    server._data_channels["relay"] = channel
    server._data_channel_session["relay"] = session
    return channel


def test_removing_another_session_of_the_peer_keeps_the_live_one_sending() -> None:
    """An orphan or superseded session going away must not silence the live one.

    Open state used to be keyed by peer id, which all relayed sessions
    share: the orphan's removal cleared it and the live session got no
    data-channel messages until it died.
    """
    server = _make_server()
    _relayed(server, "orphan")
    live = _relayed(server, "live")  # registered last, as in the real race
    server._on_data_channel_open(live, "relay", "live")

    # The orphan is torn down: its on-close fires (or only consumer removal).
    server._on_data_channel_close(MagicMock(), "relay", "orphan")
    server._open_data_channels.discard("orphan")

    server.send_data_message(None, "hello")
    live.emit.assert_called_once_with("send-string", "hello")
    assert server._data_channels["relay"] is live


def test_the_live_session_closing_stops_its_sends() -> None:
    """Its own on-close still stops sends and unregisters the channel."""
    server = _make_server()
    live = _relayed(server, "live")
    server._on_data_channel_open(live, "relay", "live")
    server._on_data_channel_close(live, "relay", "live")

    server.send_data_message(None, "hello")
    live.emit.assert_not_called()
    assert "relay" not in server._data_channels


def _create_channel(server: GstMediaServer, session: str) -> tuple[MagicMock, dict]:
    """Run the real `_setup_data_channel` for a relayed session.

    Returns the channel and its connected handlers ({signal: (cb, *args)}),
    so a test fires them exactly as GStreamer would.
    """
    channel = MagicMock()
    handlers: dict = {}
    channel.connect.side_effect = lambda sig, cb, *a: handlers.setdefault(sig, (cb, *a))
    webrtcbin = MagicMock()
    webrtcbin.emit.return_value = channel
    server._setup_data_channel("relay", webrtcbin, session)
    return channel, handlers


def _fire(handlers: dict, signal: str, channel: MagicMock) -> None:
    cb, *args = handlers[signal]
    cb(channel, *args)


def test_an_orphan_created_after_the_live_session_does_not_displace_it() -> None:
    """An orphan created after the live session opened cannot displace it.

    The duplicate-start race: the live session opens, then an orphan of the
    same relay peer is created (and never opens), then torn down. The orphan
    used to overwrite the registered channel at creation, so the live session
    got nothing. Channels are now registered at on-open.
    """
    server = _make_server()
    server._on_data_channel_message = MagicMock()  # type: ignore[method-assign]
    server._on_session_dc_message = MagicMock()  # type: ignore[method-assign]
    server._on_data_channel_error = MagicMock()  # type: ignore[method-assign]
    server._peer_webrtcbins = {}
    live, live_handlers = _create_channel(server, "live")
    _fire(live_handlers, "on-open", live)
    orphan, orphan_handlers = _create_channel(server, "orphan")  # never opens
    _fire(orphan_handlers, "on-close", orphan)
    server._open_data_channels.discard("orphan")  # its consumer removal

    server.send_data_message("relay", "targeted")
    server.send_data_message(None, "broadcast")
    sent = [c.args for c in live.emit.call_args_list if c.args[0] == "send-string"]
    assert sent == [("send-string", "targeted"), ("send-string", "broadcast")]
    assert not [c for c in orphan.emit.call_args_list if c.args[0] == "send-string"]


def test_closed_channels_are_released() -> None:
    """Each channel's creation-time reference is released at its on-close.

    Direct LAN consumers each have their own peer id: holding the reference
    past on-close leaked one data channel per session for the daemon's life.
    """
    server = _make_server()
    server._on_data_channel_message = MagicMock()  # type: ignore[method-assign]
    server._on_session_dc_message = MagicMock()  # type: ignore[method-assign]
    server._on_data_channel_error = MagicMock()  # type: ignore[method-assign]
    server._peer_webrtcbins = {}
    for i in range(5):
        channel = MagicMock()
        handlers: dict = {}
        channel.connect.side_effect = lambda sig, cb, *a, h=handlers: h.setdefault(
            sig, (cb, *a)
        )
        webrtcbin = MagicMock()
        webrtcbin.emit.return_value = channel
        server._setup_data_channel(f"lan-{i}", webrtcbin, None)
        _fire(handlers, "on-open", channel)
        _fire(handlers, "on-close", channel)
    assert server._created_data_channels == {}
    assert server._data_channels == {}
    assert server._data_channel_session == {}


def test_an_orphan_closing_keeps_the_live_reference() -> None:
    """An orphan's on-close releases only its own reference."""
    server = _make_server()
    server._on_data_channel_message = MagicMock()  # type: ignore[method-assign]
    server._on_session_dc_message = MagicMock()  # type: ignore[method-assign]
    server._on_data_channel_error = MagicMock()  # type: ignore[method-assign]
    server._peer_webrtcbins = {}
    orphan, orphan_handlers = _create_channel(server, "orphan")
    live, live_handlers = _create_channel(server, "live")
    _fire(live_handlers, "on-open", live)
    _fire(orphan_handlers, "on-close", orphan)
    assert server._created_data_channels["relay"] is live
