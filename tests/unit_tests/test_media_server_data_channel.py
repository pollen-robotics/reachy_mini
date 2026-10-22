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
