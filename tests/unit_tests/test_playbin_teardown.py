"""Unit tests for ``play_sound()`` playbin teardown on EOS/error."""

import logging
from types import SimpleNamespace
from typing import cast

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

from reachy_mini.media.audio_gstreamer import GStreamerAudio  # noqa: E402

Gst.init([])


class _FakeBus:
    """Stand-in for ``Gst.Bus`` recording signal-watch removal."""

    def __init__(self) -> None:
        self.watch_removed = False

    def remove_signal_watch(self) -> None:
        self.watch_removed = True


class _FakePlaybin:
    """Stand-in for a playbin ``Gst.Element`` recording state changes."""

    def __init__(self) -> None:
        self.bus = _FakeBus()
        self.state: Gst.State | None = None

    def get_bus(self) -> _FakeBus:
        return self.bus

    def set_state(self, state: Gst.State) -> None:
        self.state = state


def _fake_self(playbin: _FakePlaybin | None) -> GStreamerAudio:
    """Return a stand-in carrying just the attrs the helpers touch."""
    return cast(
        GStreamerAudio,
        SimpleNamespace(
            _playbin=playbin,
            logger=logging.getLogger("test_playbin_teardown"),
            _teardown_playbin=lambda pb: None,
        ),
    )


def test_teardown_stops_playbin_and_drops_watch() -> None:
    """Teardown NULLs the playbin, removes its bus watch, clears the ref."""
    playbin = _FakePlaybin()
    fake = _fake_self(playbin)

    GStreamerAudio._teardown_playbin(fake, cast(Gst.Element, playbin))

    assert playbin.state == Gst.State.NULL
    assert playbin.bus.watch_removed
    assert fake._playbin is None


def test_teardown_of_stale_playbin_keeps_current_ref() -> None:
    """Tearing down a superseded playbin leaves the newer one in place."""
    stale, current = _FakePlaybin(), _FakePlaybin()
    fake = _fake_self(current)

    GStreamerAudio._teardown_playbin(fake, cast(Gst.Element, stale))

    assert stale.state == Gst.State.NULL
    assert stale.bus.watch_removed
    assert fake._playbin is cast(Gst.Element, current)


def test_eos_message_triggers_teardown() -> None:
    """An EOS bus message tears down the playbin that produced it."""
    playbin = _FakePlaybin()
    fake = _fake_self(playbin)
    torn_down = []
    fake._teardown_playbin = torn_down.append  # type: ignore[method-assign]
    msg = SimpleNamespace(type=Gst.MessageType.EOS)

    GStreamerAudio._on_playbin_message(
        fake,
        cast(Gst.Bus, playbin.bus),
        cast(Gst.Message, msg),
        cast(Gst.Element, playbin),
    )

    assert torn_down == [playbin]


def test_error_message_triggers_teardown() -> None:
    """An ERROR bus message is logged and tears down the playbin."""
    playbin = _FakePlaybin()
    fake = _fake_self(playbin)
    torn_down = []
    fake._teardown_playbin = torn_down.append  # type: ignore[method-assign]
    gerror = SimpleNamespace(message="boom")
    msg = SimpleNamespace(
        type=Gst.MessageType.ERROR, parse_error=lambda: (gerror, "debug info")
    )

    GStreamerAudio._on_playbin_message(
        fake,
        cast(Gst.Bus, playbin.bus),
        cast(Gst.Message, msg),
        cast(Gst.Element, playbin),
    )

    assert torn_down == [playbin]


def test_other_messages_are_ignored() -> None:
    """Non-EOS/ERROR messages (e.g. STATE_CHANGED) do not tear down."""
    playbin = _FakePlaybin()
    fake = _fake_self(playbin)
    torn_down = []
    fake._teardown_playbin = torn_down.append  # type: ignore[method-assign]
    msg = SimpleNamespace(type=Gst.MessageType.STATE_CHANGED)

    GStreamerAudio._on_playbin_message(
        fake,
        cast(Gst.Bus, playbin.bus),
        cast(Gst.Message, msg),
        cast(Gst.Element, playbin),
    )

    assert torn_down == []
