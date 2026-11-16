"""SoundPlaybin frees play_sound's playbin on EOS, stop and replacement."""

import gc
import logging
import time
import wave
from collections.abc import Callable
from pathlib import Path

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

from reachy_mini.media.gstreamer_utils import SoundPlaybin  # noqa: E402

Gst.init([])


def _wav(tmp_path: Path, seconds: float = 0.1) -> str:
    path = tmp_path / "silence.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\0\0" * int(16000 * seconds))
    return path.as_uri()


def _playbin(uri: str, freed: list[str], name: str) -> Gst.Element:
    pb = Gst.ElementFactory.make("playbin", None)
    pb.set_property("uri", uri)
    sink = Gst.ElementFactory.make("fakesink", None)
    sink.set_property("sync", True)  # real-time, so a long file is still playing
    pb.set_property("audio-sink", sink)
    pb.weak_ref(lambda: freed.append(name))
    return pb


def _spin_until(cond: Callable[[], bool], timeout: float = 5.0) -> None:
    ctx = GLib.MainContext.default()
    deadline = time.monotonic() + timeout
    while not cond() and time.monotonic() < deadline:
        ctx.iteration(False)
        time.sleep(0.005)


def _assert_freed(freed: list[str], name: str) -> None:
    _spin_until(lambda: gc.collect() >= 0 and name in freed)
    assert name in freed, f"playbin {name} leaked"


def test_eos_frees_playbin(tmp_path: Path) -> None:
    """A sound that ends on its own frees its playbin and calls on_end."""
    ended: list[int] = []
    freed: list[str] = []
    slot = SoundPlaybin(logging.getLogger(__name__), on_end=lambda: ended.append(1))

    slot.play(_playbin(_wav(tmp_path), freed, "a"))
    _spin_until(lambda: bool(ended))

    assert ended == [1]
    assert slot.current is None
    _assert_freed(freed, "a")


def test_stop_and_replace_free_playbins(tmp_path: Path) -> None:
    """A replaced or stopped sound frees its playbin without on_end."""
    ended: list[int] = []
    freed: list[str] = []
    uri = _wav(tmp_path, seconds=5)
    slot = SoundPlaybin(logging.getLogger(__name__), on_end=lambda: ended.append(1))

    slot.play(_playbin(uri, freed, "a"))
    slot.play(_playbin(uri, freed, "b"))
    _assert_freed(freed, "a")
    slot.stop()
    _assert_freed(freed, "b")

    assert slot.current is None
    assert ended == []  # on_end is only for sounds that end on their own
