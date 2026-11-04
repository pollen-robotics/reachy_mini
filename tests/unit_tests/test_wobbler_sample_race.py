"""Regression tests for #1517: the wobbler appsink callback vs. (dis|en)able_wobbling.

``_on_wobbler_sample`` runs on a GStreamer streaming thread while
``disable_wobbling`` / ``enable_wobbling`` run on whichever thread handles the
app's command. The tests below cover both ``GStreamerAudio`` (local backend)
and ``GstMediaServer`` (daemon-side WebRTC), using the real callback, the real
``(dis|en)able_wobbling`` and a real ``HeadWobbler``. Only the GStreamer
appsink/sample/buffer and ``GLib.timeout_add`` are replaced by test doubles.

The race tests pin the interleaving with events instead of sleeping: the audio
thread is parked inside ``buffer.extract_dup`` (after the ``_head_wobbler``
check, before ``feed``), then the other thread toggles wobbling.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import gi
import numpy as np
import pytest

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

from reachy_mini.media.audio_gstreamer import GStreamerAudio  # noqa: E402
from reachy_mini.media.media_server import GstMediaServer  # noqa: E402
from reachy_mini.motion.head_wobbler import SpeechOffsets  # noqa: E402

ZERO: SpeechOffsets = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
JOIN_TIMEOUT_S = 5.0  # hang guard only; no step depends on timing


def _make_audio() -> GStreamerAudio:
    """``GStreamerAudio`` without its pipelines (``__init__`` boots GStreamer)."""
    inst = cast(GStreamerAudio, object.__new__(GStreamerAudio))
    inst.logger = logging.getLogger("test_wobbler_race.audio")
    inst._head_wobbler = None
    inst._head_wobbler_lock = threading.Lock()
    # Stubs for `__del__` -> `cleanup()`.
    inst._doa = SimpleNamespace(close=lambda: None)
    inst._loop = SimpleNamespace(quit=lambda: None)
    inst._bus = SimpleNamespace(remove_watch=lambda: None)
    return inst


def _make_server() -> GstMediaServer:
    """``GstMediaServer`` with only the wobbler attributes wired up."""
    inst = cast(GstMediaServer, object.__new__(GstMediaServer))
    inst._logger = logging.getLogger("test_wobbler_race.server")
    inst._head_wobbler = None
    inst._head_wobbler_lock = threading.Lock()
    # Stubs for `__del__` -> `close()`.
    inst._loop = MagicMock()
    inst._bus_sender = MagicMock()
    return inst


MAKERS = pytest.mark.parametrize(
    "make", [_make_audio, _make_server], ids=["GStreamerAudio", "GstMediaServer"]
)


def _speech_pcm() -> np.ndarray:
    """Half a second of a loud tone: enough for SwayRollRT to emit hops."""
    t = np.arange(8_000, dtype=np.float32) / 16_000
    return (np.sin(2 * np.pi * 220 * t) * 0.5).astype(np.float32)


class _FakeBuffer:
    def __init__(self, pcm: np.ndarray, on_extract: Callable[[], None]) -> None:
        self._data = pcm.tobytes()
        self._on_extract = on_extract
        self.extracted = False

    def get_size(self) -> int:
        return len(self._data)

    def extract_dup(self, offset: int, size: int) -> bytes:
        self.extracted = True
        self._on_extract()
        return self._data[offset : offset + size]


class _FakeAppsink:
    def __init__(self, buffer: _FakeBuffer | None) -> None:
        self._sample = (
            None if buffer is None else SimpleNamespace(get_buffer=lambda: buffer)
        )

    def pull_sample(self) -> Any:
        return self._sample


class _RecordingWobbler:
    def __init__(self) -> None:
        self.fed: list[tuple[np.ndarray, int]] = []

    def feed(self, pcm: np.ndarray, play_at_monotonic_ns: int) -> None:
        self.fed.append((pcm, play_at_monotonic_ns))

    def stop(self) -> None:
        pass


class _ContentionSignallingLock:
    """A lock that reports when a thread other than the owner tries to take it.

    Lets the test know that the toggling thread is parked on the lock without
    sleeping. Code that takes no lock never calls it.
    """

    def __init__(self, on_foreign_attempt: Callable[[], None]) -> None:
        self._lock = threading.Lock()
        self._owner: int | None = None
        self._on_foreign_attempt = on_foreign_attempt

    def __enter__(self) -> None:
        if self._owner is not None and self._owner != threading.get_ident():
            self._on_foreign_attempt()
        self._lock.acquire()
        self._owner = threading.get_ident()

    def __exit__(self, *exc: object) -> None:
        self._owner = None
        self._lock.release()


@pytest.fixture
def glib_schedule(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, tuple]]:
    """Record ``GLib.timeout_add`` calls from ``HeadWobbler.feed``."""
    schedule: list[tuple[Any, tuple]] = []

    def fake_timeout_add(_delay_ms: int, fn: Any, *args: Any) -> int:
        schedule.append((fn, args))
        return len(schedule)

    monkeypatch.setattr(
        "reachy_mini.motion.head_wobbler.GLib.timeout_add", fake_timeout_add
    )
    return schedule


def _run_toggle_during_sample(
    inst: Any, toggle: Callable[[], None]
) -> tuple[list[BaseException], list[Any]]:
    """Run ``toggle`` on another thread while the audio thread is mid-sample.

    Returns the exceptions raised on either thread and the callback's results.
    """
    in_window = threading.Event()
    release_audio = threading.Event()
    toggle_progressed = threading.Event()  # toggle returned, or blocked on the lock
    inst._head_wobbler_lock = _ContentionSignallingLock(toggle_progressed.set)

    def park_audio_thread() -> None:
        in_window.set()
        assert release_audio.wait(JOIN_TIMEOUT_S)

    appsink = _FakeAppsink(_FakeBuffer(_speech_pcm(), park_audio_thread))
    errors: list[BaseException] = []
    results: list[Any] = []

    def audio_thread() -> None:
        try:
            results.append(inst._on_wobbler_sample(appsink))
        except BaseException as e:  # surfaced through `errors`
            errors.append(e)

    def toggle_thread() -> None:
        try:
            toggle()
        except BaseException as e:
            errors.append(e)
        finally:
            toggle_progressed.set()

    audio = threading.Thread(target=audio_thread)
    audio.start()
    assert in_window.wait(JOIN_TIMEOUT_S)
    other = threading.Thread(target=toggle_thread)
    other.start()
    assert toggle_progressed.wait(JOIN_TIMEOUT_S)
    release_audio.set()
    audio.join(JOIN_TIMEOUT_S)
    other.join(JOIN_TIMEOUT_S)
    assert not audio.is_alive() and not other.is_alive(), "threads deadlocked"
    return errors, results


def _fire_all(schedule: list[tuple[Any, tuple]]) -> None:
    """Fire every recorded timeout, as the GLib main loop would."""
    for fn, args in schedule:
        fn(*args)


# ---------------------------------------------------------------------------
# Plain callback behaviour
# ---------------------------------------------------------------------------


@MAKERS
def test_sample_is_fed_to_the_wobbler(make: Callable[[], Any]) -> None:
    """A sample's PCM reaches the wobbler and the callback returns OK."""
    inst = make()
    wobbler = _RecordingWobbler()
    inst._head_wobbler = wobbler
    pcm = _speech_pcm()

    result = inst._on_wobbler_sample(_FakeAppsink(_FakeBuffer(pcm, lambda: None)))

    assert result == Gst.FlowReturn.OK
    assert len(wobbler.fed) == 1
    np.testing.assert_array_equal(wobbler.fed[0][0], pcm)


@MAKERS
def test_sample_without_wobbler_is_dropped(make: Callable[[], Any]) -> None:
    """With wobbling disabled the buffer is not even read."""
    inst = make()
    buffer = _FakeBuffer(_speech_pcm(), lambda: None)

    assert inst._on_wobbler_sample(_FakeAppsink(buffer)) == Gst.FlowReturn.OK
    assert not buffer.extracted


@MAKERS
def test_missing_sample_is_ignored(make: Callable[[], Any]) -> None:
    """``pull_sample()`` returning None (flushing / EOS) is not an error."""
    inst = make()
    wobbler = _RecordingWobbler()
    inst._head_wobbler = wobbler

    assert inst._on_wobbler_sample(_FakeAppsink(None)) == Gst.FlowReturn.OK
    assert wobbler.fed == []


# ---------------------------------------------------------------------------
# #1517: toggling wobbling while a sample is in flight
# ---------------------------------------------------------------------------


@MAKERS
def test_disable_during_sample_neither_raises_nor_moves_the_head_after(
    make: Callable[[], Any], glib_schedule: list[tuple[Any, tuple]]
) -> None:
    """Disabling mid-sample: no AttributeError, and the head stays at zero.

    Before the fix the callback re-read ``self._head_wobbler`` after another
    thread set it to None. Reading it once is not enough on its own: the
    callback would then feed a wobbler that ``stop()`` already ran on, and
    those hops would fire after the head was zeroed.
    """
    inst = make()
    applied: list[SpeechOffsets] = []
    inst.enable_wobbling(applied.append)

    errors, results = _run_toggle_during_sample(inst, inst.disable_wobbling)

    assert errors == []
    assert results == [Gst.FlowReturn.OK]
    assert inst._head_wobbler is None
    assert glib_schedule, "the in-flight sample should have produced hops"
    _fire_all(glib_schedule)
    assert applied == [ZERO], "offsets were applied after wobbling was disabled"


@MAKERS
def test_reenable_during_sample_cancels_the_old_wobblers_hops(
    make: Callable[[], Any], glib_schedule: list[tuple[Any, tuple]]
) -> None:
    """Replacing the wobbler mid-sample: the old one's hops never fire."""
    inst = make()
    old_applied: list[SpeechOffsets] = []
    new_applied: list[SpeechOffsets] = []
    inst.enable_wobbling(old_applied.append)
    old = inst._head_wobbler

    errors, results = _run_toggle_during_sample(
        inst, lambda: inst.enable_wobbling(new_applied.append)
    )

    assert errors == []
    assert results == [Gst.FlowReturn.OK]
    assert inst._head_wobbler is not None and inst._head_wobbler is not old
    assert glib_schedule, "the in-flight sample should have produced hops"
    _fire_all(glib_schedule)
    assert old_applied == [ZERO], "the replaced wobbler kept moving the head"
