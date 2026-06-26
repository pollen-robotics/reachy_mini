"""Unit tests for the shared appsrc gap-aware buffer push helper."""

from types import SimpleNamespace
from typing import cast

import gi
import numpy as np

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

from reachy_mini.media.audio_base import AudioBase  # noqa: E402

Gst.init([])


class _FakeAppsrc:
    """Stand-in for ``Gst.AppSrc`` capturing the pushed buffer."""

    def __init__(self, running_time_ns: int) -> None:
        self._running_time = running_time_ns
        self.pushed: Gst.Buffer | None = None

    def get_current_running_time(self) -> int:
        return self._running_time

    def push_buffer(self, buf: Gst.Buffer) -> Gst.FlowReturn:
        self.pushed = buf
        return Gst.FlowReturn.OK


def _fake_self(running_time_ns: int, prev_pts_ns: int) -> AudioBase:
    """Return a stand-in carrying just the attrs the helper touches."""
    return cast(
        AudioBase,
        SimpleNamespace(
            SAMPLE_RATE=AudioBase.SAMPLE_RATE,
            GAP_RESET_NS=AudioBase.GAP_RESET_NS,
            _appsrc=_FakeAppsrc(running_time_ns),
            _appsrc_pts=prev_pts_ns,
        ),
    )


def test_push_first_buffer_anchors_to_running_time() -> None:
    """First buffer after start/flush is stamped with running-time + DISCONT."""
    fake = _fake_self(running_time_ns=2_000_000_000, prev_pts_ns=-1)
    data = np.zeros(1600, dtype=np.float32)

    ret = AudioBase._push_appsrc_buffer(fake, data)

    assert ret == Gst.FlowReturn.OK
    buf = fake._appsrc.pushed
    assert buf is not None
    assert buf.has_flags(Gst.BufferFlags.DISCONT)
    assert buf.pts == 2_000_000_000
    assert buf.dts == 2_000_000_000
    assert fake._appsrc_pts == 2_100_000_000


def test_push_continues_without_gap() -> None:
    """Follow-up buffers within GAP_RESET_NS are placed contiguously, untimestamped."""
    fake = _fake_self(running_time_ns=1_050_000_000, prev_pts_ns=1_100_000_000)
    data = np.zeros(800, dtype=np.float32)

    AudioBase._push_appsrc_buffer(fake, data)

    buf = fake._appsrc.pushed
    assert buf is not None
    assert not buf.has_flags(Gst.BufferFlags.DISCONT)
    assert buf.pts == Gst.CLOCK_TIME_NONE
    assert buf.dts == Gst.CLOCK_TIME_NONE
    assert fake._appsrc_pts == 1_150_000_000


def test_push_resets_after_large_gap() -> None:
    """A gap larger than GAP_RESET_NS re-anchors to running-time + DISCONT."""
    fake = _fake_self(running_time_ns=1_400_000_000, prev_pts_ns=1_100_000_000)
    data = np.zeros(800, dtype=np.float32)

    AudioBase._push_appsrc_buffer(fake, data)

    buf = fake._appsrc.pushed
    assert buf is not None
    assert buf.has_flags(Gst.BufferFlags.DISCONT)
    assert buf.pts == 1_400_000_000
    assert fake._appsrc_pts == 1_450_000_000


def test_push_without_appsrc_returns_none() -> None:
    """Helper is a no-op when ``_appsrc`` is not initialized."""
    fake = cast(
        AudioBase,
        SimpleNamespace(
            SAMPLE_RATE=AudioBase.SAMPLE_RATE,
            GAP_RESET_NS=AudioBase.GAP_RESET_NS,
            _appsrc_pts=-1,
        ),
    )

    ret = AudioBase._push_appsrc_buffer(fake, np.zeros(100, dtype=np.float32))

    assert ret is None
