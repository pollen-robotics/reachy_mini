"""Tests for the Rerun logging helper."""

import logging
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("rerun")

from reachy_mini.utils import rerun as rerun_module  # noqa: E402


def test_log_camera_waits_through_missing_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WebRTC returns None until a frame is ready: keep reading, log the frame."""
    rr = MagicMock()
    monkeypatch.setattr(rerun_module, "rr", rr)
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    running = Event()
    frames = iter([None, None, frame])

    def get_frame() -> np.ndarray | None:
        f = next(frames)
        if f is not None:
            running.set()  # stop after this frame
        return f

    helper = rerun_module.Rerun.__new__(rerun_module.Rerun)
    helper._reachymini = SimpleNamespace(
        media=SimpleNamespace(camera=object(), get_frame=get_frame)
    )
    helper.running = running
    helper.recording = None
    helper.logger = logging.getLogger("test")

    helper.log_camera()

    logged = [c.args[0] for c in rr.log.call_args_list]
    assert "camera/image" in logged
