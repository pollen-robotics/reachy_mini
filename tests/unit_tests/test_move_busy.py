"""A move requested while another runs is reported as an error, not as done (#1512)."""

import asyncio
import json
import threading
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from reachy_mini.daemon.backend.abstract import MOVE_BUSY_ERROR
from reachy_mini.io.protocol import PlayRecordedMoveCmd, PlayUploadedMoveCmd


def _up(z: float) -> np.ndarray:
    pose = np.eye(4)
    pose[2, 3] = z
    return pose


@pytest.mark.asyncio
async def test_sibling_goto_on_same_loop_is_rejected(sim_backend: Any) -> None:
    """A second goto on the same loop errors instead of running alongside."""
    replies: list[dict] = []
    first = asyncio.create_task(
        sim_backend._async_goto(replies.append, _up(0.01), None, 1.0, None)
    )
    await asyncio.sleep(0.05)
    await sim_backend._async_goto(replies.append, _up(-0.01), None, 0.2, None)
    assert sim_backend._active_move_depth == 1
    await first

    assert replies == [
        {"error": MOVE_BUSY_ERROR, "command": "goto_target"},
        {"status": "ok", "command": "goto_target", "completed": True},
    ]
    assert not sim_backend.is_move_running


@pytest.mark.asyncio
async def test_goto_from_another_thread_is_rejected(sim_backend: Any) -> None:
    """A goto from another thread errors instead of answering completed."""
    replies: list[dict] = []
    first = asyncio.create_task(sim_backend.goto_target(head=_up(0.01), duration=1.0))
    await asyncio.sleep(0.05)
    other = threading.Thread(
        target=lambda: asyncio.run(
            sim_backend._async_goto(replies.append, _up(-0.01), None, 0.2, None)
        )
    )
    other.start()
    other.join()
    await first

    assert replies == [{"error": MOVE_BUSY_ERROR, "command": "goto_target"}]


@pytest.mark.asyncio
async def test_nested_initial_goto_still_runs(sim_backend: Any) -> None:
    """The initial goto inside a move is nested, not a sibling."""
    move = MagicMock(duration=0.05, sound_path=None)
    move.evaluate.return_value = (_up(0.01), None, None)

    await sim_backend.play_move(move, initial_goto_duration=0.05)

    assert move.evaluate.call_count > 1
    assert not sim_backend.is_move_running


@pytest.mark.asyncio
async def test_uploaded_move_while_busy_is_not_started(
    sim_backend: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """No started broadcast and no orphan audio when a move already runs."""
    sent: list[dict] = []
    monkeypatch.setattr(
        sim_backend, "broadcast_to_all_clients", lambda p: sent.append(json.loads(p))
    )
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    sim_backend._uploaded_moves["m1"] = MagicMock()
    sim_backend._uploaded_audios["m1"] = str(audio)
    sim_backend._active_move_depth = 1

    await sim_backend._async_play_uploaded_move(PlayUploadedMoveCmd(upload_id="m1"))

    assert sent == [
        {"type": "play_uploaded_move", "upload_id": "m1", "error": MOVE_BUSY_ERROR}
    ]
    assert not audio.exists()


@pytest.mark.asyncio
async def test_recorded_move_while_busy_is_not_acked(
    sim_backend: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """play_recorded_move answers an error instead of ok when a move already runs."""
    import reachy_mini.motion.recorded_move as recorded_move

    monkeypatch.setattr(recorded_move, "RecordedMoves", MagicMock())
    sim_backend._active_move_depth = 1
    replies: list[dict] = []

    await sim_backend._async_play_recorded_move(
        PlayRecordedMoveCmd(move_name="happy", dataset_name="d"), replies.append
    )

    assert replies == [
        {"command": "play_recorded_move", "status": "error", "error": MOVE_BUSY_ERROR}
    ]
