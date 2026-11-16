"""A move requested while another runs is reported as an error, not as done (#1512)."""

import asyncio
import json
import threading
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import numpy as np
import pytest

from reachy_mini import MoveBusyError
from reachy_mini.io.protocol import (
    MOVE_BUSY_ERROR,
    PlayRecordedMoveCmd,
    PlayUploadedMoveCmd,
)
from reachy_mini.io.ws_client import TaskState, WSClient


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
async def test_goto_between_wake_up_steps_is_rejected(sim_backend: Any) -> None:
    """A client goto can't slip between wake_up's gotos and abort it halfway."""
    woke = MagicMock()
    sim_backend._on_wake_up_callback = woke
    replies: list[dict] = []
    wake = asyncio.create_task(sim_backend.wake_up(force=True))
    await asyncio.sleep(0.15)  # past wake_up's initial 0.1 s settle
    while not wake.done():
        await sim_backend._async_goto(replies.append, None, [0.1, -0.1], 0.05, None)
        await asyncio.sleep(0.01)
    await wake

    woke.assert_called_once()
    assert replies
    assert all(
        r == {"error": MOVE_BUSY_ERROR, "command": "goto_target"} for r in replies
    )
    assert not sim_backend.is_move_running


@pytest.mark.asyncio
async def test_uploaded_move_while_busy_is_kept_for_retry(
    sim_backend: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """No started broadcast when a move already runs; the upload stays playable."""
    sent: list[dict] = []
    monkeypatch.setattr(
        sim_backend, "broadcast_to_all_clients", lambda p: sent.append(json.loads(p))
    )
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    move = MagicMock()
    sim_backend._uploaded_moves["m1"] = move
    sim_backend._uploaded_audios["m1"] = str(audio)
    sim_backend._active_move_depth = 1

    await sim_backend._async_play_uploaded_move(PlayUploadedMoveCmd(upload_id="m1"))

    assert sent == [
        {"type": "play_uploaded_move", "upload_id": "m1", "error": MOVE_BUSY_ERROR}
    ]
    assert sim_backend._uploaded_moves["m1"] is move
    assert sim_backend._uploaded_audios["m1"] == str(audio)
    assert audio.exists()


def test_sdk_raises_move_busy_error() -> None:
    """The SDK client turns the daemon's busy reply into a MoveBusyError."""
    client = WSClient.__new__(WSClient)
    client._tasks_lock = threading.Lock()
    uid = uuid4()
    client.tasks = {uid: TaskState(event=threading.Event(), error=MOVE_BUSY_ERROR)}
    client.tasks[uid].event.set()

    with pytest.raises(MoveBusyError, match=MOVE_BUSY_ERROR):
        client.wait_for_task_completion(uid)
    assert uid not in client.tasks


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
