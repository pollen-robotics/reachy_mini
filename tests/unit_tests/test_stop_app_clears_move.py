"""App stop must clear an in-flight daemon move (issue #1401)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from reachy_mini.apps.manager import AppManager
from reachy_mini.daemon.app.routers import move as move_router
from reachy_mini.daemon.backend.abstract import Backend


def _fake_current_app() -> SimpleNamespace:
    return SimpleNamespace(
        process=SimpleNamespace(returncode=0, pid=0),
        monitor_task=SimpleNamespace(done=lambda: True),
        status=SimpleNamespace(state="running", info=SimpleNamespace(name="x")),
    )


@pytest.mark.asyncio
async def test_stop_requests_stop_move_when_move_running() -> None:
    """stop_current_app must request stop_move before idle reset."""
    goto_target = AsyncMock()
    request_stop_move = MagicMock(return_value=True)
    backend = SimpleNamespace(
        get_current_head_pose=lambda: np.eye(4),
        SLEEP_HEAD_POSE=Backend.SLEEP_HEAD_POSE,
        goto_target=goto_target,
        set_motor_control_mode=MagicMock(),
        is_move_running=True,
        request_stop_move=request_stop_move,
    )

    # After stop is requested, pretend the move clears immediately.
    def _stop() -> bool:
        backend.is_move_running = False
        return True

    request_stop_move.side_effect = _stop

    mngr = AppManager(daemon=SimpleNamespace(backend=backend))
    mngr.current_app = _fake_current_app()  # type: ignore[assignment]

    await mngr.stop_current_app()

    request_stop_move.assert_called_once()
    goto_target.assert_awaited_once()
    assert mngr.current_app is None


@pytest.mark.asyncio
async def test_request_stop_move_flips_flag(sim_backend: Any) -> None:
    """Backend.request_stop_move sets the stop flag when a move is active."""
    sim_backend._active_move_depth = 1
    assert sim_backend.is_move_running
    assert sim_backend.request_stop_move() is True
    assert sim_backend._stop_move_requested is True
    assert sim_backend.request_stop_move() is True  # still running until _end_move


@pytest.mark.asyncio
async def test_cancel_all_move_tasks_empty() -> None:
    """cancel_all_move_tasks is a no-op when nothing is registered."""
    move_router.move_tasks.clear()
    await move_router.cancel_all_move_tasks()
    assert move_router.move_tasks == {}


@pytest.mark.asyncio
async def test_cancel_all_move_tasks_cancels_registered() -> None:
    """cancel_all_move_tasks cancels every registered HTTP move task."""
    move_router.move_tasks.clear()
    started = asyncio.Event()
    release = asyncio.Event()

    async def long_move() -> None:
        started.set()
        await release.wait()

    uuid = move_router.create_move_task(long_move()).uuid
    await started.wait()
    assert uuid in move_router.move_tasks

    await move_router.cancel_all_move_tasks()
    assert uuid not in move_router.move_tasks
    release.set()


@pytest.mark.asyncio
async def test_stop_timeout_warns_and_next_start_allowed(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A move that outlives the stop deadline logs a warning, not an error."""
    from reachy_mini.apps import manager as manager_mod

    clock = [0.0]

    def fake_monotonic() -> float:
        clock[0] += 1.0
        return clock[0]

    monkeypatch.setattr(manager_mod.time, "monotonic", fake_monotonic)
    backend = SimpleNamespace(
        get_current_head_pose=lambda: np.eye(4),
        SLEEP_HEAD_POSE=Backend.SLEEP_HEAD_POSE,
        goto_target=AsyncMock(),
        set_motor_control_mode=MagicMock(),
        is_move_running=True,
        request_stop_move=MagicMock(return_value=True),
    )
    mngr = AppManager(daemon=SimpleNamespace(backend=backend))
    mngr.current_app = _fake_current_app()  # type: ignore[assignment]

    with caplog.at_level("WARNING"):
        await mngr.stop_current_app()

    assert "Move still running" in caplog.text
    assert mngr.current_app is None
    assert not mngr.is_app_running()
