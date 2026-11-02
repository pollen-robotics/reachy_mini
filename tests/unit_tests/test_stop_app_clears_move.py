"""App stop must clear an in-flight daemon move (issue #1401)."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from reachy_mini.apps.manager import AppManager
from reachy_mini.daemon.backend.abstract import Backend, _PlaybackCancelToken


@pytest.mark.asyncio
async def test_stop_stops_the_move_before_leaving_the_robot_asleep() -> None:
    """Near the sleep pose, the app's move is stopped before the motors go limp."""
    calls: list[str] = []
    backend = SimpleNamespace(
        get_current_head_pose=lambda: Backend.SLEEP_HEAD_POSE.copy(),
        SLEEP_HEAD_POSE=Backend.SLEEP_HEAD_POSE,
        stop_running_move=AsyncMock(side_effect=lambda: calls.append("stop_move")),
        set_motor_control_mode=MagicMock(side_effect=lambda m: calls.append("limp")),
        reset_to_init=AsyncMock(),
        request_idle_reset=MagicMock(),
    )
    mngr = AppManager(daemon=SimpleNamespace(backend=backend))
    mngr.current_app = SimpleNamespace(  # type: ignore[assignment]
        process=SimpleNamespace(returncode=0, pid=0),
        monitor_task=SimpleNamespace(done=lambda: True),
        status=SimpleNamespace(state="running", info=SimpleNamespace(name="x")),
        stop_requested=MagicMock(),
    )

    await mngr.stop_current_app()

    assert calls == ["stop_move", "limp"]
    backend.reset_to_init.assert_not_awaited()


@pytest.mark.asyncio
async def test_stop_running_move_flags_the_move_and_warns_if_it_lingers(
    sim_backend: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """The stop flag and the upload token are set; a move that won't end is logged."""
    sim_backend._active_move_depth = 1  # a move that never ends
    token = _PlaybackCancelToken("u1")
    sim_backend._active_move_token = token

    with caplog.at_level("WARNING"):
        await sim_backend.stop_running_move(timeout=0.05)

    assert sim_backend._stop_move_requested is True
    assert token.cancelled is True
    assert "still running" in caplog.text
