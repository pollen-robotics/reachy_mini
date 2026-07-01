"""stop_current_app must not wake a robot the app deliberately put to sleep."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

from reachy_mini.apps.manager import AppManager
from reachy_mini.daemon.backend.abstract import Backend


def _fake_current_app() -> SimpleNamespace:
    # returncode set + monitor_task done → skip the kill/cancel paths.
    return SimpleNamespace(
        process=SimpleNamespace(returncode=0, pid=0),
        monitor_task=SimpleNamespace(done=lambda: True),
        status=SimpleNamespace(state="running", info=SimpleNamespace(name="x")),
    )


def _manager_with_head_pose(head_pose: np.ndarray) -> tuple[AppManager, AsyncMock]:
    goto_target = AsyncMock()
    backend = SimpleNamespace(
        get_current_head_pose=lambda: head_pose,
        SLEEP_HEAD_POSE=Backend.SLEEP_HEAD_POSE,
        goto_target=goto_target,
    )
    mngr = AppManager(daemon=SimpleNamespace(backend=backend))
    mngr.current_app = _fake_current_app()  # type: ignore[assignment]
    return mngr, goto_target


@pytest.mark.asyncio
async def test_stop_leaves_robot_asleep_when_in_sleep_pose() -> None:
    mngr, goto_target = _manager_with_head_pose(Backend.SLEEP_HEAD_POSE.copy())
    await mngr.stop_current_app()
    goto_target.assert_not_awaited()
    assert mngr.current_app is None


@pytest.mark.asyncio
async def test_stop_returns_to_zero_when_awake() -> None:
    mngr, goto_target = _manager_with_head_pose(np.eye(4))
    await mngr.stop_current_app()
    goto_target.assert_awaited_once()
    assert mngr.current_app is None
