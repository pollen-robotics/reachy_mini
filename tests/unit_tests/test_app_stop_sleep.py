"""stop_current_app must not wake a robot the app deliberately put to sleep."""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from reachy_mini.apps import manager as manager_module
from reachy_mini.apps.manager import AppManager
from reachy_mini.apps.sources import local_common_venv
from reachy_mini.daemon.backend.abstract import Backend
from reachy_mini.daemon.robot_app_lock import RobotAppLock, RobotAppLockState
from reachy_mini.io.protocol import MotorControlMode


def _fake_current_app() -> SimpleNamespace:
    # returncode set + monitor_task done → skip the kill/cancel paths.
    return SimpleNamespace(
        process=SimpleNamespace(returncode=0, pid=0),
        monitor_task=SimpleNamespace(done=lambda: True),
        status=SimpleNamespace(state="running", info=SimpleNamespace(name="x")),
        stop_requested=MagicMock(),
    )


def _manager_with_head_pose(head_pose: np.ndarray) -> tuple[AppManager, AsyncMock]:
    reset_to_init = AsyncMock()
    backend = SimpleNamespace(
        get_current_head_pose=lambda: head_pose,
        SLEEP_HEAD_POSE=Backend.SLEEP_HEAD_POSE,
        reset_to_init=reset_to_init,
        set_motor_control_mode=MagicMock(),
    )
    mngr = AppManager(daemon=SimpleNamespace(backend=backend))
    mngr.current_app = _fake_current_app()  # type: ignore[assignment]
    return mngr, reset_to_init


@pytest.mark.asyncio
async def test_stop_leaves_robot_asleep_when_in_sleep_pose() -> None:
    mngr, reset_to_init = _manager_with_head_pose(Backend.SLEEP_HEAD_POSE.copy())
    await mngr.stop_current_app()
    reset_to_init.assert_not_awaited()
    # left asleep, and made limp
    mngr.daemon.backend.set_motor_control_mode.assert_called_once_with(
        MotorControlMode.Disabled
    )
    assert mngr.current_app is None


@pytest.mark.asyncio
async def test_stop_returns_to_zero_when_awake() -> None:
    mngr, reset_to_init = _manager_with_head_pose(np.eye(4))
    await mngr.stop_current_app()
    reset_to_init.assert_awaited_once()
    assert mngr.current_app is None


@pytest.mark.asyncio
@pytest.mark.parametrize("requested", [True, False])
async def test_only_unrequested_exits_trigger_the_idle_reset(
    requested: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A user stop cleans up itself; an app exiting on its own gets the idle reset."""
    (tmp_path / "fake_app.py").write_text(
        f"import time\ntry:\n    time.sleep({30 if requested else 0.2})\n"
        "except KeyboardInterrupt:\n    pass\n"
    )
    monkeypatch.chdir(tmp_path)  # `python -m` finds the module in the cwd
    monkeypatch.setattr(local_common_venv, "get_app_module", lambda *a: "fake_app")
    monkeypatch.setattr(local_common_venv, "get_app_python", lambda *a: sys.executable)
    monkeypatch.setattr(manager_module.telemetry, "capture", MagicMock())
    lock = RobotAppLock()
    freed: list[bool] = []
    lock.set_on_became_free_handler(freed.append)

    mngr = AppManager(daemon=SimpleNamespace(robot_app_lock=lock, backend=None))
    await mngr.start_app("fake_app")
    assert mngr.current_app is not None
    if requested:
        await asyncio.sleep(0.5)
        await mngr.stop_current_app(timeout=5)
    else:
        await asyncio.wait_for(mngr.current_app.monitor_task, timeout=10)

    assert lock.status().state == RobotAppLockState.FREE
    assert freed == ([] if requested else [False])
