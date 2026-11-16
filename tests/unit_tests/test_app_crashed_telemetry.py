"""robot_app_stopped reports crashed only for exits nobody asked for."""

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from reachy_mini.apps import manager as manager_module
from reachy_mini.apps.manager import AppManager
from reachy_mini.apps.sources import local_common_venv

# Exits 3 on SIGINT, like an app that doesn't handle KeyboardInterrupt cleanly.
APP = """
import sys, time
try:
    time.sleep(30 if "{mode}" == "stop" else 0.2)
except KeyboardInterrupt:
    sys.exit(3)
sys.exit(3)
"""


async def _run(mode: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict:
    (tmp_path / "fake_app.py").write_text(APP.format(mode=mode))
    monkeypatch.chdir(tmp_path)  # `python -m` finds the module in the cwd
    monkeypatch.setattr(local_common_venv, "get_app_module", lambda *a: "fake_app")
    monkeypatch.setattr(local_common_venv, "get_app_python", lambda *a: sys.executable)
    capture = MagicMock()
    monkeypatch.setattr(manager_module.telemetry, "capture", capture)

    mngr = AppManager()
    await mngr.start_app("fake_app")
    assert mngr.current_app is not None
    if mode == "stop":
        await asyncio.sleep(0.5)
        await mngr.stop_current_app(timeout=5)
    elif mode == "stop_monitor_first":
        # A stop whose process exit the monitor sees first: it overwrites the
        # STOPPING state with ERROR before reporting.
        mngr.current_app.status.state = manager_module.AppState.STOPPING
        mngr.current_app.stop_requested.set()
        await asyncio.wait_for(mngr.current_app.monitor_task, timeout=10)
    else:
        await asyncio.wait_for(mngr.current_app.monitor_task, timeout=10)
    (stopped,) = [
        c.kwargs for c in capture.call_args_list if c.args == ("robot_app_stopped",)
    ]
    return stopped


@pytest.mark.asyncio
async def test_requested_stop_with_nonzero_exit_is_not_a_crash(monkeypatch, tmp_path):
    """A real stop (SIGINT) ending in exit code 3 is not a crash."""
    stopped = await _run("stop", monkeypatch, tmp_path)
    assert stopped["exit_code"] == 3
    assert stopped["crashed"] is False


@pytest.mark.asyncio
async def test_requested_stop_is_not_a_crash_when_monitor_sees_exit_first(
    monkeypatch, tmp_path
):
    """Not a crash even when the monitor overwrites the STOPPING state first."""
    stopped = await _run("stop_monitor_first", monkeypatch, tmp_path)
    assert stopped["exit_code"] == 3
    assert stopped["crashed"] is False


@pytest.mark.asyncio
async def test_unrequested_nonzero_exit_is_a_crash(monkeypatch, tmp_path):
    """An app exiting non-zero on its own is a crash."""
    stopped = await _run("crash", monkeypatch, tmp_path)
    assert stopped["exit_code"] == 3
    assert stopped["crashed"] is True
