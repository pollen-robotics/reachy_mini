"""An app left running by a dead daemon is killed when the next daemon starts."""

import json
import subprocess
import sys
from pathlib import Path

import psutil
import pytest

from reachy_mini.apps import manager
from reachy_mini.apps.manager import AppManager


@pytest.fixture
def record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / ".running_app.json"
    monkeypatch.setattr(manager, "_running_app_file", lambda: path)
    return path


def _sleeper() -> "subprocess.Popen[bytes]":
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])


def _write(
    path: Path,
    app: "subprocess.Popen[bytes]",
    daemon_pid: int,
    daemon_create_time: float,
    app_create_time_shift: float = 0.0,
) -> None:
    create_time = psutil.Process(app.pid).create_time() + app_create_time_shift
    path.write_text(
        json.dumps(
            {
                "pid": app.pid,
                "create_time": create_time,
                "daemon_pid": daemon_pid,
                "daemon_create_time": daemon_create_time,
            }
        )
    )


def _dead_daemon() -> tuple[int, float]:
    proc = _sleeper()
    create_time = psutil.Process(proc.pid).create_time()
    proc.kill()
    proc.wait()
    return proc.pid, create_time


def test_orphan_of_dead_daemon_is_killed(record: Path) -> None:
    app = _sleeper()
    try:
        _write(record, app, *_dead_daemon())
        AppManager()
        assert app.wait(timeout=5) is not None
        assert not record.exists()
    finally:
        app.kill()


def test_app_of_live_daemon_is_kept(record: Path) -> None:
    app = _sleeper()
    try:
        me = psutil.Process()
        _write(record, app, me.pid, me.create_time())
        AppManager()
        assert app.poll() is None
        assert record.exists()
    finally:
        app.kill()


def test_recycled_pid_is_not_killed(record: Path) -> None:
    app = _sleeper()
    try:
        _write(record, app, *_dead_daemon(), app_create_time_shift=-100)
        AppManager()
        assert app.poll() is None
    finally:
        app.kill()
