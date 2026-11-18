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
def records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(manager, "_running_apps_dir", lambda: tmp_path)
    return tmp_path


def _sleeper() -> "subprocess.Popen[bytes]":
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])


def _write(
    records: Path,
    app: "subprocess.Popen[bytes]",
    daemon_pid: int,
    daemon_create_time: float,
    app_create_time_shift: float = 0.0,
) -> Path:
    path = records / f"{daemon_pid}.json"
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
    return path


def _dead_daemon() -> tuple[int, float]:
    proc = _sleeper()
    create_time = psutil.Process(proc.pid).create_time()
    proc.kill()
    proc.wait()
    return proc.pid, create_time


def _live_daemon() -> tuple[int, float]:
    me = psutil.Process()
    return me.pid, me.create_time()


def test_orphan_of_dead_daemon_is_killed(records: Path) -> None:
    app = _sleeper()
    try:
        path = _write(records, app, *_dead_daemon())
        AppManager()
        assert app.poll() is not None  # _kill_orphan waits for the exit.
        assert not path.exists()
    finally:
        app.kill()


def test_app_of_live_daemon_is_kept(records: Path) -> None:
    app = _sleeper()
    try:
        path = _write(records, app, *_live_daemon())
        AppManager()
        assert app.poll() is None
        assert path.exists()
    finally:
        app.kill()


def test_only_the_dead_daemons_app_is_killed(records: Path) -> None:
    orphan, kept = _sleeper(), _sleeper()
    try:
        _write(records, orphan, *_dead_daemon())
        kept_path = _write(records, kept, *_live_daemon())
        AppManager()
        assert orphan.poll() is not None
        assert kept.poll() is None
        assert list(records.iterdir()) == [kept_path]
    finally:
        orphan.kill()
        kept.kill()


def test_recycled_pid_is_not_killed(records: Path) -> None:
    app = _sleeper()
    try:
        path = _write(records, app, *_dead_daemon(), app_create_time_shift=-100)
        AppManager()
        assert app.poll() is None
        assert not path.exists()
    finally:
        app.kill()


def test_unreadable_record_is_dropped(records: Path) -> None:
    path = records / "123.json"
    path.write_text("{not json")
    AppManager()
    assert not path.exists()
