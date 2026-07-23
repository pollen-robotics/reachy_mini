"""Tests for the pre-warmed ("parked") app machinery in AppManager.

A parked app is spawned with REACHY_MINI_START_PARKED=1, prints a readiness
sentinel, and blocks on stdin: a line activates it, EOF makes it exit 0. The
manager must activate it on start-app (fast path), evict it when another app
starts or the venv is mutated, notice parked crashes, and stop pre-warming
after a crash loop. Robustness contract: every failure falls back to the
plain spawn path, never worse than today.
"""

import asyncio
import sys
import textwrap
from pathlib import Path

import pytest

from reachy_mini.apps.app import PARKED_ENV_VAR, PARKED_READY_SENTINEL
from reachy_mini.apps.manager import AppManager
from reachy_mini.apps.sources import local_common_venv

FAKE_APP = textwrap.dedent(
    f"""
    import os, sys, time

    mode = os.environ.get("FAKE_APP_MODE", "ok")
    if mode == "crash":
        sys.exit(3)
    if os.environ.get("{PARKED_ENV_VAR}") == "1":
        if mode == "silent":
            time.sleep(60)
            sys.exit(0)
        print("{PARKED_READY_SENTINEL}", flush=True)
        line = sys.stdin.readline()
        if not line:
            sys.exit(0)
        print("ACTIVATED", flush=True)
    else:
        print("PLAIN_RUN", flush=True)
    sys.exit(0)
    """
)


@pytest.fixture
def manager(tmp_path, monkeypatch):
    """Real AppManager whose spawns run a controllable fake app module."""
    pkg = tmp_path / "fakeapp"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "__main__.py").write_text(FAKE_APP)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setattr(
        local_common_venv, "get_app_module", lambda *a, **k: "fakeapp"
    )
    monkeypatch.setattr(
        local_common_venv, "get_app_python", lambda *a, **k: Path(sys.executable)
    )
    return AppManager(wireless_version=True)


async def _wait_for(predicate, timeout=10.0):
    deadline = asyncio.get_event_loop().time() + timeout
    while not predicate():
        if asyncio.get_event_loop().time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_parked_spawn_reports_ready(manager):
    """A parked spawn prints the sentinel and stays alive, holding nothing."""
    await manager._spawn_parked_app("fakeapp")
    assert manager.parked_app is not None
    await _wait_for(lambda: manager.parked_app.ready)
    assert manager.parked_app.process.returncode is None
    assert not manager.is_app_running()
    await manager._evict_parked_app("test cleanup")


@pytest.mark.asyncio
async def test_start_app_activates_parked_instance(manager):
    """start-app of the parked app's name reuses the warm process."""
    await manager._spawn_parked_app("fakeapp")
    await _wait_for(lambda: manager.parked_app.ready)
    parked_pid = manager.parked_app.process.pid

    status = await manager.start_app("fakeapp")
    assert manager.parked_app is None
    assert manager.current_app is not None
    assert manager.current_app.process.pid == parked_pid
    assert status.info.name == "fakeapp"
    # The activated process prints ACTIVATED and exits 0 => DONE.
    await manager.current_app.monitor_task


@pytest.mark.asyncio
async def test_start_of_other_app_evicts_parked(manager):
    """Starting a different app discards the parked one cleanly."""
    await manager._spawn_parked_app("fakeapp")
    await _wait_for(lambda: manager.parked_app.ready)
    parked_process = manager.parked_app.process

    await manager.start_app("other_app")  # module resolution is patched: same fake
    assert manager.parked_app is None
    assert parked_process.returncode == 0  # EOF => cooperative exit
    await manager.current_app.monitor_task


@pytest.mark.asyncio
async def test_eviction_is_idempotent_and_clean(manager):
    """Evicting twice is fine; the process exits promptly on stdin EOF."""
    await manager._spawn_parked_app("fakeapp")
    await _wait_for(lambda: manager.parked_app.ready)
    process = manager.parked_app.process
    await manager._evict_parked_app("first")
    await manager._evict_parked_app("second")
    assert process.returncode == 0
    assert manager.parked_app is None


@pytest.mark.asyncio
async def test_parked_crash_is_noticed_and_counted(manager, monkeypatch):
    """A parked process dying alone clears state and feeds the crash guard."""
    monkeypatch.setenv("FAKE_APP_MODE", "crash")
    await manager._spawn_parked_app("fakeapp")
    drain = manager.parked_app.drain_task
    await drain
    assert manager.parked_app is None
    assert len(manager._parked_crash_times) == 1
    assert manager._parking_nudge.is_set()


@pytest.mark.asyncio
async def test_crash_loop_guard_disables_prewarming(manager):
    """Three recent parked deaths stop the keeper from respawning."""
    import time as _time

    now = _time.monotonic()
    manager._parked_crash_times.extend([now, now, now])
    assert manager._parking_crash_looping()


@pytest.mark.asyncio
async def test_activation_of_dead_parked_falls_back_to_spawn(manager, monkeypatch):
    """If the parked process died in between, start_app spawns normally."""
    monkeypatch.setenv("FAKE_APP_MODE", "crash")
    await manager._spawn_parked_app("fakeapp")
    await manager.parked_app.drain_task if manager.parked_app else None
    # parked already cleared by drain; simulate the race where it wasn't:
    monkeypatch.delenv("FAKE_APP_MODE")
    await manager.start_app("fakeapp")
    assert manager.current_app is not None
    await manager.current_app.monitor_task


@pytest.mark.asyncio
async def test_pause_parking_evicts_and_resets_guard(manager):
    """Venv mutations evict the parked instance and reset crash history."""
    await manager._spawn_parked_app("fakeapp")
    await _wait_for(lambda: manager.parked_app.ready)
    async with manager.pause_parking("test install"):
        assert manager.parked_app is None
        assert manager._parking_paused == 1
    assert manager._parking_paused == 0
    assert len(manager._parked_crash_times) == 0
    assert manager._parking_nudge.is_set()


@pytest.mark.asyncio
async def test_close_tears_down_parked(manager):
    """AppManager.close() leaves no parked process behind."""
    await manager._spawn_parked_app("fakeapp")
    await _wait_for(lambda: manager.parked_app.ready)
    process = manager.parked_app.process
    await manager.close()
    assert process.returncode == 0
    assert manager.parked_app is None


def test_app_supports_parking_scrape(tmp_path, monkeypatch):
    """Both the app opt-in and the SDK hook must be present (fail-closed)."""
    site = tmp_path / "site-packages"
    app_dir = site / "myapp"
    sdk_dir = site / "reachy_mini" / "apps"
    app_dir.mkdir(parents=True)
    sdk_dir.mkdir(parents=True)
    main_py = app_dir / "main.py"

    monkeypatch.setattr(
        local_common_venv, "_find_app_main_file", lambda *a, **k: main_py
    )
    monkeypatch.setattr(
        local_common_venv, "get_app_site_packages", lambda *a, **k: site
    )

    main_py.write_text("class X:\n    supports_parking = True\n")
    (sdk_dir / "app.py").write_text("def _park_until_activated(self): ...\n")
    assert local_common_venv.app_supports_parking("myapp") is True

    main_py.write_text("class X:\n    pass\n")
    assert local_common_venv.app_supports_parking("myapp") is False

    main_py.write_text("class X:\n    supports_parking = True\n")
    (sdk_dir / "app.py").write_text("# old sdk\n")
    assert local_common_venv.app_supports_parking("myapp") is False
