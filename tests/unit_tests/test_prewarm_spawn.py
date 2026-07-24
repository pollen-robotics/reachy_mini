"""Tests for the early parked-app spawn (before the daemon's heavy imports).

`spawn_early_parked_app` runs at daemon-entry time on the wireless unit and
must be fail-safe: any doubt (no flag, no stamp, no prewarm app, no parking
support, spawn error) returns None and the normal keeper path takes over.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from reachy_mini.apps import prewarm_spawn
from reachy_mini.apps.sources import local_venv_paths

FAKE_APP = textwrap.dedent(
    f"""
    import os, sys

    if os.environ.get("{prewarm_spawn.PARKED_ENV_VAR}") == "1":
        print("{prewarm_spawn.PARKED_READY_SENTINEL}", flush=True)
        line = sys.stdin.readline()
        if not line:
            sys.exit(0)
        print("ACTIVATED", flush=True)
    sys.exit(0)
    """
)

WIRELESS_ARGV = ["main.py", "--wireless-version", "--no-wake-up-on-start"]


@pytest.fixture
def spawnable(tmp_path, monkeypatch):
    """All gates green: stamp present, prewarm configured, fake app spawnable."""
    pkg = tmp_path / "fakeapp"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "__main__.py").write_text(FAKE_APP)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))

    stamp = tmp_path / "stamp.json"
    stamp.write_text("{}")
    monkeypatch.setattr(prewarm_spawn, "STAMP_PATH", stamp)
    monkeypatch.setattr(
        prewarm_spawn.startup_app_config, "get_prewarm_app", lambda: "fakeapp"
    )
    monkeypatch.setattr(
        local_venv_paths, "app_supports_parking", lambda *a, **k: True
    )
    monkeypatch.setattr(local_venv_paths, "get_app_module", lambda *a, **k: "fakeapp")
    monkeypatch.setattr(
        local_venv_paths, "get_app_python", lambda *a, **k: Path(sys.executable)
    )
    return stamp


def test_parked_env_var_matches_app_protocol() -> None:
    """The duplicated constants must stay in sync with the app-side protocol."""
    from reachy_mini.apps.app import PARKED_ENV_VAR, PARKED_READY_SENTINEL

    assert prewarm_spawn.PARKED_ENV_VAR == PARKED_ENV_VAR
    assert prewarm_spawn.PARKED_READY_SENTINEL == PARKED_READY_SENTINEL


def test_stamp_path_matches_startup_check() -> None:
    """The mirrored stamp path must stay in sync with startup_check."""
    from reachy_mini.utils.wireless_version.startup_check import STAMP_PATH

    assert prewarm_spawn.STAMP_PATH == STAMP_PATH


def test_no_spawn_without_wireless_flag(spawnable) -> None:
    assert prewarm_spawn.spawn_early_parked_app(["main.py"]) is None


def test_no_spawn_with_no_autostart(spawnable) -> None:
    argv = WIRELESS_ARGV + ["--no-autostart"]
    assert prewarm_spawn.spawn_early_parked_app(argv) is None


def test_no_spawn_without_stamp(spawnable) -> None:
    """No stamp = startup checks may repair venvs this boot; don't race them."""
    spawnable.unlink()
    assert prewarm_spawn.spawn_early_parked_app(WIRELESS_ARGV) is None


def test_no_spawn_without_prewarm_app(spawnable, monkeypatch) -> None:
    monkeypatch.setattr(
        prewarm_spawn.startup_app_config, "get_prewarm_app", lambda: None
    )
    assert prewarm_spawn.spawn_early_parked_app(WIRELESS_ARGV) is None


def test_no_spawn_without_parking_support(spawnable, monkeypatch) -> None:
    monkeypatch.setattr(
        local_venv_paths, "app_supports_parking", lambda *a, **k: False
    )
    assert prewarm_spawn.spawn_early_parked_app(WIRELESS_ARGV) is None


def test_spawn_failure_returns_none(spawnable, monkeypatch) -> None:
    def boom(*a, **k):
        raise RuntimeError("no module")

    monkeypatch.setattr(local_venv_paths, "get_app_module", boom)
    assert prewarm_spawn.spawn_early_parked_app(WIRELESS_ARGV) is None


def test_spawn_returns_parked_popen(spawnable) -> None:
    """The spawned process parks (sentinel on stdout) and exits 0 on EOF."""
    result = prewarm_spawn.spawn_early_parked_app(WIRELESS_ARGV)
    assert result is not None
    process, name = result
    assert name == "fakeapp"
    try:
        assert process.stdin is not None and process.stdout is not None
        line = process.stdout.readline().decode().strip()
        assert line == prewarm_spawn.PARKED_READY_SENTINEL
        assert process.poll() is None  # parked, waiting on stdin
    finally:
        process.stdin.close()
        assert process.wait(timeout=10) == 0


def test_spawned_env_scrubs_gst_and_sets_parked(spawnable, monkeypatch) -> None:
    monkeypatch.setenv("GST_REGISTRY_1_0", "/daemon/venv/registry")
    env = prewarm_spawn.build_app_env(parked=True)
    assert env[prewarm_spawn.PARKED_ENV_VAR] == "1"
    assert "GST_REGISTRY_1_0" not in env
    assert prewarm_spawn.build_app_env(parked=False).get(
        prewarm_spawn.PARKED_ENV_VAR
    ) is None
