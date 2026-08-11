"""Tests for the wireless startup-check stamp fast path.

The stamp lets boots skip the expensive startup checks (full /venvs ownership
scan, apps_venv SDK sync probe, restore-venv pip check) when nothing changed
since the last fully-successful run. These tests pin the safety contract:
any doubt (missing/corrupt/stale stamp, failed check, editable install,
ownership probe failure) must fall back to the full checks, and a stamp is
only ever written after every expensive check succeeded.
"""

import json
import logging
import os
from pathlib import Path

import pytest

from reachy_mini.utils.wireless_version import startup_check as sc

DAEMON_INFO = {
    "version": "1.9.0",
    "source": "git",
    "git_ref": "main",
    "commit": "abc12345",
}


def _make_venv(root: Path, with_dist: bool = True) -> Path:
    site = root / "lib/python3.12/site-packages"
    site.mkdir(parents=True)
    (root / "bin").mkdir()
    (root / "bin/python").write_text("#!fake\n")
    if with_dist:
        dist = site / "reachy_mini-1.9.0.dist-info"
        dist.mkdir()
        (dist / "RECORD").write_text("reachy_mini/__init__.py,,\n")
    return root


@pytest.fixture
def fake_venvs(tmp_path, monkeypatch):
    """Fake /venvs + /restore trees with dist-info, patched into the module."""
    daemon = _make_venv(tmp_path / "venvs/mini_daemon")
    apps = _make_venv(tmp_path / "venvs/apps_venv")
    restore = _make_venv(tmp_path / "restore/venvs/mini_daemon")
    stamp = daemon / ".startup_check_stamp.json"

    monkeypatch.setattr(sc, "DAEMON_VENV", daemon)
    monkeypatch.setattr(sc, "APPS_VENV", apps)
    monkeypatch.setattr(sc, "RESTORE_VENV", restore)
    monkeypatch.setattr(sc, "STAMP_PATH", stamp)
    monkeypatch.setattr(
        "reachy_mini.utils.wireless_version.update_available.get_install_source",
        lambda name: dict(DAEMON_INFO),
    )
    # The bounded ownership probe needs the "pollen" user; not present on CI.
    monkeypatch.setattr(sc, "_quick_ownership_ok", lambda: True)
    return {"daemon": daemon, "apps": apps, "restore": restore, "stamp": stamp}


@pytest.fixture
def spy_checks(monkeypatch):
    """Replace all six checks with recording fakes (all succeed by default)."""
    calls = []

    def make(name, ret=True):
        def fake(*args, **kwargs):
            calls.append(name)
            return ret

        return fake

    monkeypatch.setattr(sc, "check_and_fix_venvs_ownership", make("ownership"))
    monkeypatch.setattr(sc, "check_and_sync_apps_venv_sdk", make("apps_sync"))
    monkeypatch.setattr(sc, "check_and_fix_restore_venv", make("restore"))
    monkeypatch.setattr(sc, "check_and_update_bluetooth_service", make("bluetooth", None))
    monkeypatch.setattr(sc, "check_and_update_wireless_launcher", make("launcher", None))
    monkeypatch.setattr(
        sc, "check_and_update_gpio_shutdown_service", make("gpio_shutdown", None)
    )
    return calls


EXPENSIVE = {"ownership", "apps_sync", "restore"}
CHEAP = {"bluetooth", "launcher", "gpio_shutdown"}


def test_first_run_does_full_checks_and_writes_stamp(fake_venvs, spy_checks):
    """Without a stamp, every check runs and a stamp is written on success."""
    sc.run_wireless_startup_checks()
    assert set(spy_checks) == EXPENSIVE | CHEAP
    assert fake_venvs["stamp"].exists()
    stored = json.loads(fake_venvs["stamp"].read_text())
    assert stored["daemon"] == DAEMON_INFO
    assert stored["format"] == sc.STAMP_FORMAT


def test_valid_stamp_skips_expensive_but_runs_cheap_checks(fake_venvs, spy_checks):
    """A valid stamp skips only the expensive checks."""
    sc.run_wireless_startup_checks()
    spy_checks.clear()

    sc.run_wireless_startup_checks()
    assert set(spy_checks) == CHEAP


def test_failed_check_prevents_stamp_and_clears_existing(
    fake_venvs, spy_checks, monkeypatch
):
    """A failing check must clear any stamp and prevent writing one."""
    sc.run_wireless_startup_checks()
    assert fake_venvs["stamp"].exists()

    # Invalidate so the full path runs again, this time with a failing check.
    os.utime(
        next(fake_venvs["daemon"].glob("lib/python*/site-packages/*.dist-info/RECORD")),
        ns=(1, 1),
    )
    monkeypatch.setattr(
        sc, "check_and_fix_restore_venv", lambda *a, **k: (spy_checks.append("restore"), False)[1]
    )
    spy_checks.clear()
    sc.run_wireless_startup_checks()
    assert set(spy_checks) == EXPENSIVE | CHEAP
    assert not fake_venvs["stamp"].exists()


def test_reinstall_invalidates_stamp(fake_venvs, spy_checks):
    """A dist-info RECORD mtime change (any reinstall) invalidates the stamp."""
    sc.run_wireless_startup_checks()
    spy_checks.clear()

    # A pip/uv (re)install rewrites dist-info; mtime change must invalidate.
    record = next(
        fake_venvs["apps"].glob("lib/python*/site-packages/*.dist-info/RECORD")
    )
    os.utime(record, ns=(123, 456))

    sc.run_wireless_startup_checks()
    assert set(spy_checks) == EXPENSIVE | CHEAP
    # And the stamp was refreshed to the new signature: next run is fast again.
    spy_checks.clear()
    sc.run_wireless_startup_checks()
    assert set(spy_checks) == CHEAP


def test_corrupt_stamp_falls_back_to_full_checks(fake_venvs, spy_checks):
    """A corrupt stamp file falls back to the full checks."""
    fake_venvs["stamp"].write_text("{not json")
    sc.run_wireless_startup_checks()
    assert set(spy_checks) == EXPENSIVE | CHEAP
    # Corrupt stamp got replaced by a valid one.
    json.loads(fake_venvs["stamp"].read_text())


def test_editable_install_never_uses_fast_path(fake_venvs, spy_checks, monkeypatch):
    """Editable installs never fast-path and never write a stamp."""
    monkeypatch.setattr(
        "reachy_mini.utils.wireless_version.update_available.get_install_source",
        lambda name: {"version": "1.9.0", "source": "editable"},
    )
    sc.run_wireless_startup_checks()
    assert set(spy_checks) == EXPENSIVE | CHEAP
    # Signature computation refuses editable installs → no stamp written.
    assert not fake_venvs["stamp"].exists()

    spy_checks.clear()
    sc.run_wireless_startup_checks()
    assert set(spy_checks) == EXPENSIVE | CHEAP


def test_ownership_probe_failure_disables_fast_path(fake_venvs, spy_checks, monkeypatch):
    """A failing ownership probe forces the full checks."""
    sc.run_wireless_startup_checks()
    spy_checks.clear()

    monkeypatch.setattr(sc, "_quick_ownership_ok", lambda: False)
    sc.run_wireless_startup_checks()
    assert set(spy_checks) == EXPENSIVE | CHEAP


def test_missing_daemon_dist_info_disables_fast_path(fake_venvs, spy_checks):
    """Losing the daemon dist-info disables the fast path and drops the stamp."""
    sc.run_wireless_startup_checks()
    spy_checks.clear()

    for dist in fake_venvs["daemon"].glob("lib/python*/site-packages/*.dist-info"):
        (dist / "RECORD").unlink()
        dist.rmdir()

    sc.run_wireless_startup_checks()
    assert set(spy_checks) == EXPENSIVE | CHEAP
    # No dist-info → signature can't be computed → stamp write fails safely.
    assert not fake_venvs["stamp"].exists()


def test_dist_record_sig_none_without_distribution(tmp_path):
    """No reachy_mini distribution in a venv yields a None signature."""
    venv = _make_venv(tmp_path / "v", with_dist=False)
    assert sc._dist_record_sig(venv) is None


def test_clear_startup_stamp_tolerates_missing_file(fake_venvs):
    """Clearing an absent stamp is a no-op, not an error."""
    assert not fake_venvs["stamp"].exists()
    sc.clear_startup_stamp()  # must not raise


def test_stamp_write_is_atomic_no_tmp_left_behind(fake_venvs, spy_checks):
    """Atomic stamp write leaves no temp file behind."""
    sc.run_wireless_startup_checks()
    leftovers = list(fake_venvs["daemon"].glob("*.tmp"))
    assert leftovers == []


def test_stale_stamp_says_why_full_checks_run(fake_venvs, spy_checks, caplog):
    """A parseable-but-stale stamp must say WHY the full checks run.

    Field debugging relies on the journal: a silent fallback would look
    identical to a boot that never had a stamp.
    """
    sc.run_wireless_startup_checks()
    record = next(
        fake_venvs["daemon"].glob("lib/python*/site-packages/*.dist-info/RECORD")
    )
    os.utime(record, ns=(9, 9))

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=sc.logger.name):
        sc.run_wireless_startup_checks()

    assert "Startup stamp mismatch" in caplog.text


def test_stamp_path_being_a_directory_never_crashes_boot(fake_venvs, spy_checks):
    """A directory squatting the stamp path degrades to full checks, no crash."""
    fake_venvs["stamp"].mkdir()
    sc.run_wireless_startup_checks()  # must not raise
    assert set(spy_checks) == EXPENSIVE | CHEAP


@pytest.mark.asyncio
async def test_update_clears_stamp_before_first_install(fake_venvs, monkeypatch):
    """update_reachy_mini drops the stamp before any venv is touched.

    This is the first defense against interrupted updates: even if the
    install is killed halfway, the next boot cannot find a valid stamp.
    """
    from reachy_mini.utils.wireless_version import update as up

    fake_venvs["stamp"].write_text("{}")
    stamp_present_at_install = []

    async def fake_call(cmd, logger, env=None, ok_returncodes=(0,)):
        stamp_present_at_install.append(fake_venvs["stamp"].exists())

    monkeypatch.setattr(up, "build_install_command", lambda **kw: ("true", None))
    monkeypatch.setattr(up, "call_logger_wrapper", fake_call)

    await up.update_reachy_mini(sc.logger)

    assert stamp_present_at_install, "install command was never invoked"
    assert stamp_present_at_install[0] is False


@pytest.mark.asyncio
async def test_update_logs_error_but_continues_when_stamp_is_stuck(
    fake_venvs, monkeypatch, caplog
):
    """An unremovable stamp is an ERROR with a traceback, never a failed update.

    A stamp we cannot delete means the filesystem or the ownership is broken,
    which deserves a loud signal. It must not abort the update though: the
    install signature invalidates the stamp on the next boot anyway, and
    updating is how a robot in a bad state gets repaired.
    """
    from reachy_mini.utils.wireless_version import update as up

    def stuck() -> None:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(sc, "clear_startup_stamp", stuck)
    monkeypatch.setattr(up, "build_install_command", lambda **kw: ("true", None))

    installed = []

    async def fake_call(cmd, logger, env=None, ok_returncodes=(0,)):
        installed.append(cmd)

    monkeypatch.setattr(up, "call_logger_wrapper", fake_call)

    with caplog.at_level("ERROR"):
        await up.update_reachy_mini(sc.logger)

    assert installed, "the update must go ahead despite the stuck stamp"
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert errors, "a stuck stamp must be logged at ERROR, not swallowed"
    assert errors[0].exc_info is not None, "the exception must reach the log"


def test_boot_path_survives_an_unremovable_stamp(fake_venvs, spy_checks, monkeypatch):
    """On the boot path a stuck stamp degrades to full checks, never a crash.

    run_wireless_startup_checks is called during daemon startup, so letting an
    OSError escape here would stop the robot from booting.
    """

    def stuck() -> None:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(sc, "clear_startup_stamp", stuck)
    monkeypatch.setattr(sc, "check_and_fix_restore_venv", lambda: False)

    sc.run_wireless_startup_checks()  # must not raise

    assert not fake_venvs["stamp"].exists()
