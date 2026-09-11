"""Tests for the gpio-shutdown systemd unit self-heal check.

Older images generated /etc/systemd/system/gpio-shutdown-daemon.service at
install time with an ExecStart pointing at wherever the SDK lived back then,
so launcher fixes shipped in the package never reached updated robots. The
startup check must sync the packaged unit on every boot: copy + reload +
restart on a difference, install + enable when missing, and never touch
anything when the installed unit already matches.
"""

import subprocess
from pathlib import Path

import pytest

from reachy_mini.utils.wireless_version import startup_check as sc

PACKAGED_UNIT = (
    Path(sc.__file__).parent
    / "../../daemon/app/services/gpio_shutdown/gpio-shutdown-daemon.service"
).resolve()


@pytest.fixture
def run_spy(monkeypatch):
    """Record subprocess.run invocations instead of running them (no sudo/systemctl)."""
    calls: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(sc.subprocess, "run", fake_run)
    return calls


@pytest.fixture
def fake_target(tmp_path, monkeypatch):
    """Redirect the /etc/systemd/system target into a temp dir."""
    target = tmp_path / "gpio-shutdown-daemon.service"
    monkeypatch.setattr(sc, "GPIO_SHUTDOWN_UNIT_TARGET", target)
    return target


def test_packaged_unit_exists_and_points_at_venv_launcher() -> None:
    """The packaged unit ships with the SDK and targets the fleet venv path."""
    content = PACKAGED_UNIT.read_text()
    assert (
        "ExecStart=/venvs/mini_daemon/lib/python3.12/site-packages/"
        "reachy_mini/daemon/app/services/gpio_shutdown/launcher.sh" in content
    )
    assert "User=pollen" in content


def test_target_differs_gets_copied_and_restarted(fake_target, run_spy) -> None:
    """An installed unit with stale content is replaced and the service restarted."""
    fake_target.write_text(
        "[Service]\nExecStart=/venvs/src/reachy_mini/src/reachy_mini/daemon/app/"
        "services/gpio_shutdown/launcher.sh\n"
    )

    sc.check_and_update_gpio_shutdown_service()

    assert run_spy == [
        ["sudo", "cp", str(PACKAGED_UNIT), str(fake_target)],
        ["sudo", "systemctl", "daemon-reload"],
        ["sudo", "systemctl", "restart", "gpio-shutdown-daemon"],
    ]


def test_target_same_is_untouched(fake_target, run_spy) -> None:
    """A matching installed unit triggers no subprocess call at all."""
    fake_target.write_text(PACKAGED_UNIT.read_text())

    sc.check_and_update_gpio_shutdown_service()

    assert run_spy == []


def test_target_missing_gets_installed_and_enabled(fake_target, run_spy) -> None:
    """A missing unit is installed and enabled so it survives the next boot."""
    assert not fake_target.exists()

    sc.check_and_update_gpio_shutdown_service()

    assert run_spy == [
        ["sudo", "cp", str(PACKAGED_UNIT), str(fake_target)],
        ["sudo", "systemctl", "daemon-reload"],
        ["sudo", "systemctl", "enable", "--now", "gpio-shutdown-daemon"],
    ]


def test_source_missing_is_a_noop(fake_target, run_spy, monkeypatch) -> None:
    """Without a packaged unit (broken install) the check must not touch systemd."""
    real_exists = Path.exists

    def fake_exists(self: Path) -> bool:
        if self == PACKAGED_UNIT:
            return False
        return real_exists(self)

    monkeypatch.setattr(Path, "exists", fake_exists)

    sc.check_and_update_gpio_shutdown_service()

    assert run_spy == []


def test_failed_systemctl_is_tolerated(fake_target, run_spy, monkeypatch) -> None:
    """A failing sudo/systemctl call must not raise out of the startup check."""
    fake_target.write_text("stale\n")

    def failing_run(cmd, *args, **kwargs):
        run_spy.append(list(cmd))
        raise subprocess.CalledProcessError(1, cmd, stderr="permission denied")

    monkeypatch.setattr(sc.subprocess, "run", failing_run)

    sc.check_and_update_gpio_shutdown_service()  # must not raise

    assert run_spy == [["sudo", "cp", str(PACKAGED_UNIT), str(fake_target)]]


@pytest.fixture
def real_fs_run(monkeypatch):
    """subprocess.run spy that actually performs cp/rm, so target state is real.

    The plain run_spy records without touching the filesystem, which cannot
    show what the next boot would see after a half-applied update.
    """
    calls: list[list[str]] = []
    fail_on: dict[str, int] = {"index": -1}

    def fake_run(cmd, *args, **kwargs):
        cmd = list(cmd)
        calls.append(cmd)
        if len(calls) - 1 == fail_on["index"]:
            raise subprocess.CalledProcessError(1, cmd, stderr="systemctl refused")
        if cmd[:2] == ["sudo", "cp"]:
            Path(cmd[3]).write_bytes(Path(cmd[2]).read_bytes())
        elif cmd[:3] == ["sudo", "rm", "-f"]:
            Path(cmd[3]).unlink(missing_ok=True)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(sc.subprocess, "run", fake_run)
    return calls, fail_on


def test_failed_daemon_reload_is_retried_on_the_next_boot(fake_target, real_fs_run):
    """A copy followed by a failed daemon-reload must not look done next boot.

    Leaving the new unit in place would make filecmp report equality forever
    while systemd still runs the old one.
    """
    calls, fail_on = real_fs_run
    fake_target.write_text("stale unit\n")
    fail_on["index"] = 1  # the daemon-reload, right after a successful cp

    sc.check_and_update_gpio_shutdown_service()

    assert fake_target.read_text() == "stale unit\n", (
        "the half-applied unit must be rolled back, otherwise the next boot "
        "sees matching files and skips the sync forever"
    )

    # Next boot: the difference is visible again, so the work is retried.
    calls.clear()
    fail_on["index"] = -1
    sc.check_and_update_gpio_shutdown_service()

    assert ["sudo", "cp", str(PACKAGED_UNIT), str(fake_target)] in calls
    assert ["sudo", "systemctl", "daemon-reload"] in calls
    assert fake_target.read_bytes() == PACKAGED_UNIT.read_bytes()


def test_failed_enable_on_fresh_install_is_retried_on_the_next_boot(
    fake_target, real_fs_run
):
    """The severe case: a unit installed but never enabled starts on no boot.

    A reboot does not rescue this one (an un-enabled unit stays un-enabled),
    so the check has to notice and retry rather than see matching files.
    """
    calls, fail_on = real_fs_run
    assert not fake_target.exists()
    fail_on["index"] = 2  # enable --now, after cp and daemon-reload

    sc.check_and_update_gpio_shutdown_service()

    assert not fake_target.exists(), (
        "a unit that was copied but never enabled must not be left behind"
    )

    calls.clear()
    fail_on["index"] = -1
    sc.check_and_update_gpio_shutdown_service()

    assert ["sudo", "systemctl", "enable", "--now", "gpio-shutdown-daemon"] in calls
    assert fake_target.read_bytes() == PACKAGED_UNIT.read_bytes()


def test_failed_copy_leaves_the_installed_unit_alone(fake_target, real_fs_run):
    """When the copy itself fails there is nothing to roll back."""
    calls, fail_on = real_fs_run
    fake_target.write_text("stale unit\n")
    fail_on["index"] = 0  # the cp

    sc.check_and_update_gpio_shutdown_service()

    assert fake_target.read_text() == "stale unit\n"
    assert not any(c[:3] == ["sudo", "rm", "-f"] for c in calls), (
        "no rollback should be attempted when nothing was copied"
    )
