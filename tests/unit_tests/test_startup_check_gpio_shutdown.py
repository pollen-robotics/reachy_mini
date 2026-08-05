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
