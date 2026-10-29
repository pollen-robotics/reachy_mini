"""Tests for keeping Wi-Fi power save off on the robot.

With power save on (brcmfmac's default, which NetworkManager leaves alone),
traffic to the robot waits at the access point and remote clients see stalls.
The startup check must install a NetworkManager drop-in that disables it for
every activation, reload NetworkManager only when the drop-in changed, and
turn power save off on the current link on every boot. A failure in either
part must never break the daemon's startup.
"""

import subprocess

import pytest

from reachy_mini.utils.wireless_version import startup_check as sc

IW_OFF = ["sudo", "iw", "dev", "wlan0", "set", "power_save", "off"]
RELOAD = ["sudo", "nmcli", "general", "reload", "conf"]


@pytest.fixture
def fake_target(tmp_path, monkeypatch):
    """Redirect the /etc/NetworkManager/conf.d drop-in into a temp dir."""
    target = tmp_path / "reachy-mini-wifi-powersave.conf"
    monkeypatch.setattr(sc, "WIFI_POWERSAVE_CONF_TARGET", target)
    return target


class RunSpy(list):
    """subprocess.run calls, plus the commands (by name) made to fail."""

    def __init__(self) -> None:
        """Start with no calls and nothing failing."""
        super().__init__()
        self.failing: set[str] = set()


@pytest.fixture
def run_spy(monkeypatch, fake_target):
    """Record subprocess.run calls; `sudo install` copies into the fake target."""
    calls = RunSpy()

    def fake_run(cmd, *args, **kwargs):
        cmd = list(cmd)
        calls.append(cmd)
        if cmd[1] in calls.failing:
            raise subprocess.CalledProcessError(1, cmd, stderr=f"{cmd[1]} failed")
        if cmd[1] == "install":
            fake_target.write_text(open(cmd[-2]).read())
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(sc.subprocess, "run", fake_run)
    return calls


def test_drop_in_disables_powersave_for_every_connection() -> None:
    """The drop-in sets NetworkManager's global default to 2 (disable)."""
    assert "[connection]\nwifi.powersave = 2\n" in sc.WIFI_POWERSAVE_CONF


def test_missing_drop_in_is_installed_then_nm_reloaded(fake_target, run_spy) -> None:
    """First boot: install the drop-in, reload NetworkManager, then iw off."""
    sc.check_and_disable_wifi_powersave()

    assert fake_target.read_text() == sc.WIFI_POWERSAVE_CONF
    assert run_spy[0][:4] == ["sudo", "install", "-m", "644"]
    assert run_spy[0][-1] == str(fake_target)
    assert run_spy[1:] == [RELOAD, IW_OFF]


def test_stale_drop_in_is_replaced(fake_target, run_spy) -> None:
    """A drop-in with other content is rewritten and NetworkManager reloaded."""
    fake_target.write_text("[connection]\nwifi.powersave = 3\n")

    sc.check_and_disable_wifi_powersave()

    assert fake_target.read_text() == sc.WIFI_POWERSAVE_CONF
    assert RELOAD in run_spy


def test_up_to_date_drop_in_only_turns_powersave_off(fake_target, run_spy) -> None:
    """Every later boot leaves NetworkManager alone but still fixes the link."""
    fake_target.write_text(sc.WIFI_POWERSAVE_CONF)

    sc.check_and_disable_wifi_powersave()

    assert run_spy == [IW_OFF]


def test_failures_are_logged_never_raised(fake_target, run_spy, caplog) -> None:
    """A missing nmcli or wlan0 must not break the daemon's startup."""
    run_spy.failing.update({"nmcli", "iw"})

    sc.check_and_disable_wifi_powersave()

    assert "Failed to install the Wi-Fi power save drop-in" in caplog.text
    assert "Could not turn Wi-Fi power save off on wlan0" in caplog.text
    # iw still runs after the drop-in failed: the current link is fixed anyway.
    assert run_spy[-1] == IW_OFF
