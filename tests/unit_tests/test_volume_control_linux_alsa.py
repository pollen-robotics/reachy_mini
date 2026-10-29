"""ALSA backend of VolumeControlLinux, with subprocess.run faked (no audio hardware)."""

import subprocess
import sys
from typing import Any

import pytest

if sys.platform != "linux":
    pytest.skip("volume_control_linux needs pulsectl, Linux only", allow_module_level=True)

from reachy_mini.daemon.app.routers import volume_control_linux  # noqa: E402
from reachy_mini.daemon.app.routers.volume_control_linux import (  # noqa: E402
    VolumeControlLinux,
)

APLAY = "card 0: Headphones [bcm2835 Headphones], device 0: bcm2835 Headphones\n"
ARECORD = "card 2: Mic [Reachy Mini Audio Mic], device 0: USB Audio [USB Audio]\n"


def _volume_control() -> VolumeControlLinux:
    # Skip __post_init__: it scans real hardware.
    return object.__new__(VolumeControlLinux)


def _fake_run(monkeypatch: pytest.MonkeyPatch, outputs: dict[str, Any]) -> list[list[str]]:
    """Answer each command by its tool name; an exception value is raised."""
    calls: list[list[str]] = []

    def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert isinstance(cmd, list) and not kwargs.get("shell")
        calls.append(cmd)
        out = outputs[cmd[0]]
        if isinstance(out, BaseException):
            raise out
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setattr(volume_control_linux.subprocess, "run", run)
    return calls


def test_scan_lists_capture_only_cards(monkeypatch: pytest.MonkeyPatch) -> None:
    """A card that only arecord lists is found."""
    calls = _fake_run(monkeypatch, {"aplay": APLAY, "arecord": ARECORD})

    devices = _volume_control()._alsa_get_all_devices()

    assert devices == {0: "bcm2835 Headphones", 2: "Reachy Mini Audio Mic"}
    assert calls == [["aplay", "-l"], ["arecord", "-l"]]


def test_scan_survives_one_tool_failing(monkeypatch: pytest.MonkeyPatch) -> None:
    """aplay failing still leaves the capture cards."""
    _fake_run(
        monkeypatch,
        {"aplay": subprocess.CalledProcessError(1, "aplay"), "arecord": ARECORD},
    )

    assert _volume_control()._alsa_get_all_devices() == {2: "Reachy Mini Audio Mic"}


def test_scan_raises_when_both_tools_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """No aplay and no arecord: the scan fails loudly as before."""
    _fake_run(monkeypatch, {"aplay": FileNotFoundError(), "arecord": FileNotFoundError()})

    with pytest.raises(RuntimeError, match="Could not scan audio devices"):
        _volume_control()._alsa_get_all_devices()
