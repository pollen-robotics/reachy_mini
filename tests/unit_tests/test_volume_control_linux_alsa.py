"""ALSA backend of VolumeControlLinux, with subprocess.run faked (no audio hardware)."""

import subprocess
import sys
from typing import Any, Callable

import pytest

if sys.platform != "linux":
    pytest.skip("volume_control_linux needs pulsectl, Linux only", allow_module_level=True)

from reachy_mini.daemon.app.routers import volume_control_linux  # noqa: E402
from reachy_mini.daemon.app.routers.volume_control import (  # noqa: E402
    AudioDevice,
    AudioDeviceType,
)
from reachy_mini.daemon.app.routers.volume_control_linux import (  # noqa: E402
    VolumeControlLinux,
)

APLAY = "card 0: Headphones [bcm2835 Headphones], device 0: bcm2835 Headphones\n"
ARECORD = "card 2: Mic [Reachy Mini Audio Mic], device 0: USB Audio [USB Audio]\n"


def _volume_control() -> VolumeControlLinux:
    # Skip __post_init__: it scans real hardware.
    return object.__new__(VolumeControlLinux)


def _fake_run(
    monkeypatch: pytest.MonkeyPatch, outputs: dict[str, Any] | Callable[[list[str]], Any]
) -> list[list[str]]:
    """Answer each command by its tool name (or a function of it); an exception value is raised."""
    calls: list[list[str]] = []

    def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert isinstance(cmd, list) and not kwargs.get("shell")
        calls.append(cmd)
        out = outputs(cmd) if callable(outputs) else outputs[cmd[0]]
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


OUTPUT = AudioDevice(0, "Card", AudioDeviceType.OUTPUT)
SCONTENTS = (
    "Simple mixer control 'Master Playback',0\n"
    "  Capabilities: pvolume\n"
    "Simple mixer control 'PCM',0\n"
    "  Capabilities: pvolume\n"
)


def _amixer(fail_controls: tuple[str, ...] = (), scontents: str = SCONTENTS) -> Callable:
    def answer(cmd: list[str]) -> Any:
        if cmd[-1] == "scontents":
            return scontents
        if cmd[4].split(",")[0] in fail_controls:
            return subprocess.CalledProcessError(1, cmd)
        return "  Front Left: Playback 40 [62%] [on]\n"

    return answer


def test_set_volume_passes_control_names_with_spaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control name is one argument, not split or shell-parsed."""
    calls = _fake_run(monkeypatch, _amixer())

    assert _volume_control()._alsa_set_device_volume(OUTPUT, 150)

    assert calls[-1] == ["amixer", "-c", "0", "sset", "Master Playback,0", "100%"]


def test_set_volume_falls_through_to_the_next_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A control that fails is skipped, like the old `||` chain."""
    calls = _fake_run(monkeypatch, _amixer(fail_controls=("Master Playback",)))

    assert _volume_control()._alsa_set_device_volume(OUTPUT, 40)

    assert [c[4] for c in calls[1:]] == ["Master Playback,0", "PCM,0"]


def test_set_volume_without_controls_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """No volume control on the card is a failure, not a silent success."""
    _fake_run(monkeypatch, _amixer(scontents=""))

    assert not _volume_control()._alsa_set_device_volume(OUTPUT, 40)


def test_get_volume_reads_the_first_working_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """sget runs with the same argument list and its percentage is parsed."""
    calls = _fake_run(monkeypatch, _amixer())

    assert _volume_control()._alsa_get_device_volume(OUTPUT) == 62

    assert calls[-1] == ["amixer", "-c", "0", "sget", "Master Playback,0"]
