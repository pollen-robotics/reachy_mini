"""Tests for reachy_mini.media.device_detection.

Uses captured ``gst-device-monitor-1.0`` output from
``tests/unit_tests/data/`` to test the parser and the pure device
detection functions on all supported platforms — without requiring
GStreamer at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from reachy_mini.media.camera_constants import (
    ArducamSpecs,
    CameraSpecs,
    ReachyMiniLiteCamSpecs,
)
from reachy_mini.media.device_detection import (
    DeviceInfo,
    find_audio_device,
    find_video_device,
    parse_gst_device_monitor_output,
)


_DATA_DIR = Path(__file__).parent / "data"


def _load(filename: str) -> str:
    """Load a test data file, handling UTF-16LE (Windows dump)."""
    path = _DATA_DIR / filename
    raw = path.read_bytes()
    # UTF-16LE files start with a BOM (FF FE).
    if raw[:2] == b"\xff\xfe":
        return raw.decode("utf-16-le")
    return raw.decode("utf-8")


def _parse(filename: str) -> List[DeviceInfo]:
    return parse_gst_device_monitor_output(_load(filename))


def _filter_class(devices: List[DeviceInfo], device_class: str) -> List[DeviceInfo]:
    """Filter devices by class, mirroring Gst.DeviceMonitor.add_filter() in production."""
    return [d for d in devices if d.device_class == device_class]


@pytest.fixture()
def pipewire_devices() -> List[DeviceInfo]:
    return _parse("gst-device-monitor-linux-pipewire.txt")


@pytest.fixture()
def pulseaudio_devices() -> List[DeviceInfo]:
    return _parse("gst-device-monitor-linux-pulseaudio.txt")


@pytest.fixture()
def macos_devices() -> List[DeviceInfo]:
    return _parse("gst-device-monitor-osx.txt")


@pytest.fixture()
def windows_devices() -> List[DeviceInfo]:
    return _parse("gst-device-monitor-windows.txt")


class TestParser:
    """Verify that the gst-device-monitor parser extracts the right devices."""

    def test_pipewire_device_count(self, pipewire_devices: List[DeviceInfo]) -> None:
        assert len(pipewire_devices) > 0

    def test_pulseaudio_device_count(
        self, pulseaudio_devices: List[DeviceInfo]
    ) -> None:
        assert len(pulseaudio_devices) > 0

    def test_macos_device_count(self, macos_devices: List[DeviceInfo]) -> None:
        assert len(macos_devices) > 0

    def test_windows_device_count(self, windows_devices: List[DeviceInfo]) -> None:
        assert len(windows_devices) > 0

    def test_pipewire_has_reachy_audio_source(
        self, pipewire_devices: List[DeviceInfo]
    ) -> None:
        names = [d.display_name for d in pipewire_devices]
        assert any("Reachy Mini Audio" in n for n in names)

    def test_pipewire_has_reachy_camera(
        self, pipewire_devices: List[DeviceInfo]
    ) -> None:
        names = [d.display_name for d in pipewire_devices]
        assert any("Reachy Mini Camera" in n for n in names)

    def test_pulseaudio_has_monitor_device(
        self, pulseaudio_devices: List[DeviceInfo]
    ) -> None:
        """The PulseAudio dump contains a 'Monitor of Reachy Mini Audio' entry."""
        monitor_devs = [
            d
            for d in pulseaudio_devices
            if d.properties.get("device.class") == "monitor"
            and "Reachy Mini Audio" in d.display_name
        ]
        assert len(monitor_devs) >= 1

    def test_parser_extracts_properties(
        self, pipewire_devices: List[DeviceInfo]
    ) -> None:
        """Each device with a properties section should have a non-empty dict."""
        audio_devs = [
            d for d in pipewire_devices if d.device_class.startswith("Audio/")
        ]
        for d in audio_devs:
            assert len(d.properties) > 0, f"{d.display_name} has no properties"

    def test_parser_preserves_index_order(
        self, pipewire_devices: List[DeviceInfo]
    ) -> None:
        for i, d in enumerate(pipewire_devices):
            assert d.index == i

    def test_windows_utf16_parsing(self, windows_devices: List[DeviceInfo]) -> None:
        """The Windows file is UTF-16LE; parsing should still work."""
        names = [d.display_name for d in windows_devices]
        assert any("Reachy Mini Audio" in n for n in names)
        assert any("Reachy Mini Camera" in n for n in names)


class TestFindAudioDevice:
    """Test find_audio_device across all platforms."""

    def test_pipewire_source(self, pipewire_devices: List[DeviceInfo]) -> None:
        devices = _filter_class(pipewire_devices, "Audio/Source")
        result = find_audio_device(devices, "Source", "Linux")
        assert result is not None
        assert "Reachy_Mini_Audio" in result or "reachy" in result.lower()
        assert result.startswith("alsa_input.")

    def test_pipewire_sink(self, pipewire_devices: List[DeviceInfo]) -> None:
        devices = _filter_class(pipewire_devices, "Audio/Sink")
        result = find_audio_device(devices, "Sink", "Linux")
        assert result is not None
        assert result.startswith("alsa_output.")

    def test_pulseaudio_source(self, pulseaudio_devices: List[DeviceInfo]) -> None:
        devices = _filter_class(pulseaudio_devices, "Audio/Source")
        result = find_audio_device(devices, "Source", "Linux")
        assert result is not None
        assert result == (
            "alsa_input."
            "usb-Pollen_Robotics_Reachy_Mini_Audio___________________-00."
            "analog-stereo"
        )

    def test_pulseaudio_sink(self, pulseaudio_devices: List[DeviceInfo]) -> None:
        devices = _filter_class(pulseaudio_devices, "Audio/Sink")
        result = find_audio_device(devices, "Sink", "Linux")
        assert result is not None
        assert result == (
            "alsa_output."
            "usb-Pollen_Robotics_Reachy_Mini_Audio___________________-00."
            "analog-stereo"
        )

    def test_pulseaudio_skips_monitor(
        self, pulseaudio_devices: List[DeviceInfo]
    ) -> None:
        """The 'Monitor of Reachy Mini Audio' device must be skipped."""
        devices = _filter_class(pulseaudio_devices, "Audio/Source")
        result = find_audio_device(devices, "Source", "Linux")
        assert result is not None
        assert "monitor" not in result.lower()

    def test_macos_source(self, macos_devices: List[DeviceInfo]) -> None:
        devices = _filter_class(macos_devices, "Audio/Source")
        result = find_audio_device(devices, "Source", "Darwin")
        assert result is not None
        assert "Pollen Robotics" in result or "Reachy Mini Audio" in result

    def test_macos_sink(self, macos_devices: List[DeviceInfo]) -> None:
        devices = _filter_class(macos_devices, "Audio/Sink")
        result = find_audio_device(devices, "Sink", "Darwin")
        assert result is not None
        assert "Pollen Robotics" in result or "Reachy Mini Audio" in result

    def test_windows_source(self, windows_devices: List[DeviceInfo]) -> None:
        devices = _filter_class(windows_devices, "Audio/Source")
        result = find_audio_device(devices, "Source", "Windows")
        assert result is not None
        assert result == "{0.0.1.00000000}.{d9fee2b6-0d5d-42b2-9785-df20fd39ac3b}"

    def test_windows_sink(self, windows_devices: List[DeviceInfo]) -> None:
        devices = _filter_class(windows_devices, "Audio/Sink")
        result = find_audio_device(devices, "Sink", "Windows")
        assert result is not None
        assert result == "{0.0.0.00000000}.{e79e8ab7-4b13-4542-bb4b-edad0943c8ee}"

    def test_windows_skips_loopback_source(
        self, windows_devices: List[DeviceInfo]
    ) -> None:
        """Loopback source with wasapi2.device.loopback=true must be skipped."""
        devices = _filter_class(windows_devices, "Audio/Source")
        result = find_audio_device(devices, "Source", "Windows")
        assert result is not None
        # The loopback device shares the Sink's device.id; the real source
        # has a different device.id.
        assert result != "{0.0.0.00000000}.{e79e8ab7-4b13-4542-bb4b-edad0943c8ee}"

    def test_windows_skips_loopback_source_uppercase_bool(self) -> None:
        """Loopback must be skipped even when the value is 'True' (uppercase).

        The live GStreamer API converts GBoolean properties via Python's
        ``str(True)`` → ``"True"``, while the text dump from
        ``gst-device-monitor`` uses lowercase ``"true"``.  Both must be
        rejected for loopback devices so the real capture device is selected.
        """
        devices = [
            DeviceInfo(
                display_name="Echo Cancelling Speakerphone (Reachy Mini Audio)",
                device_class="Audio/Source",
                properties={
                    "device.api": "wasapi2",
                    "device.id": "{0.0.0.00000000}.{e79e8ab7-4b13-4542-bb4b-edad0943c8ee}",
                    "wasapi2.device.loopback": "True",  # str(True) from live GStreamer
                },
            ),
            DeviceInfo(
                display_name="Echo Cancelling Speakerphone (Reachy Mini Audio)",
                device_class="Audio/Source",
                properties={
                    "device.api": "wasapi2",
                    "device.id": "{0.0.1.00000000}.{d9fee2b6-0d5d-42b2-9785-df20fd39ac3b}",
                    "wasapi2.device.loopback": "False",  # str(False) from live GStreamer
                },
            ),
        ]
        result = find_audio_device(devices, "Source", "Windows")
        assert result == "{0.0.1.00000000}.{d9fee2b6-0d5d-42b2-9785-df20fd39ac3b}"

    def test_empty_device_list(self) -> None:
        result = find_audio_device([], "Source", "Linux")
        assert result is None

    def test_no_matching_device(self) -> None:
        devices = [
            DeviceInfo(
                display_name="Some Other Audio",
                device_class="Audio/Source",
                properties={"node.name": "other"},
            )
        ]
        result = find_audio_device(devices, "Source", "Linux")
        assert result is None

    def test_custom_target_name(self) -> None:
        devices = [
            DeviceInfo(
                display_name="My Custom Card",
                device_class="Audio/Source",
                properties={"node.name": "custom_node"},
            )
        ]
        result = find_audio_device(
            devices, "Source", "Linux", target_name="My Custom Card"
        )
        assert result == "custom_node"


class TestFindVideoDevice:
    """Test find_video_device across all platforms."""

    def test_linux_pipewire_reachy_camera(
        self, pipewire_devices: List[DeviceInfo]
    ) -> None:
        devices = _filter_class(pipewire_devices, "Video/Source")
        path, specs = find_video_device(devices, "Linux")
        assert path == "/dev/video2"
        assert specs is not None
        assert isinstance(specs, ReachyMiniLiteCamSpecs)

    def test_linux_pipewire_priority_reachy_over_arducam(
        self, pipewire_devices: List[DeviceInfo]
    ) -> None:
        """'Reachy' has higher priority than 'Arducam_12MP'."""
        devices = _filter_class(pipewire_devices, "Video/Source")
        path, specs = find_video_device(devices, "Linux")
        # Reachy Mini Camera is at /dev/video2, Arducam_16MP at /dev/video4.
        # "Reachy" matches first so we get /dev/video2.
        assert path == "/dev/video2"

    def test_linux_pulseaudio_reachy_camera(
        self, pulseaudio_devices: List[DeviceInfo]
    ) -> None:
        devices = _filter_class(pulseaudio_devices, "Video/Source")
        path, specs = find_video_device(devices, "Linux")
        assert path == "/dev/video4"
        assert specs is not None

    def test_macos_reachy_camera(self, macos_devices: List[DeviceInfo]) -> None:
        devices = _filter_class(macos_devices, "Video/Source")
        path, specs = find_video_device(devices, "Darwin")
        # Reachy Mini Camera is at index 0 in the macOS dump.
        assert path == "0"
        assert specs is not None
        assert isinstance(specs, ReachyMiniLiteCamSpecs)

    def test_windows_reachy_camera(self, windows_devices: List[DeviceInfo]) -> None:
        devices = _filter_class(windows_devices, "Video/Source")
        path, specs = find_video_device(devices, "Windows")
        # On Windows, the display name is returned.
        assert "Reachy Mini Camera" in path
        assert specs is not None
        assert isinstance(specs, ReachyMiniLiteCamSpecs)

    def test_arducam_only(self) -> None:
        """When only an Arducam is present, it should be found with ArducamSpecs."""
        devices = [
            DeviceInfo(
                display_name="Arducam_12MP",
                device_class="Video/Source",
                properties={"api.v4l2.path": "/dev/video0"},
            )
        ]
        path, specs = find_video_device(devices, "Linux")
        assert path == "/dev/video0"
        assert isinstance(specs, ArducamSpecs)

    def test_empty_device_list(self) -> None:
        path, specs = find_video_device([], "Linux")
        assert path == ""
        assert specs is None

    def test_no_matching_camera(self) -> None:
        devices = [
            DeviceInfo(
                display_name="Some Random Webcam",
                device_class="Video/Source",
                properties={"api.v4l2.path": "/dev/video0"},
            )
        ]
        path, specs = find_video_device(devices, "Linux")
        assert path == ""
        assert specs is None

    def test_imx708_rpi_camera(self) -> None:
        """RPi CSI camera (imx708) returns the literal string 'imx708'."""
        devices = [
            DeviceInfo(
                display_name="imx708 camera",
                device_class="Video/Source",
                properties={},  # No V4L2 path for CSI cameras
            )
        ]
        path, specs = find_video_device(devices, "Linux")
        assert path == "imx708"
        assert specs is not None
