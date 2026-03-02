"""Tests for the volume control module."""

import platform
from unittest.mock import patch

import pytest

from reachy_mini.daemon.app.routers.volume_control import VolumeControl, create_volume_control

_LINUX_BACKENDS = ["pulsectl", "alsa"] if platform.system() == "Linux" else [None]


@pytest.fixture(params=_LINUX_BACKENDS)
def volume_control(request):
    """Create a VolumeControl instance.

    On Linux the fixture is parametrized so every test runs against
    both the pulsectl and ALSA backends.
    """
    backend = request.param

    if platform.system() == "Linux" and backend is not None:
        force_pulsectl = backend == "pulsectl"
        with patch(
            "reachy_mini.daemon.app.routers.volume_control_linux._PULSECTL_AVAILABLE",
            force_pulsectl,
        ):
            yield create_volume_control()
            return

    yield create_volume_control()


# ---- Factory tests ----

@pytest.mark.audio
def test_factory_returns_correct_subclass(volume_control):
    """The factory should return the platform-specific VolumeControl subclass."""
    system = platform.system()

    assert isinstance(volume_control, VolumeControl)

    if system == "Darwin":
        from reachy_mini.daemon.app.routers.volume_control_macos import VolumeControlMacOS

        assert isinstance(volume_control, VolumeControlMacOS)
    elif system == "Linux":
        from reachy_mini.daemon.app.routers.volume_control_linux import VolumeControlLinux

        assert isinstance(volume_control, VolumeControlLinux)
    elif system == "Windows":
        from reachy_mini.daemon.app.routers.volume_control_windows import VolumeControlWindows

        assert isinstance(volume_control, VolumeControlWindows)
    else:
        pytest.fail(f"Unexpected platform: {system}")

@pytest.mark.audio
def test_factory_raises_on_unsupported_platform():
    """The factory should raise RuntimeError on an unsupported platform."""
    with patch("reachy_mini.daemon.app.routers.volume_control.platform") as mock_platform:
        mock_platform.system.return_value = "FreeBSD"
        with pytest.raises(RuntimeError, match="Unsupported platform"):
            create_volume_control()


# ---- Get volume tests ----


@pytest.mark.audio
def test_get_output_volume(volume_control):
    """Getting output volume should return an int between 0 and 100."""
    volume = volume_control.get_output_volume()
    assert isinstance(volume, int)
    assert 0 <= volume <= 100


@pytest.mark.audio
def test_get_input_volume(volume_control):
    """Getting input volume should return an int between 0 and 100."""
    volume = volume_control.get_input_volume()
    assert isinstance(volume, int)
    assert 0 <= volume <= 100


# ---- Set volume tests ----


@pytest.mark.audio
def test_set_and_restore_output_volume(volume_control):
    """Setting output volume should apply the value, then restore the original."""
    original = volume_control.get_output_volume()

    target = 50 if original != 50 else 30
    result = volume_control.set_output_volume(target)
    assert result is True

    current = volume_control.get_output_volume()
    assert abs(current - target) <= 5  # allow small rounding tolerance

    # Restore original volume
    volume_control.set_output_volume(original)


@pytest.mark.audio
def test_set_and_restore_input_volume(volume_control):
    """Setting input volume should apply the value, then restore the original."""
    original = volume_control.get_input_volume()

    target = 50 if original != 50 else 30
    result = volume_control.set_input_volume(target)
    assert result is True

    current = volume_control.get_input_volume()
    assert abs(current - target) <= 5  # allow small rounding tolerance

    # Restore original volume
    volume_control.set_input_volume(original)


@pytest.mark.audio
def test_set_output_volume_clamps(volume_control):
    """Volume values outside [0, 100] should be clamped, not rejected."""
    original = volume_control.get_output_volume()

    assert volume_control.set_output_volume(0) is True
    assert volume_control.get_output_volume() == pytest.approx(0, abs=5)

    assert volume_control.set_output_volume(100) is True
    assert volume_control.get_output_volume() == pytest.approx(100, abs=5)

    # Restore original volume
    volume_control.set_output_volume(original)
