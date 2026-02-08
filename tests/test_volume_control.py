import platform
from unittest.mock import patch

import pytest

from reachy_mini.daemon.app.routers.volume_control import VolumeControl, create_volume_control


def test_factory_returns_correct_subclass():
    """The factory should return the platform-specific VolumeControl subclass."""
    system = platform.system()
    vc = create_volume_control()

    assert isinstance(vc, VolumeControl)

    if system == "Darwin":
        from reachy_mini.daemon.app.routers.volume_control_macos import VolumeControlMacOS
        assert isinstance(vc, VolumeControlMacOS)
    elif system == "Linux":
        pytest.fail(f"Linux volume control is not implemented yet")
    elif system == "Windows":
        pytest.fail(f"Windows volume control is not implemented yet")
    else:
        pytest.fail(f"Unexpected platform: {system}")


def test_factory_raises_on_unsupported_platform():
    """The factory should raise RuntimeError on an unsupported platform."""
    with patch("reachy_mini.daemon.app.routers.volume_control.platform") as mock_platform:
        mock_platform.system.return_value = "FreeBSD"
        with pytest.raises(RuntimeError, match="Unsupported platform"):
            create_volume_control()


def test_get_output_volume():
    """Getting output volume should return a float between 0.0 and 1.0."""
    vc = create_volume_control()
    volume = vc.get_output_volume()
    assert isinstance(volume, float)
    assert 0.0 <= volume <= 1.0


def test_get_input_volume():
    """Getting input volume should return a float between 0.0 and 1.0."""
    vc = create_volume_control()
    volume = vc.get_input_volume()
    assert isinstance(volume, float)
    assert 0.0 <= volume <= 1.0


def test_set_and_restore_output_volume():
    """Setting output volume should apply the value, then restore the original."""
    vc = create_volume_control()
    original = vc.get_output_volume()

    target = 0.5 if original != 0.5 else 0.3
    result = vc.set_output_volume(target)
    assert result is True

    current = vc.get_output_volume()
    assert abs(current - target) < 0.05  # allow small rounding tolerance

    # Restore original volume
    vc.set_output_volume(original)


def test_set_and_restore_input_volume():
    """Setting input volume should apply the value, then restore the original."""
    vc = create_volume_control()
    original = vc.get_input_volume()

    target = 0.5 if original != 0.5 else 0.3
    result = vc.set_input_volume(target)
    assert result is True

    current = vc.get_input_volume()
    assert abs(current - target) < 0.05  # allow small rounding tolerance

    # Restore original volume
    vc.set_input_volume(original)


def test_set_output_volume_clamps():
    """Volume values outside [0, 1] should be clamped, not rejected."""
    vc = create_volume_control()
    original = vc.get_output_volume()

    assert vc.set_output_volume(0.0) is True
    assert vc.get_output_volume() == pytest.approx(0.0, abs=0.05)

    assert vc.set_output_volume(1.0) is True
    assert vc.get_output_volume() == pytest.approx(1.0, abs=0.05)

    # Restore original volume
    vc.set_output_volume(original)
