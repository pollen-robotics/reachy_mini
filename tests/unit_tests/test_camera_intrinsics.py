"""Tests for camera intrinsics at streamed resolutions."""

import numpy as np
import pytest

from reachy_mini.media.camera_constants import (
    ArducamSpecs,
    CameraSpecs,
    GenericWebcamSpecs,
    MujocoCameraSpecs,
    ReachyMiniLiteCamSpecs,
)
from reachy_mini.media.camera_utils import intrinsics_for_size


@pytest.mark.parametrize(
    ("specs_type", "calibration_size"),
    [
        (ArducamSpecs, (1280, 720)),
        (GenericWebcamSpecs, (1280, 720)),
        (MujocoCameraSpecs, (1280, 720)),
        (ReachyMiniLiteCamSpecs, (3840, 2592)),
    ],
)
def test_camera_specs_keep_the_size_their_intrinsics_use(
    specs_type: type[CameraSpecs], calibration_size: tuple[int, int]
) -> None:
    """Instances preserve each camera matrix's calibration image size."""
    assert specs_type().calibration_size == calibration_size


def test_mujoco_intrinsics_match_the_rendered_camera_fov() -> None:
    """MuJoCo's 80-degree vertical FOV produces square-pixel intrinsics."""
    expected_focal_length = 720 / (2 * np.tan(np.deg2rad(80) / 2))
    np.testing.assert_allclose(MujocoCameraSpecs.K[0, 0], expected_focal_length)
    np.testing.assert_allclose(MujocoCameraSpecs.K[1, 1], expected_focal_length)


@pytest.mark.parametrize(
    ("specs_type", "expected_focal_length"),
    [
        (
            MujocoCameraSpecs,
            720 / (2 * np.tan(np.deg2rad(80) / 2)) * 320 / 1280,
        ),
        (GenericWebcamSpecs, 640 * 320 / 1280),
    ],
)
def test_sim_and_webcam_intrinsics_scale_from_720p(
    specs_type: type[CameraSpecs], expected_focal_length: float
) -> None:
    """Scaling 720p camera matrices to 320x180 preserves the frame center."""
    specs = specs_type()
    scaled = intrinsics_for_size(
        specs.K,
        specs.default_resolution.value[3],
        (320, 180),
        specs.calibration_size,
    )

    np.testing.assert_allclose(scaled[0, 0], expected_focal_length)
    np.testing.assert_allclose(scaled[1, 1], expected_focal_length)
    np.testing.assert_allclose(scaled[:2, 2], (160, 90))


def test_reachy_camera_intrinsics_still_scale_from_sensor_size() -> None:
    """Physical Reachy camera matrices continue scaling from the full sensor."""
    specs = ReachyMiniLiteCamSpecs()
    scaled = intrinsics_for_size(specs.K, 1.0, (320, 180), specs.calibration_size)

    np.testing.assert_allclose(scaled[0, 2], specs.K[0, 2] * 320 / 3840)
    np.testing.assert_allclose(scaled[1, 2], specs.K[1, 2] * 180 / 2592)
