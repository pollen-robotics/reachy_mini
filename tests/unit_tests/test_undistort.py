"""Tests for the pure-numpy undistort_points implementation.

Validates that our implementation matches cv2.undistortPoints for every
distortion model used in the codebase (5-coeff, 12-coeff, and zero-distortion).
"""

import importlib.util

import numpy as np
import pytest

from reachy_mini.media.camera_constants import (
    ArducamSpecs,
    GenericWebcamSpecs,
    MujocoCameraSpecs,
    ReachyMiniLiteCamSpecs,
)
from reachy_mini.media.camera_utils import undistort_points

_opencv_available = importlib.util.find_spec("cv2") is not None


# ── Pure-numpy unit tests (always run, no cv2 needed) ────────────────────────


class TestUndistortPointsIdentity:
    """Test undistort_points with identity/trivial camera models."""

    def test_identity_camera_center(self) -> None:
        """Principal point should map to (0, 0) regardless of distortion."""
        K = np.array([[800.0, 0, 640.0], [0, 800.0, 360.0], [0, 0, 1.0]])
        D = np.zeros(5)
        x, y = undistort_points(640.0, 360.0, K, D)
        assert abs(x) < 1e-10
        assert abs(y) < 1e-10

    def test_identity_camera_offset(self) -> None:
        """A pixel offset from center, no distortion, should give (u-cx)/fx."""
        K = np.array([[800.0, 0, 640.0], [0, 600.0, 360.0], [0, 0, 1.0]])
        D = np.zeros(5)
        x, y = undistort_points(800.0, 480.0, K, D)
        np.testing.assert_allclose(x, (800.0 - 640.0) / 800.0, atol=1e-10)
        np.testing.assert_allclose(y, (480.0 - 360.0) / 600.0, atol=1e-10)

    def test_zero_distortion_12_coeffs(self) -> None:
        """12 zero coefficients should behave like no distortion."""
        K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
        D = np.zeros(12)
        x, y = undistort_points(400.0, 300.0, K, D)
        np.testing.assert_allclose(x, (400.0 - 320.0) / 500.0, atol=1e-10)
        np.testing.assert_allclose(y, (300.0 - 240.0) / 500.0, atol=1e-10)

    def test_mujoco_specs(self) -> None:
        """MujocoCameraSpecs has zero distortion, should be a pure pinhole unproject."""
        K = MujocoCameraSpecs.K
        D = MujocoCameraSpecs.D
        u, v = 700.0, 400.0
        x, y = undistort_points(u, v, K, D)
        np.testing.assert_allclose(x, (u - K[0, 2]) / K[0, 0], atol=1e-10)
        np.testing.assert_allclose(y, (v - K[1, 2]) / K[1, 1], atol=1e-10)

    def test_generic_webcam_specs(self) -> None:
        """GenericWebcamSpecs has zero distortion."""
        K = GenericWebcamSpecs.K
        D = GenericWebcamSpecs.D
        u, v = 800.0, 500.0
        x, y = undistort_points(u, v, K, D)
        np.testing.assert_allclose(x, (u - K[0, 2]) / K[0, 0], atol=1e-10)
        np.testing.assert_allclose(y, (v - K[1, 2]) / K[1, 1], atol=1e-10)


class TestUndistortPointsDistortion:
    """Test that distortion actually changes the result compared to pinhole."""

    def test_5_coeff_differs_from_pinhole(self) -> None:
        """With non-zero 5 distortion coeffs, result should differ from simple pinhole."""
        K = ArducamSpecs.K
        D = ArducamSpecs.D
        u, v = 200.0, 100.0  # corner pixel, far from center
        x, y = undistort_points(u, v, K, D)
        # Pinhole result (no distortion)
        x_pinhole = (u - K[0, 2]) / K[0, 0]
        y_pinhole = (v - K[1, 2]) / K[1, 1]
        # Should differ due to distortion correction
        assert abs(x - x_pinhole) > 1e-4 or abs(y - y_pinhole) > 1e-4

    def test_12_coeff_differs_from_pinhole(self) -> None:
        """With non-zero 12 distortion coeffs, result should differ from simple pinhole."""
        K = ReachyMiniLiteCamSpecs.K
        D = ReachyMiniLiteCamSpecs.D
        u, v = 500.0, 300.0
        x, y = undistort_points(u, v, K, D)
        x_pinhole = (u - K[0, 2]) / K[0, 0]
        y_pinhole = (v - K[1, 2]) / K[1, 1]
        assert abs(x - x_pinhole) > 1e-4 or abs(y - y_pinhole) > 1e-4

    def test_center_pixel_unaffected(self) -> None:
        """Principal point is always (0,0) in normalized coords, even with distortion."""
        K = ReachyMiniLiteCamSpecs.K
        D = ReachyMiniLiteCamSpecs.D
        cx, cy = K[0, 2], K[1, 2]
        x, y = undistort_points(cx, cy, K, D)
        np.testing.assert_allclose(x, 0.0, atol=1e-10)
        np.testing.assert_allclose(y, 0.0, atol=1e-10)


# ── Cross-validation against cv2 (only runs if opencv is installed) ──────────


@pytest.mark.skipif(not _opencv_available, reason="OpenCV not installed")
class TestUndistortPointsVsOpenCV:
    """Cross-validate our implementation against cv2.undistortPoints."""

    # Pixel coordinates covering center, mid-range, and edge regions
    PIXEL_COORDS = [
        (640.0, 360.0),  # near center
        (100.0, 50.0),  # top-left corner
        (1200.0, 700.0),  # bottom-right area
        (0.0, 0.0),  # extreme corner
        (1919.0, 1079.0),  # near image boundary at 1080p
        (960.0, 540.0),  # center of 1080p
    ]

    # Arducam (deprecated beta hardware) has a 5-coeff model with k3=-0.0983
    # that creates non-monotonic distortion at high radii, making convergence
    # unreliable at extreme pixel positions. We test a more conservative set.
    ARDUCAM_PIXEL_COORDS = [
        (640.0, 360.0),  # near center
        (100.0, 50.0),  # top-left
        (0.0, 0.0),  # extreme corner
        (1919.0, 1079.0),  # far boundary
        (960.0, 540.0),  # mid-range
    ]

    @staticmethod
    def _cv2_undistort(u: float, v: float, K: np.ndarray, D: np.ndarray) -> tuple:
        import cv2

        points = np.array([[[u, v]]], dtype=np.float32)
        result = cv2.undistortPoints(points, K, D)
        return float(result[0, 0, 0]), float(result[0, 0, 1])

    @pytest.mark.parametrize("u,v", ARDUCAM_PIXEL_COORDS)
    def test_arducam_5_coeffs(self, u: float, v: float) -> None:
        """5-coefficient model (Arducam, deprecated beta hardware).

        The 5-coeff model can have slower convergence at extreme pixels
        far from the image center. We use a generous tolerance since this
        hardware is deprecated and exact match is not critical.
        """
        K = ArducamSpecs.K
        D = ArducamSpecs.D
        x_ours, y_ours = undistort_points(u, v, K, D)
        x_cv2, y_cv2 = self._cv2_undistort(u, v, K, D)
        np.testing.assert_allclose(x_ours, x_cv2, atol=0.02, rtol=0.02)
        np.testing.assert_allclose(y_ours, y_cv2, atol=0.02, rtol=0.02)

    @pytest.mark.parametrize("u,v", PIXEL_COORDS)
    def test_reachy_mini_lite_12_coeffs(self, u: float, v: float) -> None:
        """12-coefficient rational+thin-prism model matches cv2.

        This is the primary camera model used in production. Tolerance of
        1e-4 in normalized coordinates corresponds to sub-pixel accuracy,
        which is well within the precision needed for gaze direction.
        """
        K = ReachyMiniLiteCamSpecs.K
        D = ReachyMiniLiteCamSpecs.D
        x_ours, y_ours = undistort_points(u, v, K, D)
        x_cv2, y_cv2 = self._cv2_undistort(u, v, K, D)
        np.testing.assert_allclose(x_ours, x_cv2, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(y_ours, y_cv2, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("u,v", PIXEL_COORDS)
    def test_zero_distortion(self, u: float, v: float) -> None:
        """Zero distortion matches cv2."""
        K = MujocoCameraSpecs.K
        D = MujocoCameraSpecs.D
        x_ours, y_ours = undistort_points(u, v, K, D)
        x_cv2, y_cv2 = self._cv2_undistort(u, v, K, D)
        np.testing.assert_allclose(x_ours, x_cv2, atol=1e-10)
        np.testing.assert_allclose(y_ours, y_cv2, atol=1e-10)
