"""Tests for the YuNet face detector."""

import numpy as np
import pytest

pytest.importorskip("cv2")

from reachy_mini.vision.face_detector import Face, FaceDetector


def test_detect_returns_empty_on_blank_frame() -> None:
    """A frame with no faces yields no detections."""
    detector = FaceDetector()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    assert detector.detect(blank) == []


class _FakeYuNet:
    """Stub of cv2.FaceDetectorYN returning a fixed detection array."""

    def __init__(self, faces: np.ndarray) -> None:
        self._faces = faces

    def setInputSize(self, size: tuple[int, int]) -> None:
        pass

    def detect(self, frame: np.ndarray) -> tuple[int, np.ndarray]:
        return 1, self._faces


def test_detect_maps_yunet_columns_to_face() -> None:
    """Detections map YuNet's [bbox, eyes, ...] columns onto Face fields."""
    detector = FaceDetector()
    detector._detector = _FakeYuNet(np.arange(15, dtype=np.float32).reshape(1, 15))

    faces = detector.detect(np.zeros((10, 10, 3), dtype=np.uint8))

    assert faces == [
        Face(bbox=(0.0, 1.0, 2.0, 3.0), right_eye=(4.0, 5.0), left_eye=(6.0, 7.0))
    ]
