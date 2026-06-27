"""Tests for out-of-process face-tracking observation reduction."""

from types import SimpleNamespace

import numpy as np

from reachy_mini.vision.face_tracking import observe


def _face(
    bbox: tuple[float, float, float, float],
    right_eye: tuple[float, float],
    left_eye: tuple[float, float],
) -> SimpleNamespace:
    return SimpleNamespace(bbox=bbox, right_eye=right_eye, left_eye=left_eye)


def test_observe_no_faces_reports_undetected() -> None:
    """With no faces the observation is undetected but carries frame metadata."""
    obs = observe([], 640, 480, np.eye(3), np.zeros(5), 2.0)
    assert obs.eye_center is None
    assert obs.roll is None
    assert obs.timestamp == 2.0


def test_observe_picks_largest_face_and_normalizes() -> None:
    """The largest face wins; eye center is normalized to [-1, 1] and roll signed."""
    far = _face((0.0, 0.0, 10.0, 10.0), (9.0, 9.0), (9.0, 9.0))
    near = _face((0.0, 0.0, 100.0, 100.0), (0.0, 0.0), (2.0, 2.0))

    obs = observe([far, near], 3, 3, np.eye(3), np.zeros(5), 1.0)

    assert obs.eye_center == (0.0, 0.0)
    assert obs.roll == float(np.arctan2(2.0, 2.0))
