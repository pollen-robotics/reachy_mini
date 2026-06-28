"""Tests for out-of-process face-tracking selection and observation reduction."""

from types import SimpleNamespace

import numpy as np

from reachy_mini.vision.face_tracking import Tracker, to_observation


def _face(
    bbox: tuple[float, float, float, float],
    right_eye: tuple[float, float],
    left_eye: tuple[float, float],
) -> SimpleNamespace:
    return SimpleNamespace(bbox=bbox, right_eye=right_eye, left_eye=left_eye)


def test_to_observation_no_face_reports_undetected() -> None:
    """No face yields an undetected observation that still carries frame metadata."""
    obs = to_observation(None, 640, 480, np.eye(3), np.zeros(5), 2.0)
    assert obs.eye_center is None
    assert obs.roll is None
    assert obs.timestamp == 2.0


def test_to_observation_normalizes_center_and_roll() -> None:
    """Eye center is normalized to [-1, 1] and roll is the signed eye angle."""
    obs = to_observation(
        _face((0.0, 0.0, 2.0, 2.0), (0.0, 0.0), (2.0, 2.0)),
        3,
        3,
        np.eye(3),
        np.zeros(5),
        1.0,
    )
    assert obs.eye_center == (0.0, 0.0)
    assert obs.roll == float(np.arctan2(2.0, 2.0))


def test_tracker_acquires_largest_above_min_size() -> None:
    """A fresh track picks the largest face once it clears the size gate."""
    tracker = Tracker(min_area_frac=0.0)
    big = _face((0.0, 0.0, 100.0, 100.0), (10.0, 10.0), (90.0, 10.0))
    small = _face((0.0, 0.0, 10.0, 10.0), (1.0, 1.0), (9.0, 1.0))
    assert tracker.select([big, small], 200, 200) is big


def test_tracker_rejects_specks_on_acquisition() -> None:
    """A too-small detection is ignored so the head won't lock onto distant noise."""
    tracker = Tracker(min_area_frac=0.5)
    speck = _face((0.0, 0.0, 10.0, 10.0), (1.0, 1.0), (9.0, 1.0))
    assert tracker.select([speck], 200, 200) is None


def test_tracker_sticks_to_track_and_rejects_far_jump() -> None:
    """Once tracking, the nearest face wins and a stray far detection is dropped."""
    tracker = Tracker(min_area_frac=0.0, max_jump=0.3)
    here = _face((90.0, 90.0, 20.0, 20.0), (95.0, 95.0), (105.0, 95.0))
    far = _face((0.0, 0.0, 20.0, 20.0), (5.0, 5.0), (15.0, 5.0))
    assert tracker.select([here], 200, 200) is here
    assert tracker.select([here, far], 200, 200) is here
    assert tracker.select([far], 200, 200) is None


def test_tracker_drops_track_after_misses_then_reacquires() -> None:
    """A sustained run of misses drops the track so a new face can be acquired."""
    tracker = Tracker(min_area_frac=0.0, max_jump=0.1, max_misses=1)
    here = _face((90.0, 90.0, 20.0, 20.0), (95.0, 95.0), (105.0, 95.0))
    far = _face((0.0, 0.0, 20.0, 20.0), (5.0, 5.0), (15.0, 5.0))
    assert tracker.select([here], 200, 200) is here
    assert tracker.select([far], 200, 200) is None  # miss 1, track held
    assert tracker.select([far], 200, 200) is None  # miss 2, track dropped
    assert tracker.select([far], 200, 200) is far  # re-acquired
