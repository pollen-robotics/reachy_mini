"""Tests for face-tracking selection, observation reduction, and thread lifecycle."""

import threading
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from reachy_mini.vision.face_tracking import FaceTracker, Tracker, to_observation

if TYPE_CHECKING:
    from reachy_mini.media.camera_constants import CameraSpecs


def _face(
    bbox: tuple[float, float, float, float],
    right_eye: tuple[float, float],
    left_eye: tuple[float, float],
    nose: tuple[float, float],
) -> SimpleNamespace:
    return SimpleNamespace(bbox=bbox, right_eye=right_eye, left_eye=left_eye, nose=nose)


def test_to_observation_no_face_reports_undetected() -> None:
    """No face yields an undetected observation that still carries frame metadata."""
    obs = to_observation(None, 640, 480, np.eye(3), np.zeros(5), 2.0)
    assert obs.center is None
    assert obs.roll is None
    assert obs.timestamp == 2.0


def test_to_observation_normalizes_nose_and_roll() -> None:
    """The aim point is the normalized nose and roll is the signed eye angle."""
    obs = to_observation(
        _face((0.0, 0.0, 2.0, 2.0), (0.0, 0.0), (2.0, 2.0), (1.0, 1.0)),
        3,
        3,
        np.eye(3),
        np.zeros(5),
        1.0,
    )
    assert obs.center == (0.0, 0.0)
    assert obs.roll == float(np.arctan2(2.0, 2.0))


def test_face_tracker_filters_emitted_detection_sequence() -> None:
    """FaceTracker applies its slow, fast, and dead-zone paths to emitted observations."""
    face_tracker = FaceTracker()
    camera_matrix = np.eye(3)
    distortion = np.zeros(5)

    def emit(nose_x: float, timestamp: float) -> tuple[float, float] | None:
        face = _face(
            (nose_x - 5.0, 45.0, 10.0, 10.0),
            (nose_x - 2.0, 48.0),
            (nose_x + 2.0, 48.0),
            (nose_x, 50.0),
        )
        face_tracker._process_detections(
            [face], 101, 101, camera_matrix, distortion, timestamp
        )
        observation = face_tracker.latest()
        assert observation is not None
        return observation.center

    assert emit(50.0, 1.0) == (0.0, 0.0)
    assert emit(50.5, 2.0) == (0.0, 0.0)  # radial dead zone
    assert emit(55.0, 3.0) == pytest.approx((0.03, 0.0))  # slow EMA
    assert emit(75.0, 4.0) == pytest.approx((0.312, 0.0))  # fast EMA


def test_face_tracker_resets_filter_after_target_loss() -> None:
    """A reacquired face is emitted immediately instead of blending with stale aim."""
    face_tracker = FaceTracker()
    camera_matrix = np.eye(3)
    distortion = np.zeros(5)
    old_face = _face(
        (45.0, 45.0, 10.0, 10.0),
        (48.0, 48.0),
        (52.0, 48.0),
        (50.0, 50.0),
    )
    new_face = _face(
        (5.0, 45.0, 10.0, 10.0),
        (8.0, 48.0),
        (12.0, 48.0),
        (10.0, 50.0),
    )
    face_tracker._process_detections(
        [old_face], 101, 101, camera_matrix, distortion, 1.0
    )
    for timestamp in range(2, 23):
        face_tracker._process_detections(
            [], 101, 101, camera_matrix, distortion, float(timestamp)
        )
    face_tracker._process_detections(
        [new_face], 101, 101, camera_matrix, distortion, 23.0
    )

    observation = face_tracker.latest()
    assert observation is not None
    assert observation.center == pytest.approx((-0.8, 0.0))


def test_tracker_acquires_largest_above_min_size() -> None:
    """A fresh track picks the largest face once it clears the size gate."""
    tracker = Tracker(min_area_frac=0.0)
    big = _face((0.0, 0.0, 100.0, 100.0), (10.0, 10.0), (90.0, 10.0), (50.0, 40.0))
    small = _face((0.0, 0.0, 10.0, 10.0), (1.0, 1.0), (9.0, 1.0), (5.0, 4.0))
    assert tracker.select([big, small], 200, 200) is big


def test_tracker_rejects_specks_on_acquisition() -> None:
    """A too-small detection is ignored so the head won't lock onto distant noise."""
    tracker = Tracker(min_area_frac=0.5)
    speck = _face((0.0, 0.0, 10.0, 10.0), (1.0, 1.0), (9.0, 1.0), (5.0, 4.0))
    assert tracker.select([speck], 200, 200) is None


def test_tracker_sticks_to_track_and_rejects_far_jump() -> None:
    """Once tracking, the nearest face wins and a stray far detection is dropped."""
    tracker = Tracker(min_area_frac=0.0, max_jump=0.3)
    here = _face((90.0, 90.0, 20.0, 20.0), (95.0, 95.0), (105.0, 95.0), (100.0, 100.0))
    far = _face((0.0, 0.0, 20.0, 20.0), (5.0, 5.0), (15.0, 5.0), (10.0, 10.0))
    assert tracker.select([here], 200, 200) is here
    assert tracker.select([here, far], 200, 200) is here
    assert tracker.select([far], 200, 200) is None


def test_tracker_drops_track_after_misses_then_reacquires() -> None:
    """A sustained run of misses drops the track so a new face can be acquired."""
    tracker = Tracker(min_area_frac=0.0, max_jump=0.1, max_misses=1)
    here = _face((90.0, 90.0, 20.0, 20.0), (95.0, 95.0), (105.0, 95.0), (100.0, 100.0))
    far = _face((0.0, 0.0, 20.0, 20.0), (5.0, 5.0), (15.0, 5.0), (10.0, 10.0))
    assert tracker.select([here], 200, 200) is here
    assert tracker.select([far], 200, 200) is None  # miss 1, track held
    assert tracker.select([far], 200, 200) is None  # miss 2, track dropped
    assert tracker.select([far], 200, 200) is far  # re-acquired


def test_stuck_stop_never_double_spawns(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stop() that times out keeps the thread handle so restarts stay single."""
    release = threading.Event()
    runs: list[threading.Thread] = []

    def fake_run(self: FaceTracker, camera_specs: "CameraSpecs") -> None:
        runs.append(threading.current_thread())
        release.wait(10.0)

    monkeypatch.setattr(FaceTracker, "_run", fake_run)
    specs = cast("CameraSpecs", None)
    tracker = FaceTracker()
    tracker.start(specs)
    tracker.start(specs)
    assert len(runs) == 1  # start on a running tracker is a no-op

    tracker.stop()  # the stub ignores the stop event, so the join times out
    tracker.start(specs)
    assert len(runs) == 1  # the live thread is kept; no second detector

    release.set()
    runs[0].join(timeout=5.0)
    assert not runs[0].is_alive()
    tracker.start(specs)
    assert len(runs) == 2  # once the thread has exited, restart works
    tracker.stop()
