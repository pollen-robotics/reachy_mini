"""Tests for face-tracking selection, target publication, and process lifecycle."""

import multiprocessing
import threading
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from reachy_mini.media.camera_constants import ReachyMiniLiteCamSpecs
from reachy_mini.vision import face_tracking
from reachy_mini.vision.face_tracking import (
    FaceTracker,
    Tracker,
    TrackingTarget,
    _DetectionWorker,
)

if TYPE_CHECKING:
    from multiprocessing.sharedctypes import SynchronizedArray

    from reachy_mini.media.camera_constants import CameraSpecs
    from reachy_mini.vision.face_tracking import _EventLike


def _face(
    bbox: tuple[float, float, float, float],
    right_eye: tuple[float, float],
    left_eye: tuple[float, float],
    nose: tuple[float, float],
) -> SimpleNamespace:
    return SimpleNamespace(bbox=bbox, right_eye=right_eye, left_eye=left_eye, nose=nose)


def _make_worker() -> tuple[
    _DetectionWorker, "SynchronizedArray[float]", "SynchronizedArray[float]"
]:
    target_mailbox = multiprocessing.Array("d", face_tracking._TARGET_SLOTS)
    pose_mailbox = multiprocessing.Array("d", face_tracking._POSE_SLOTS)
    worker = _DetectionWorker(
        ReachyMiniLiteCamSpecs(),
        target_mailbox,
        pose_mailbox,
        threading.Event(),
        threading.Event(),
    )
    return worker, target_mailbox, pose_mailbox


def _read_target(mailbox: "SynchronizedArray[float]") -> list[float]:
    with mailbox.get_lock():
        return list(mailbox)


# Principal point at (50, 50): a nose there looks straight along the camera axis.
_K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
_D = np.zeros(5)
_CENTERED_FACE = _face(
    (45.0, 45.0, 10.0, 10.0), (48.0, 48.0), (52.0, 48.0), (50.0, 50.0)
)


def test_worker_publishes_absolute_angles_with_pitch_trim() -> None:
    """A centered face with an identity head pose yields yaw 0 and the pitch trim.

    The camera axis coincides with the head's forward axis for an identity
    pose, so the only pitch in the published target is the deliberate
    look-lower trim.
    """
    worker, target_mailbox, pose_mailbox = _make_worker()
    with pose_mailbox.get_lock():
        pose_mailbox[0] = 1.0  # valid, angles all zero (identity head pose)

    worker._process_detections([_CENTERED_FACE], 101, 101, _K, _D)

    values = _read_target(target_mailbox)
    assert values[0] == 1.0  # seq
    assert values[1] == 1.0  # detected
    assert values[2] == pytest.approx(0.0, abs=1e-6)  # roll
    assert values[3] == pytest.approx(face_tracking._PITCH_OFFSET_RAD, abs=1e-6)
    assert values[4] == pytest.approx(0.0, abs=1e-6)  # yaw
    assert values[5] == pytest.approx(0.0, abs=1e-9)  # x_norm
    assert values[6] == pytest.approx(0.0, abs=1e-9)  # y_norm


def test_worker_without_head_pose_reports_a_miss() -> None:
    """Before the daemon publishes its head pose, a face cannot become a target."""
    worker, target_mailbox, _ = _make_worker()

    worker._process_detections([_CENTERED_FACE], 101, 101, _K, _D)

    values = _read_target(target_mailbox)
    assert values[0] == 1.0  # the publication still happened
    assert values[1] == 0.0  # but as a miss


def test_worker_publishes_miss_and_increments_seq_without_faces() -> None:
    """Every processed frame publishes, so the daemon can distinguish fresh misses."""
    worker, target_mailbox, pose_mailbox = _make_worker()
    with pose_mailbox.get_lock():
        pose_mailbox[0] = 1.0

    worker._process_detections([_CENTERED_FACE], 101, 101, _K, _D)
    worker._process_detections([], 101, 101, _K, _D)

    values = _read_target(target_mailbox)
    assert values[0] == 2.0
    assert values[1] == 0.0


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


def _run_worker_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    fake_find: "object | None" = None,
) -> dict[str, object]:
    """Build the worker pipeline with a fake detector; return created elements."""
    pipeline_ready = threading.Event()
    created: dict[str, object] = {}
    element_factory = face_tracking.Gst.ElementFactory
    original_make = element_factory.make

    def fake_make(factory_name: str, name: str | None = None) -> object:
        if factory_name == "v4l2convert" and fake_find is not None:
            raise RuntimeError("No such element: v4l2convert")
        element = original_make(factory_name, name)
        created.setdefault(factory_name, element)
        return element

    class FakeFaceDetector:
        """Signal that the worker pipeline was built successfully."""

        def __init__(self) -> None:
            pipeline_ready.set()

    if fake_find is not None:
        monkeypatch.setattr(element_factory, "find", staticmethod(fake_find))
    monkeypatch.setattr(element_factory, "make", staticmethod(fake_make))
    monkeypatch.setattr(face_tracking, "FaceDetector", FakeFaceDetector)

    worker, _, _ = _make_worker()
    stop = threading.Event()
    worker._stop = stop
    thread = threading.Thread(target=worker.run, daemon=True)
    thread.start()
    try:
        assert pipeline_ready.wait(5.0)
    finally:
        stop.set()
        thread.join(5.0)
    assert not thread.is_alive()
    return created


def test_worker_falls_back_when_v4l2convert_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing Linux converter falls back to the portable software chain."""
    element_factory = face_tracking.Gst.ElementFactory
    original_find = element_factory.find

    def fake_find(factory_name: str) -> object | None:
        if factory_name == "v4l2convert":
            return None
        return original_find(factory_name)

    created = _run_worker_pipeline(monkeypatch, fake_find=fake_find)
    assert "v4l2convert" not in created
    assert "videoscale" in created
    assert "videoconvert" in created


def test_rate_gate_caps_a_fast_feed_and_passes_a_slow_one() -> None:
    """The gate settles a 30 Hz feed at the cap and passes a slower feed through."""
    gate = face_tracking._RateGate(10.0)
    accepted = [t for t in np.arange(0.0, 2.0, 1 / 30) if gate.ready(float(t))]
    rate = (len(accepted) - 1) / (accepted[-1] - accepted[0])
    assert rate == pytest.approx(10.0, rel=0.05)

    gate = face_tracking._RateGate(10.0)
    slow = [t for t in np.arange(0.0, 2.0, 1 / 5) if gate.ready(float(t))]
    assert len(slow) == 10  # every 5 Hz frame passes


def test_rate_gate_tolerates_jitter_at_the_cap_rate() -> None:
    """Feed jitter at exactly the cap rate must not drop every other frame."""
    gate = face_tracking._RateGate(10.0)
    rng = np.random.default_rng(0)
    times = np.cumsum(0.1 + rng.uniform(-0.005, 0.005, 100))
    accepted = [t for t in times if gate.ready(float(t))]
    assert len(accepted) >= 97


def _sleepy_worker(
    camera_specs: "CameraSpecs",
    target_mailbox: "SynchronizedArray[float]",
    pose_mailbox: "SynchronizedArray[float]",
    active: "_EventLike",
    stop: "_EventLike",
) -> None:
    """Stub detector process: hold until asked to stop."""
    stop.wait(30.0)


def _emit_one_worker(
    camera_specs: "CameraSpecs",
    target_mailbox: "SynchronizedArray[float]",
    pose_mailbox: "SynchronizedArray[float]",
    active: "_EventLike",
    stop: "_EventLike",
) -> None:
    """Stub detector process: publish one target, then hold."""
    with target_mailbox.get_lock():
        target_mailbox[0] = 1.0  # seq
        target_mailbox[1] = 1.0  # detected
        target_mailbox[2] = 0.1  # roll
        target_mailbox[3] = 0.2  # pitch
        target_mailbox[4] = 0.3  # yaw
        target_mailbox[5] = 0.25  # x
        target_mailbox[6] = -0.5  # y
        target_mailbox[7] = 0.05  # face roll
    stop.wait(30.0)


def test_start_is_idempotent_and_stop_reaps_the_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """start() on a live tracker is a no-op and stop() reliably reaps the child."""
    monkeypatch.setattr(face_tracking, "_worker_entry", _sleepy_worker)
    specs = cast("CameraSpecs", None)
    tracker = FaceTracker()
    tracker.start(specs)
    first = tracker._process
    assert first is not None and first.is_alive()
    tracker.start(specs)
    assert tracker._process is first  # no second detector

    tracker.stop()
    assert tracker._process is None
    assert not first.is_alive()

    tracker.start(specs)  # a stopped tracker can be restarted
    assert tracker._process is not None and tracker._process.is_alive()
    tracker.stop()


def test_latest_returns_target_from_detector_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A target published by the child round-trips through the mailbox intact."""
    monkeypatch.setattr(face_tracking, "_worker_entry", _emit_one_worker)
    tracker = FaceTracker()
    tracker.start(cast("CameraSpecs", None))
    try:
        deadline = time.monotonic() + 15.0
        target: TrackingTarget | None = None
        while target is None and time.monotonic() < deadline:
            target = tracker.latest()
            time.sleep(0.05)
        assert target is not None
        assert target.detected is True
        assert (target.roll, target.pitch, target.yaw) == (0.1, 0.2, 0.3)
        assert (target.x, target.y, target.face_roll) == (0.25, -0.5, 0.05)
    finally:
        tracker.stop()


def test_publish_head_pose_reaches_the_pose_mailbox() -> None:
    """The daemon's head angles land in the mailbox the worker reads."""
    tracker = FaceTracker()
    tracker.publish_head_pose(0.1, -0.2, 0.3)
    with tracker._pose_mailbox.get_lock():
        assert list(tracker._pose_mailbox) == [1.0, 0.1, -0.2, 0.3]


def test_start_hides_stale_target_from_a_previous_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """start() zeroes the mailboxes so a previous run's target is not replayed."""
    monkeypatch.setattr(face_tracking, "_worker_entry", _sleepy_worker)
    tracker = FaceTracker()
    with tracker._target_mailbox.get_lock():
        tracker._target_mailbox[0] = 7.0
        tracker._target_mailbox[1] = 1.0
    assert tracker.latest() is not None

    tracker.start(cast("CameraSpecs", None))
    try:
        assert tracker.latest() is None
    finally:
        tracker.stop()
