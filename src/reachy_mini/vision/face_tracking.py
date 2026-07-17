"""Face tracking: a YuNet detector thread feeding the daemon the latest observation."""

import logging
import os
import platform
import queue
import threading
import time
from dataclasses import dataclass

import gi
import numpy as np
from numpy.typing import NDArray

from reachy_mini.daemon.utils import CAMERA_PIPE_NAME, CAMERA_SOCKET_PATH
from reachy_mini.media.camera_constants import CameraSpecs
from reachy_mini.media.camera_utils import intrinsics_for_size
from reachy_mini.vision.face_detector import Face, FaceDetector

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
# GstApp is unused directly but installs appsink.try_pull_sample().
from gi.repository import Gst, GstApp  # noqa: E402, F401

logger = logging.getLogger(__name__)

# Detector input width; smaller trades recall for CPU.
_TRACKING_WIDTH = 320


@dataclass
class FaceObservation:
    """A face target in ``set_tracking_face`` form (face center normalized to [-1, 1])."""

    center: tuple[float, float] | None
    roll: float | None
    width: int
    height: int
    camera_matrix: NDArray[np.float64]
    distortion: NDArray[np.float64]
    timestamp: float


class _AdaptiveCenterFilter:
    """Internal face-center smoother with fixed tracking parameters."""

    _ALPHA = 0.3
    _FAST_ALPHA = 0.6
    _MOVEMENT_THRESHOLD = 0.15
    _DEAD_ZONE = 0.02

    def __init__(self) -> None:
        self._value: NDArray[np.float64] | None = None
        self._previous_input: NDArray[np.float64] | None = None

    def update(self, center: tuple[float, float]) -> tuple[float, float]:
        """Consume one raw center and return the filtered center."""
        current = np.asarray(center, dtype=np.float64)
        if self._value is None or self._previous_input is None:
            self._value = current.copy()
            self._previous_input = current.copy()
            return center

        movement = float(np.linalg.norm(current - self._previous_input))
        self._previous_input = current.copy()
        delta = current - self._value
        if float(np.linalg.norm(delta)) < self._DEAD_ZONE:
            return (float(self._value[0]), float(self._value[1]))

        alpha = self._FAST_ALPHA if movement > self._MOVEMENT_THRESHOLD else self._ALPHA
        self._value += alpha * delta
        return (float(self._value[0]), float(self._value[1]))

    def reset(self) -> None:
        """Forget filter history so the next observation is accepted immediately."""
        self._value = None
        self._previous_input = None


def _area(face: Face) -> float:
    return face.bbox[2] * face.bbox[3]


def _center(face: Face, width: int, height: int) -> tuple[float, float]:
    # Aim at the nose, because centering on the eye midpoint makes the robot look slightly above.
    return (
        face.nose[0] / max(width - 1, 1) * 2 - 1,
        face.nose[1] / max(height - 1, 1) * 2 - 1,
    )


def _dist2(a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


class Tracker:
    """Track one face: acquire largest, associate nearest, drop after misses."""

    def __init__(
        self,
        min_area_frac: float = 0.003,
        max_jump: float = 0.5,
        max_misses: int = 20,
    ) -> None:
        """Create a tracker with the given selection gates."""
        self._min_area_frac = min_area_frac
        self._max_jump = max_jump
        self._max_misses = max_misses
        self._center: tuple[float, float] | None = None
        self._misses = 0

    def select(self, faces: list[Face], width: int, height: int) -> Face | None:
        """Pick the face to aim at, or None when no plausible target is present."""
        if not faces:
            self._miss()
            return None
        if self._center is None:
            face = max(faces, key=_area)
            if _area(face) < self._min_area_frac * width * height:
                self._miss()
                return None
        else:
            center = self._center
            face = min(faces, key=lambda f: _dist2(_center(f, width, height), center))
            if _dist2(_center(face, width, height), center) > self._max_jump**2:
                self._miss()
                return None
        self._center = _center(face, width, height)
        self._misses = 0
        return face

    def _miss(self) -> None:
        self._misses += 1
        if self._misses > self._max_misses:
            self._center = None

    @property
    def has_target(self) -> bool:
        """Whether the tracker is still associated with a face."""
        return self._center is not None


def to_observation(
    face: Face | None,
    width: int,
    height: int,
    camera_matrix: NDArray[np.float64],
    distortion: NDArray[np.float64],
    timestamp: float,
) -> FaceObservation:
    """Reduce the tracked face (or its absence) to one normalized observation."""
    if face is None:
        return FaceObservation(
            None, None, width, height, camera_matrix, distortion, timestamp
        )
    roll = float(
        np.arctan2(
            face.left_eye[1] - face.right_eye[1],
            face.left_eye[0] - face.right_eye[0],
        )
    )
    return FaceObservation(
        _center(face, width, height),
        roll,
        width,
        height,
        camera_matrix,
        distortion,
        timestamp,
    )


class FaceTracker:
    """Run the face detector in a daemon thread and expose the latest observation."""

    def __init__(self) -> None:
        """Initialize the tracker; no thread is started until ``start``."""
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._active = threading.Event()
        self._observations: queue.SimpleQueue[FaceObservation] = queue.SimpleQueue()
        self._selector = Tracker()
        self._center_filter = _AdaptiveCenterFilter()

    def start(self, camera_specs: CameraSpecs) -> None:
        """Start the detector thread if it is not already running."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        # Drop observations left over from a previous run.
        self._observations = queue.SimpleQueue()
        self._selector = Tracker()
        self._center_filter.reset()
        self._thread = threading.Thread(
            target=self._run, args=(camera_specs,), daemon=True, name="face-tracker"
        )
        self._thread.start()

    def set_active(self, active: bool) -> None:
        """Pause or resume detection; a paused tracker disconnects from the camera feed."""
        if active:
            self._active.set()
        else:
            self._active.clear()

    def latest(self) -> FaceObservation | None:
        """Return the most recent observation, draining any backlog."""
        obs: FaceObservation | None = None
        while not self._observations.empty():
            obs = self._observations.get_nowait()
        return obs

    def stop(self) -> None:
        """Stop the detector thread."""
        self._stop.set()
        if self._thread is None:
            return
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            # Keep the handle: forgetting a live thread lets start() clear its stop flag and double-run.
            logger.warning("Face tracker thread did not stop in time.")
        else:
            self._thread = None

    def _process_detections(
        self,
        faces: list[Face],
        width: int,
        height: int,
        camera_matrix: NDArray[np.float64],
        distortion: NDArray[np.float64],
        timestamp: float,
    ) -> None:
        """Select, filter, and emit one observation from a detection frame."""
        face = self._selector.select(faces, width, height)
        if face is None and not self._selector.has_target:
            self._center_filter.reset()
        observation = to_observation(
            face, width, height, camera_matrix, distortion, timestamp
        )
        if observation.center is not None:
            observation.center = self._center_filter.update(observation.center)
        self._observations.put(observation)

    def _run(self, camera_specs: CameraSpecs) -> None:
        # Lowest priority (Linux-only per-thread nice) so the detector yields CPU to the rest of the daemon.
        if platform.system() == "Linux":
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 19)
        Gst.init([])
        windows = platform.system() == "Windows"
        source = Gst.ElementFactory.make("win32ipcvideosrc" if windows else "unixfdsrc")
        queue_frames = Gst.ElementFactory.make("queue")
        # Prefer v4l2convert: on the RPi the ISP does the scale + convert in hardware.
        convert_chain = [Gst.ElementFactory.make("v4l2convert")]
        if convert_chain[0] is None:
            convert_chain = [
                Gst.ElementFactory.make("videoscale"),
                Gst.ElementFactory.make("videoconvert"),
            ]
        capsfilter = Gst.ElementFactory.make("capsfilter")
        appsink = Gst.ElementFactory.make("appsink")
        chain = [source, queue_frames, *convert_chain, capsfilter, appsink]
        if any(element is None for element in chain):
            logger.warning("Face tracking unavailable: missing GStreamer plugins.")
            return
        if windows:
            source.set_property("pipe-name", CAMERA_PIPE_NAME)
        else:
            source.set_property("socket-path", CAMERA_SOCKET_PATH)
        queue_frames.set_property("leaky", 2)
        queue_frames.set_property("max-size-buffers", 1)
        src_w, src_h = camera_specs.default_resolution.value[:2]
        width = min(_TRACKING_WIDTH, src_w)
        height = max(2, round(width * src_h / src_w / 2) * 2)
        capsfilter.set_property(
            "caps",
            Gst.Caps.from_string(
                f"video/x-raw,format=BGR,width={width},height={height}"
            ),
        )
        appsink.set_property("drop", True)
        appsink.set_property("max-buffers", 1)
        appsink.set_property("sync", False)

        pipeline = Gst.Pipeline.new("face-tracker")
        for element in chain:
            pipeline.add(element)
        for upstream, downstream in zip(chain, chain[1:]):
            if not upstream.link(downstream):
                logger.warning(
                    "Face tracking unavailable: could not link %s to %s.",
                    upstream.get_name(),
                    downstream.get_name(),
                )
                return

        crop_scale = camera_specs.default_resolution.value[3]
        camera_matrix: NDArray[np.float64] | None = None
        playing = False
        feed_lost = False
        try:
            detector = FaceDetector()
            while not self._stop.is_set():
                if not self._active.is_set():
                    self._center_filter.reset()
                    if playing:
                        # Disconnect while paused so the daemon serves nothing to this client.
                        pipeline.set_state(Gst.State.NULL)
                        playing = False
                    self._active.wait(0.2)
                    continue
                if not playing:
                    if (
                        pipeline.set_state(Gst.State.PLAYING)
                        == Gst.StateChangeReturn.FAILURE
                    ):
                        if not feed_lost:
                            feed_lost = True
                            logger.warning(
                                "Face tracker cannot reach the camera feed; retrying."
                            )
                        # A stopped appsink returns instantly, so back off instead of busy-polling it.
                        pipeline.set_state(Gst.State.NULL)
                        self._stop.wait(1.0)
                        continue
                    feed_lost = False
                    playing = True
                sample = appsink.try_pull_sample(200 * Gst.MSECOND)
                if sample is None:
                    continue
                structure = sample.get_caps().get_structure(0)
                frame_width = structure.get_value("width")
                frame_height = structure.get_value("height")
                buf = sample.get_buffer()
                frame = np.frombuffer(
                    buf.extract_dup(0, buf.get_size()), dtype=np.uint8
                ).reshape((frame_height, frame_width, 3))
                if camera_matrix is None:
                    camera_matrix = intrinsics_for_size(
                        camera_specs.K, crop_scale, (frame_width, frame_height)
                    )
                self._process_detections(
                    detector.detect(frame),
                    frame_width,
                    frame_height,
                    camera_matrix,
                    camera_specs.D,
                    time.monotonic(),
                )
        except Exception:
            # With no process boundary left, this is the only place a detector crash gets reported.
            logger.exception("Face tracker crashed.")
        finally:
            pipeline.set_state(Gst.State.NULL)
