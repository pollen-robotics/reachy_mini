"""Out-of-process face tracking: a YuNet detector feeding the daemon over a pipe."""

import logging
import multiprocessing as mp
import os
import platform
import time
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from multiprocessing.synchronize import Event as EventType
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from reachy_mini.media.camera_constants import CameraSpecs
    from reachy_mini.vision.face_detector import Face

_logger = logging.getLogger("reachymini-face-tracker")


@dataclass
class FaceObservation:
    """A face target in ``set_tracking_face`` form (eye center normalized to [-1, 1])."""

    eye_center: tuple[float, float] | None
    roll: float | None
    width: int
    height: int
    camera_matrix: NDArray[np.float64]
    distortion: NDArray[np.float64]
    timestamp: float


def _area(face: "Face") -> float:
    return face.bbox[2] * face.bbox[3]


def _eye_center(face: "Face", width: int, height: int) -> tuple[float, float]:
    cx = (face.right_eye[0] + face.left_eye[0]) / 2
    cy = (face.right_eye[1] + face.left_eye[1]) / 2
    return (cx / max(width - 1, 1) * 2 - 1, cy / max(height - 1, 1) * 2 - 1)


def _dist2(a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


class Tracker:
    """Greedy single-face track that rejects spurious detections.

    Acquires the largest face above a minimum size, then sticks to the face
    nearest the current track, so a stray detection elsewhere can't yank the
    head around. Drops the track after a run of misses so a new person can be
    acquired.
    """

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

    def select(self, faces: "list[Face]", width: int, height: int) -> "Face | None":
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
            face = min(
                faces, key=lambda f: _dist2(_eye_center(f, width, height), center)
            )
            if _dist2(_eye_center(face, width, height), center) > self._max_jump**2:
                self._miss()
                return None
        self._center = _eye_center(face, width, height)
        self._misses = 0
        return face

    def _miss(self) -> None:
        self._misses += 1
        if self._misses > self._max_misses:
            self._center = None


def to_observation(
    face: "Face | None",
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
        _eye_center(face, width, height),
        roll,
        width,
        height,
        camera_matrix,
        distortion,
        timestamp,
    )


def run(conn: Connection, stop: EventType, camera_specs: "CameraSpecs") -> None:
    """Read the tracker camera stream, detect faces, and stream observations until stopped.

    The daemon serves an already downscaled + rate-limited BGR stream on the tracking
    socket, so this only detects and reports normalized eye centers — no resize or pacing.
    """
    # Lowest priority so YuNet bursts never preempt the daemon's 50 Hz control loop.
    if hasattr(os, "nice"):
        os.nice(19)
    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        from reachy_mini.daemon.utils import (
            CAMERA_TRACKING_PIPE_NAME,
            CAMERA_TRACKING_SOCKET_PATH,
        )
        from reachy_mini.media.camera_utils import intrinsics_for_size
        from reachy_mini.vision.face_detector import FaceDetector

        Gst.init([])
        if platform.system() == "Windows":
            source = f"win32ipcvideosrc pipe-name={CAMERA_TRACKING_PIPE_NAME}"
        else:
            source = f"unixfdsrc socket-path={CAMERA_TRACKING_SOCKET_PATH}"
        pipeline = Gst.parse_launch(
            f"{source} ! queue leaky=2 max-size-buffers=1 ! "
            "appsink name=sink drop=true max-buffers=1 sync=false"
        )
        pipeline.set_state(Gst.State.PLAYING)
        appsink = pipeline.get_by_name("sink")
        detector = FaceDetector()
        tracker = Tracker()
    except Exception as e:
        _logger.warning("Face tracker failed to start: %s", e)
        return

    crop_scale = camera_specs.default_resolution.value[3]
    camera_matrix: NDArray[np.float64] | None = None
    try:
        while not stop.is_set():
            sample = appsink.emit("try-pull-sample", 200_000_000)
            if sample is None:
                continue
            structure = sample.get_caps().get_structure(0)
            width = structure.get_value("width")
            height = structure.get_value("height")
            buf = sample.get_buffer()
            frame = np.frombuffer(
                buf.extract_dup(0, buf.get_size()), dtype=np.uint8
            ).reshape((height, width, 3))
            if camera_matrix is None:
                camera_matrix = intrinsics_for_size(
                    camera_specs.K, crop_scale, (width, height)
                )
            face = tracker.select(detector.detect(frame), width, height)
            obs = to_observation(
                face,
                width,
                height,
                camera_matrix,
                camera_specs.D,
                time.monotonic(),
            )
            try:
                conn.send(obs)
            except (BrokenPipeError, OSError):
                break
    finally:
        pipeline.set_state(Gst.State.NULL)


class FaceTrackerProcess:
    """Spawn and manage the out-of-process face detector."""

    def __init__(self) -> None:
        """Initialize the manager; no process is started until ``start``."""
        self._ctx = mp.get_context("spawn")
        self._proc: BaseProcess | None = None
        self._conn: Connection | None = None
        self._stop: EventType | None = None

    def start(self, camera_specs: "CameraSpecs") -> None:
        """Spawn the detector process if it is not already running."""
        if self._proc is not None and self._proc.is_alive():
            return
        recv_conn, send_conn = self._ctx.Pipe(duplex=False)
        self._stop = self._ctx.Event()
        self._conn = recv_conn
        self._proc = self._ctx.Process(
            target=run,
            args=(send_conn, self._stop, camera_specs),
            daemon=True,
        )
        self._proc.start()
        send_conn.close()

    def latest(self) -> FaceObservation | None:
        """Return the most recent observation, draining any backlog."""
        if self._conn is None:
            return None
        obs: FaceObservation | None = None
        try:
            while self._conn.poll():
                obs = self._conn.recv()
        except (EOFError, OSError):
            return obs
        return obs

    def stop(self) -> None:
        """Stop the detector process and release the pipe."""
        if self._stop is not None:
            self._stop.set()
        if self._proc is not None:
            self._proc.join(timeout=2.0)
            if self._proc.is_alive():
                self._proc.terminate()
        if self._conn is not None:
            self._conn.close()
        self._proc = None
        self._conn = None
        self._stop = None
