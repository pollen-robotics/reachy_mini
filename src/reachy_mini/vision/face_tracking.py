"""Out-of-process face tracking: a YuNet detector feeding the daemon over a pipe."""

import logging
import multiprocessing as mp
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


def observe(
    faces: "list[Face]",
    width: int,
    height: int,
    camera_matrix: NDArray[np.float64],
    distortion: NDArray[np.float64],
    timestamp: float,
) -> FaceObservation:
    """Reduce detected faces to the nearest (largest) one as a normalized observation."""
    if not faces:
        return FaceObservation(
            None, None, width, height, camera_matrix, distortion, timestamp
        )
    face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
    cx = (face.right_eye[0] + face.left_eye[0]) / 2
    cy = (face.right_eye[1] + face.left_eye[1]) / 2
    eye_center = (
        cx / max(width - 1, 1) * 2 - 1,
        cy / max(height - 1, 1) * 2 - 1,
    )
    roll = float(
        np.arctan2(
            face.left_eye[1] - face.right_eye[1],
            face.left_eye[0] - face.right_eye[0],
        )
    )
    return FaceObservation(
        eye_center, roll, width, height, camera_matrix, distortion, timestamp
    )


def run(
    conn: Connection,
    stop: EventType,
    camera_specs: "CameraSpecs",
    log_level: str = "INFO",
) -> None:
    """Read camera frames, detect faces, and stream observations until stopped."""
    try:
        from reachy_mini.media.camera_gstreamer import GStreamerCamera
        from reachy_mini.media.camera_utils import intrinsics_for_size
        from reachy_mini.vision.face_detector import FaceDetector

        camera = GStreamerCamera(log_level=log_level, camera_specs=camera_specs)
        camera.open()
        detector = FaceDetector()
    except Exception as e:
        _logger.warning("Face tracker failed to start: %s", e)
        return

    crop_scale = camera_specs.default_resolution.value[3]
    camera_matrix: NDArray[np.float64] | None = None
    try:
        while not stop.is_set():
            frame = camera.read()
            if frame is None:
                continue
            height, width = frame.shape[0], frame.shape[1]
            if camera_matrix is None:
                camera_matrix = intrinsics_for_size(
                    camera_specs.K, crop_scale, (width, height)
                )
            obs = observe(
                detector.detect(frame),
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
        camera.close()


class FaceTrackerProcess:
    """Spawn and manage the out-of-process face detector."""

    def __init__(self, log_level: str = "INFO") -> None:
        """Initialize the manager; no process is started until ``start``."""
        self._ctx = mp.get_context("spawn")
        self._log_level = log_level
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
            args=(send_conn, self._stop, camera_specs, self._log_level),
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
