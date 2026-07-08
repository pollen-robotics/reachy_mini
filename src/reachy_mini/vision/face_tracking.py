"""Out-of-process face tracking: a YuNet detector feeding the daemon over a pipe."""

import logging
import multiprocessing as mp
import os
import platform
import signal
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

# Detector input budget: width and fps trade recall and freshness against CPU.
_TRACKING_WIDTH = 320
_TRACKING_FPS = 10


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


def _area(face: "Face") -> float:
    return face.bbox[2] * face.bbox[3]


def _center(face: "Face", width: int, height: int) -> tuple[float, float]:
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
        _center(face, width, height),
        roll,
        width,
        height,
        camera_matrix,
        distortion,
        timestamp,
    )


def run(
    conn: Connection, stop: EventType, active: EventType, camera_specs: "CameraSpecs"
) -> None:
    """Detect faces on the camera IPC feed and pipe observations until stopped."""
    # The daemon coordinates shutdown via `stop`; ignore Ctrl+C so it doesn't traceback here.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # Lowest priority so YuNet bursts never preempt the daemon's 50 Hz control loop.
    if hasattr(os, "nice"):
        os.nice(19)
    try:
        import gi

        gi.require_version("Gst", "1.0")
        gi.require_version("GstApp", "1.0")
        # GstApp is unused directly but installs appsink.try_pull_sample().
        from gi.repository import Gst, GstApp  # noqa: F401

        from reachy_mini.daemon.utils import CAMERA_PIPE_NAME, CAMERA_SOCKET_PATH
        from reachy_mini.media.camera_utils import intrinsics_for_size
        from reachy_mini.vision.face_detector import FaceDetector

        Gst.init([])
        if platform.system() == "Windows":
            source = Gst.ElementFactory.make("win32ipcvideosrc")
            source.set_property("pipe-name", CAMERA_PIPE_NAME)
        else:
            source = Gst.ElementFactory.make("unixfdsrc")
            source.set_property("socket-path", CAMERA_SOCKET_PATH)
        queue = Gst.ElementFactory.make("queue")
        queue.set_property("leaky", 2)
        queue.set_property("max-size-buffers", 1)
        # Drop to the detection rate before converting so skipped frames cost nothing.
        videorate = Gst.ElementFactory.make("videorate")
        videorate.set_property("drop-only", True)
        videorate.set_property("max-rate", _TRACKING_FPS)
        # Prefer v4l2convert: on the RPi the ISP does the scale + convert in hardware.
        convert_chain = [Gst.ElementFactory.make("v4l2convert")]
        if convert_chain[0] is None:
            convert_chain = [
                Gst.ElementFactory.make("videoscale"),
                Gst.ElementFactory.make("videoconvert"),
            ]
        src_w, src_h = camera_specs.default_resolution.value[:2]
        width = min(_TRACKING_WIDTH, src_w)
        height = max(2, round(width * src_h / src_w / 2) * 2)
        capsfilter = Gst.ElementFactory.make("capsfilter")
        capsfilter.set_property(
            "caps",
            Gst.Caps.from_string(
                f"video/x-raw,format=BGR,width={width},height={height}"
            ),
        )
        appsink = Gst.ElementFactory.make("appsink")
        appsink.set_property("drop", True)
        appsink.set_property("max-buffers", 1)
        appsink.set_property("sync", False)

        pipeline = Gst.Pipeline.new("face-tracker")
        chain = [source, queue, videorate, *convert_chain, capsfilter, appsink]
        for element in chain:
            pipeline.add(element)
        for upstream, downstream in zip(chain, chain[1:]):
            upstream.link(downstream)
        detector = FaceDetector()
        tracker = Tracker()
    except Exception as e:
        _logger.warning("Face tracker failed to start: %s", e)
        return

    crop_scale = camera_specs.default_resolution.value[3]
    camera_matrix: NDArray[np.float64] | None = None
    playing = False
    try:
        while not stop.is_set():
            if not active.is_set():
                if playing:
                    # Disconnect while paused so the daemon serves nothing to this client.
                    pipeline.set_state(Gst.State.NULL)
                    playing = False
                active.wait(0.2)
                continue
            if not playing:
                pipeline.set_state(Gst.State.PLAYING)
                playing = True
            sample = appsink.try_pull_sample(200 * Gst.MSECOND)
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
        self._active: EventType | None = None

    def start(self, camera_specs: "CameraSpecs") -> None:
        """Spawn the detector process if it is not already running."""
        if self._proc is not None and self._proc.is_alive():
            return
        recv_conn, send_conn = self._ctx.Pipe(duplex=False)
        self._stop = self._ctx.Event()
        self._active = self._ctx.Event()
        self._conn = recv_conn
        self._proc = self._ctx.Process(
            target=run,
            args=(send_conn, self._stop, self._active, camera_specs),
            daemon=True,
        )
        self._proc.start()
        send_conn.close()

    def set_active(self, active: bool) -> None:
        """Pause or resume detection; a paused worker disconnects from the camera feed."""
        if self._active is None:
            return
        if active:
            self._active.set()
        else:
            self._active.clear()

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
