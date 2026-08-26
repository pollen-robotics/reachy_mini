"""Face tracking: a detector process publishing absolute head angles to aim at.

Detection runs in a separate *process*, not a thread: YuNet's pre/post
processing holds the GIL in chunks that measurably stall the daemon's 50 Hz
control loop when they share an interpreter. The exchange with the daemon is
two fixed-size "latest value" mailboxes, deliberately without timestamps or
queues:

- daemon -> detector: the current head orientation (roll, pitch, yaw), so the
  detector can turn a face pixel into *absolute* target angles in the robot
  frame;
- detector -> daemon: "a face is at these absolute head angles" (plus the
  normalized face position as telemetry for ``get_tracked_face``).

The daemon treats the latest published target as where the person is *now* and
servos toward it with a small bounded step per control tick. Re-applying a
stale absolute target converges toward it and stops; this is what makes the
no-timestamp design safe (a stale *delta* would instead be integrated over and
over and overshoot).
"""

import logging
import math
import multiprocessing
import os
import platform
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import gi
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from reachy_mini.daemon.utils import CAMERA_PIPE_NAME, CAMERA_SOCKET_PATH
from reachy_mini.media.camera_constants import CameraSpecs
from reachy_mini.media.camera_utils import intrinsics_for_size
from reachy_mini.vision.face_detector import Face, FaceDetector
from reachy_mini.vision.look_at import look_at_image_pose

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
# GstApp is unused directly but installs appsink.try_pull_sample().
from gi.repository import Gst, GstApp  # noqa: E402, F401

if TYPE_CHECKING:
    from multiprocessing.process import BaseProcess
    from multiprocessing.sharedctypes import SynchronizedArray

logger = logging.getLogger(__name__)

# Detector input width; smaller trades recall for CPU.
_TRACKING_WIDTH = 320

# The detector's own detection-rate cap. The local IPC feed is itself capped at
# 10 FPS today, so this is redundant until the feed rate becomes configurable
# (https://github.com/pollen-robotics/reachy_mini/issues/1263); at that point
# this constant should read the configured rate instead of being hardcoded.
_DETECTION_FPS = 10

# Gaze trim: added to the target pitch so the robot looks at the face rather
# than above it. The raw look-at consistently aims high (the geometry ignores
# the camera's offset from the head center, and the nose target sits above the
# perceived face center), so positive values pitch the gaze DOWN. Tune on
# hardware.
_PITCH_OFFSET_RAD = float(np.radians(15.0))

# Target mailbox layout (detector -> daemon), all doubles:
# [seq, detected, roll, pitch, yaw, x_norm, y_norm, face_roll]
_TARGET_SLOTS = 8
# Head-pose mailbox layout (daemon -> detector): [valid, roll, pitch, yaw]
_POSE_SLOTS = 4


class _RateGate:
    """Time-based frame gate allowing at most ``fps`` detections per second.

    The cap must live here, in the consumer loop, not in a GStreamer
    ``videorate``: buffers cross the unixfd IPC boundary with PTS 0 (verified
    on wireless hardware), so any timestamp-based dropping passes the first
    frame and then drops every one after it, silently disabling tracking.
    """

    def __init__(self, fps: float) -> None:
        self._interval = 1.0 / fps
        self._next: float | None = None

    def ready(self, now: float) -> bool:
        """Whether a frame arriving at ``now`` should be processed."""
        if self._next is None:
            self._next = now + self._interval
            return True
        # Quarter-interval tolerance so feed jitter at exactly the cap rate
        # does not drop every other frame.
        if now < self._next - 0.25 * self._interval:
            return False
        # Advance by whole intervals so a fast feed settles at the cap and a
        # slow feed passes straight through.
        self._next = max(self._next + self._interval, now)
        return True


@dataclass(frozen=True)
class TrackingTarget:
    """One published aim target: absolute head angles plus face telemetry.

    ``seq`` increments on every processed frame; the daemon uses it to tell a
    fresh publication from the previous one. Angles are radians, extrinsic
    x-y-z (roll, pitch, yaw) in the robot frame. ``x``/``y``/``face_roll`` are
    telemetry for ``get_tracked_face`` (normalized face center in [-1, 1] and
    the eye-line roll), None when no face is detected.
    """

    seq: int
    detected: bool
    roll: float
    pitch: float
    yaw: float
    x: float | None
    y: float | None
    face_roll: float | None


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


class _EventLike(Protocol):
    """The Event subset shared by threading and multiprocessing events."""

    def is_set(self) -> bool:
        """Whether the event is set."""
        ...

    def wait(self, timeout: float | None = None) -> bool:
        """Wait up to ``timeout`` for the event; return whether it is set."""
        ...


class _DetectionWorker:
    """Detector-side pipeline, detection, selection, and target publication.

    Runs in the detector process in production; unit tests run it in a thread
    (the shared arrays and events work identically in-process), which is why
    the events are duck-typed.
    """

    def __init__(
        self,
        camera_specs: CameraSpecs,
        target_mailbox: "SynchronizedArray[float]",
        pose_mailbox: "SynchronizedArray[float]",
        active: _EventLike,
        stop: _EventLike,
    ) -> None:
        """Wire the worker to its camera specs, mailboxes, and control events."""
        self._camera_specs = camera_specs
        self._target_mailbox = target_mailbox
        self._pose_mailbox = pose_mailbox
        self._active = active
        self._stop = stop
        self._selector = Tracker()
        self._seq = 0

    def _current_head_pose(self) -> NDArray[np.float64] | None:
        """Rebuild the daemon's last published head pose, or None if none yet."""
        with self._pose_mailbox.get_lock():
            valid = self._pose_mailbox[0]
            roll, pitch, yaw = (
                self._pose_mailbox[1],
                self._pose_mailbox[2],
                self._pose_mailbox[3],
            )
        if valid == 0.0:
            return None
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
        return pose

    def _publish(
        self,
        detected: bool,
        rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
        x: float = math.nan,
        y: float = math.nan,
        face_roll: float = math.nan,
    ) -> None:
        """Publish one target to the mailbox (latest value wins)."""
        self._seq += 1
        with self._target_mailbox.get_lock():
            self._target_mailbox[0] = float(self._seq)
            self._target_mailbox[1] = 1.0 if detected else 0.0
            self._target_mailbox[2] = rpy[0]
            self._target_mailbox[3] = rpy[1]
            self._target_mailbox[4] = rpy[2]
            self._target_mailbox[5] = x
            self._target_mailbox[6] = y
            self._target_mailbox[7] = face_roll

    def _process_detections(
        self,
        faces: list[Face],
        width: int,
        height: int,
        camera_matrix: NDArray[np.float64],
        distortion: NDArray[np.float64],
    ) -> None:
        """Select a face and publish the absolute head angles that aim at it."""
        face = self._selector.select(faces, width, height)
        if face is None:
            self._publish(detected=False)
            return
        head_pose = self._current_head_pose()
        if head_pose is None:
            # The daemon has not published its head pose yet; without it a
            # pixel cannot become an absolute angle, so report a miss.
            self._publish(detected=False)
            return
        target_pose = look_at_image_pose(
            u=face.nose[0],
            v=face.nose[1],
            K=camera_matrix,
            D=distortion,
            T_world_head=head_pose,
        )
        rpy = Rotation.from_matrix(target_pose[:3, :3]).as_euler("xyz")
        rpy[1] += _PITCH_OFFSET_RAD
        face_roll = float(
            np.arctan2(
                face.left_eye[1] - face.right_eye[1],
                face.left_eye[0] - face.right_eye[0],
            )
        )
        x, y = _center(face, width, height)
        self._publish(
            detected=True,
            rpy=(float(rpy[0]), float(rpy[1]), float(rpy[2])),
            x=x,
            y=y,
            face_roll=face_roll,
        )

    def run(self) -> None:
        """Consume the camera feed and publish aim targets until stopped."""
        Gst.init([])
        windows = platform.system() == "Windows"
        source = Gst.ElementFactory.make("win32ipcvideosrc" if windows else "unixfdsrc")
        queue_frames = Gst.ElementFactory.make("queue")
        # Prefer v4l2convert: on the RPi the ISP does the scale + convert in hardware.
        # Probe first because the bundled bindings raise when an optional factory is missing.
        if Gst.ElementFactory.find("v4l2convert") is not None:
            convert_chain = [Gst.ElementFactory.make("v4l2convert")]
        else:
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
        src_w, src_h = self._camera_specs.default_resolution.value[:2]
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

        crop_scale = self._camera_specs.default_resolution.value[3]
        camera_matrix: NDArray[np.float64] | None = None
        bus = pipeline.get_bus()
        rate_gate = _RateGate(_DETECTION_FPS)
        playing = False
        feed_lost = False
        try:
            detector = FaceDetector()
            while not self._stop.is_set():
                if not self._active.is_set():
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
                    # A pipeline error (lost feed, failed negotiation) leaves the
                    # appsink returning None forever; surface it and reconnect
                    # instead of dying silently.
                    message = bus.timed_pop_filtered(0, Gst.MessageType.ERROR)
                    if message is not None:
                        error, _ = message.parse_error()
                        logger.warning(
                            "Face tracking pipeline error: %s; reconnecting.",
                            error.message,
                        )
                        pipeline.set_state(Gst.State.NULL)
                        playing = False
                        self._stop.wait(1.0)
                    continue
                # Cap the detection rate before the frame copy and inference.
                # The conversion upstream still runs per frame, but on the RPi
                # that is offloaded to the ISP; the CPU cost lives below here.
                if not rate_gate.ready(time.monotonic()):
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
                        self._camera_specs.K, crop_scale, (frame_width, frame_height)
                    )
                self._process_detections(
                    detector.detect(frame),
                    frame_width,
                    frame_height,
                    camera_matrix,
                    self._camera_specs.D,
                )
        except Exception:
            # Logged in the detector process (stderr -> journal); the parent
            # additionally notices the dead process in FaceTracker.latest().
            logger.exception("Face tracker crashed.")
        finally:
            pipeline.set_state(Gst.State.NULL)


def _worker_entry(
    camera_specs: CameraSpecs,
    target_mailbox: "SynchronizedArray[float]",
    pose_mailbox: "SynchronizedArray[float]",
    active: _EventLike,
    stop: _EventLike,
) -> None:
    """Run the detection worker; entry point of the detector process."""
    logging.basicConfig(level=logging.INFO)
    if hasattr(os, "nice"):
        try:
            # The whole detector process yields to the daemon's control loop.
            os.nice(19)
        except OSError:
            pass
    _DetectionWorker(camera_specs, target_mailbox, pose_mailbox, active, stop).run()


class FaceTracker:
    """Run the face detector in a child process and expose the latest aim target."""

    def __init__(self) -> None:
        """Initialize the tracker; no process is started until ``start``."""
        # spawn, not fork: forking the daemon would duplicate its GStreamer,
        # asyncio, and motor-controller state into the child.
        self._ctx = multiprocessing.get_context("spawn")
        self._process: BaseProcess | None = None
        self._target_mailbox: "SynchronizedArray[float]" = self._ctx.Array(
            "d", _TARGET_SLOTS
        )
        self._pose_mailbox: "SynchronizedArray[float]" = self._ctx.Array(
            "d", _POSE_SLOTS
        )
        self._active = self._ctx.Event()
        self._stop = self._ctx.Event()
        self._death_logged = False

    def start(self, camera_specs: CameraSpecs) -> None:
        """Start the detector process if it is not already running."""
        if self._process is not None and self._process.is_alive():
            return
        self._stop = self._ctx.Event()
        # Zero the mailboxes so stale targets from a previous run are dropped.
        with self._target_mailbox.get_lock():
            for i in range(_TARGET_SLOTS):
                self._target_mailbox[i] = 0.0
        with self._pose_mailbox.get_lock():
            for i in range(_POSE_SLOTS):
                self._pose_mailbox[i] = 0.0
        self._death_logged = False
        self._process = self._ctx.Process(
            target=_worker_entry,
            args=(
                camera_specs,
                self._target_mailbox,
                self._pose_mailbox,
                self._active,
                self._stop,
            ),
            daemon=True,
            name="face-tracker",
        )
        self._process.start()

    def set_active(self, active: bool) -> None:
        """Pause or resume detection; a paused tracker disconnects from the camera feed."""
        if active:
            self._active.set()
        else:
            self._active.clear()

    def publish_head_pose(self, roll: float, pitch: float, yaw: float) -> None:
        """Share the daemon's current head orientation with the detector."""
        with self._pose_mailbox.get_lock():
            self._pose_mailbox[0] = 1.0
            self._pose_mailbox[1] = roll
            self._pose_mailbox[2] = pitch
            self._pose_mailbox[3] = yaw

    def latest(self) -> TrackingTarget | None:
        """Return the latest published aim target, or None before the first one."""
        if (
            self._process is not None
            and not self._process.is_alive()
            and not self._stop.is_set()
            and not self._death_logged
        ):
            self._death_logged = True
            logger.warning(
                "Face tracker process died (exit code %s); no more targets.",
                self._process.exitcode,
            )
        with self._target_mailbox.get_lock():
            values = [self._target_mailbox[i] for i in range(_TARGET_SLOTS)]
        seq = int(values[0])
        if seq == 0:
            return None
        detected = values[1] != 0.0
        return TrackingTarget(
            seq=seq,
            detected=detected,
            roll=values[2],
            pitch=values[3],
            yaw=values[4],
            x=values[5] if detected else None,
            y=values[6] if detected else None,
            face_roll=values[7] if detected else None,
        )

    def stop(self) -> None:
        """Stop the detector process."""
        self._stop.set()
        process = self._process
        if process is None:
            return
        process.join(timeout=2.0)
        if process.is_alive():
            # Unlike a thread, a wedged child can be reclaimed by force.
            process.terminate()
            process.join(timeout=1.0)
        if process.is_alive():
            # Keep the handle: forgetting a live process lets start() double-run.
            logger.warning("Face tracker process did not stop in time.")
            return
        self._process = None
