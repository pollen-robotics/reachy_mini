"""Base class for robot backends, simulated or real.

This module defines the `Backend` class, which serves as a base for implementing
different types of robot backends, whether they are simulated (like Mujoco) or real
(connected via serial port). The class provides methods for managing joint positions,
torque control, and other backend-specific functionalities.
It is designed to be extended by subclasses that implement the specific behavior for
each type of backend.
"""

import asyncio
import json
import logging
import math
import os
import random
import tempfile
import threading
import time
import typing
from abc import abstractmethod
from pathlib import Path
from typing import Annotated, Any, Callable, Dict, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from reachy_mini.io.jsonrpc import looks_like_jsonrpc
from reachy_mini.io.protocol import (
    AnyCommand,
    AppendRecordCmd,
    ApplyAudioConfigCmd,
    CancelAudioCmd,
    CancelMoveCmd,
    ClearIncomingAudioCmd,
    DeleteHfTokenCmd,
    DoaSnapshot,
    FaceTarget,
    GetFirstWakeUpCmd,
    GetHardwareIdCmd,
    GetImuCmd,
    GetMicrophoneVolumeCmd,
    GetMotorModeCmd,
    GetRobotNameCmd,
    GetStateCmd,
    GetTrackedFaceCmd,
    GetVersionCmd,
    GetVolumeCmd,
    GotoSleepCmd,
    GotoTargetCmd,
    HeadTrackingMode,
    ImuDataMsg,
    LogLineMsg,
    LogStreamErrorMsg,
    MockupSimBackendStatus,
    MotorControlMode,
    MujocoBackendStatus,
    PlayRecordedMoveCmd,
    PlaySoundCmd,
    PlayUploadedAudioCmd,
    PlayUploadedMoveCmd,
    PoseFrame,
    PreloadDatasetCmd,
    ReadAudioParameterCmd,
    RecordedDataMsg,
    RestartDaemonCmd,
    RobotBackendStatus,
    SetAntennasCmd,
    SetAutomaticBodyYawCmd,
    SetBodyYawCmd,
    SetFirstWakeUpCmd,
    SetFullTargetCmd,
    SetGravityCompensationCmd,
    SetHeadJointsCmd,
    SetHeadTrackingCmd,
    SetMicrophoneVolumeCmd,
    SetMotorModeCmd,
    SetRobotNameCmd,
    SetSpeechOffsetsCmd,
    SetTargetCmd,
    SetTorqueCmd,
    SetVolumeCmd,
    SetWobblingCmd,
    StartRecordingCmd,
    StartUpdateCmd,
    StateSnapshot,
    StopMoveCmd,
    StopRecordingCmd,
    SubscribeLogsCmd,
    SubscribePoseCmd,
    UnsubscribeLogsCmd,
    UnsubscribePoseCmd,
    UploadAudioChunkCmd,
    UploadAudioFinishCmd,
    UploadAudioStartCmd,
    UploadMoveChunkCmd,
    UploadMoveFinishCmd,
    UploadMoveStartCmd,
    WakeUpCmd,
    command_adapter,
)
from reachy_mini.io.publisher import Publisher

if typing.TYPE_CHECKING:
    from reachy_mini.kinematics import AnyKinematics
# MediaManager no longer used here — play_sound delegated to GstMediaServer
from reachy_mini.media.audio_doa import AudioDoA
from reachy_mini.motion.goto import GotoMove
from reachy_mini.motion.move import Move
from reachy_mini.utils import create_head_pose
from reachy_mini.utils.constants import MODELS_ROOT_PATH, URDF_ROOT_PATH
from reachy_mini.utils.interpolation import (
    InterpolationTechnique,
    compose_world_offset,
    distance_between_poses,
    linear_pose_interpolation,
    time_trajectory,
)
from reachy_mini.vision.face_tracking import FaceTracker
from reachy_mini.vision.look_at import default_head_to_camera_transform


class _PlaybackCancelToken:
    """Per-run cancellation handle for ``Backend.play_move``.

    Created by ``_async_play_uploaded_move`` and stored on the backend
    under ``_active_move_token`` while the run is live. ``cancel_move``
    only flips the token whose ``upload_id`` matches the incoming
    command, so two back-to-back plays can't cross-cancel each other.
    """

    __slots__ = ("upload_id", "cancelled")

    def __init__(self, upload_id: str) -> None:
        self.upload_id = upload_id
        self.cancelled = False


class Backend:
    """Base class for robot backends, simulated or real."""

    def __init__(
        self,
        log_level: str = "INFO",
        check_collision: bool = False,
        kinematics_engine: str = "AnalyticalKinematics",
        use_audio: bool = True,
        wireless_version: bool = False,
    ) -> None:
        """Initialize the backend."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        self.use_audio = use_audio

        self.doa = AudioDoA() if use_audio else None
        # Cache for the latest DoA reading, refreshed by a demand-driven
        # poller thread (see `_doa_poll_loop` for the rationale and the
        # locking rules).
        self._doa_lock = threading.Lock()
        # Serializes every daemon-side ReSpeaker USB conversation (DoA reads
        # and audio-config commands): `ReSpeaker.read()` is a multi-step
        # conversation on a non-thread-safe pyusb device, so two of them must
        # never interleave.
        self._doa_usb_lock = threading.Lock()
        self._last_doa: tuple[float, bool] | None = None
        # Monotonic time of the last cache consumer; keeps the poller alive.
        self._doa_demand_ts: float = 0.0
        self._doa_stop = threading.Event()
        self._doa_thread: threading.Thread | None = None

        self.should_stop = threading.Event()
        self.ready = threading.Event()

        self.check_collision = (
            check_collision  # Flag to enable/disable collision checking
        )
        self.kinematics_engine = kinematics_engine

        self.logger.info(f"Using {self.kinematics_engine} kinematics engine")

        if self.check_collision:
            assert self.kinematics_engine == "Placo", (
                "Collision checking is only available with Placo Kinematics"
            )

        self.gravity_compensation_mode = False  # Flag for gravity compensation mode

        if self.gravity_compensation_mode:
            assert self.kinematics_engine == "Placo", (
                "Gravity compensation is only available with Placo kinematics"
            )

        if self.kinematics_engine == "Placo":
            from reachy_mini.kinematics import PlacoKinematics

            self.head_kinematics: AnyKinematics = PlacoKinematics(
                URDF_ROOT_PATH, check_collision=self.check_collision
            )
        elif self.kinematics_engine == "NN":
            from reachy_mini.kinematics import NNKinematics

            self.head_kinematics = NNKinematics(MODELS_ROOT_PATH)
        elif self.kinematics_engine == "AnalyticalKinematics":
            from reachy_mini.kinematics import AnalyticalKinematics

            self.head_kinematics = AnalyticalKinematics()
        else:
            raise ValueError(
                f"Unknown kinematics engine: {self.kinematics_engine}. Use 'Placo', 'NN' or 'AnalyticalKinematics'."
            )

        self.current_head_pose: Annotated[NDArray[np.float64], (4, 4)] | None = (
            None  # 4x4 pose matrix
        )
        self.target_head_pose: Annotated[NDArray[np.float64], (4, 4)] | None = (
            None  # 4x4 pose matrix
        )
        self.target_body_yaw: float | None = (
            None  # Last body yaw used in IK computations
        )

        self.target_head_joint_positions: (
            Annotated[NDArray[np.float64], (7,)] | None
        ) = None  # [yaw, 0, 1, 2, 3, 4, 5]
        self.current_head_joint_positions: (
            Annotated[NDArray[np.float64], (7,)] | None
        ) = None  # [yaw, 0, 1, 2, 3, 4, 5]
        self.target_antenna_joint_positions: (
            Annotated[NDArray[np.float64], (2,)] | None
        ) = None  # [0, 1]
        self.current_antenna_joint_positions: (
            Annotated[NDArray[np.float64], (2,)] | None
        ) = None  # [0, 1]

        self.joint_positions_publisher: Publisher | None = None
        self.pose_publisher: Publisher | None = None
        self.recording_publisher: Publisher | None = None
        self.imu_publisher: Publisher | None = None
        self.error: str | None = None  # To store any error that occurs during execution
        self.is_recording = False  # Flag to indicate if recording is active
        self.recorded_data: list[dict[str, Any]] = []  # List to store recorded data

        # variables to store the last computed head joint positions and pose
        self._last_target_body_yaw: float | None = (
            None  # Last body yaw used in IK computations
        )
        self._last_target_head_pose: Annotated[NDArray[np.float64], (4, 4)] | None = (
            None  # Last head pose used in IK computations
        )
        self.target_head_joint_current: Annotated[NDArray[np.float64], (7,)] | None = (
            None  # Placeholder for head joint torque
        )
        self.ik_required = False  # Flag to indicate if IK computation is required

        self.is_shutting_down = False

        # Tolerance for kinematics computations
        # For Forward kinematics (around 0.25deg)
        # - FK is calculated at each timestep and is susceptible to noise
        self._fk_kin_tolerance = 1e-3  # rads
        # For Inverse kinematics (around 0.5mm and 0.1 degrees)
        # - IK is calculated only when the head pose is set by the user
        self._ik_kin_tolerance = {
            "rad": 2e-3,  # rads
            "m": 0.5e-3,  # m
        }

        # Recording lock to guard buffer swaps and appends
        self._rec_lock = threading.Lock()

        # Reference to the media server for play_sound delegation.
        # Set via setup_media_server().
        self._media_server: Optional[Any] = None

        # Guard to ensure only one play_move/goto is executed at a time (goto itself uses play_move, so we need an RLock)
        self._play_move_lock = threading.RLock()
        self._active_move_depth = (
            0  # Tracks nested acquisitions within the owning thread
        )
        # Global "stop now" flag flipped by ``StopMoveCmd`` and polled by
        # every ``play_move`` loop (nested gotos included), regardless of
        # who started the move. Cleared when a new top-level move starts,
        # so a stale stop can't kill the next run.
        self._stop_move_requested = False

        # Per-run cancellation handle for the active play_uploaded_move.
        # Set by ``_async_play_uploaded_move`` before each play, cleared
        # in its finally. ``CancelMoveCmd`` only flips this token if its
        # upload_id matches — see _PlaybackCancelToken.
        # Mutated under ``_play_move_lock`` to close the race where a
        # cancel arrives between the play task being scheduled and the
        # token being installed.
        self._active_move_token: Optional[_PlaybackCancelToken] = None
        # upload_id of the standalone audio currently playing via
        # ``play_uploaded_audio``. None when no standalone audio is
        # playing. ``CancelAudioCmd`` only stops if its upload_id
        # matches — keeps cancel_audio from killing the audio
        # attached to a play_uploaded_move that's also running.
        self._active_audio_upload_id: Optional[str] = None

        # In-progress uploads keyed by client-supplied upload_id.
        # ``_chunks`` is a list of received fragments (string parts of
        # the JSON-serialized move) appended in chunk_index order;
        # any out-of-order delivery discards the slot.  ``_meta``
        # holds the declared total_chunks + description so we can
        # validate completeness.
        self._upload_chunks: Dict[str, list[str]] = {}
        self._upload_meta: Dict[str, dict[str, Any]] = {}
        self._upload_ts: Dict[str, float] = {}
        # Parsed RecordedMove instances, keyed by upload_id.  Filled
        # by upload_move_finish, consumed by play_uploaded_move, and
        # evicted whenever the slot is no longer needed (play
        # finished/cancelled, or TTL expiry).
        self._uploaded_moves: Dict[str, Any] = {}
        # Audio upload slots, parallel to the move ones.  The audio
        # bytes are base64-encoded WAV fragments; on finish we decode
        # and write to ``<tempdir>/reachy-mini-uploads/audio/{upload_id}.wav``
        # so GStreamer playbin can consume them by path.
        # ``_uploaded_audios`` maps upload_id -> on-disk path; cleared
        # after play.
        self._audio_chunks: Dict[str, list[str]] = {}
        self._audio_meta: Dict[str, dict[str, Any]] = {}
        self._audio_ts: Dict[str, float] = {}
        self._uploaded_audios: Dict[str, str] = {}
        # Soft caps to keep memory bounded.  At 100 Hz a 10-minute
        # move is ~60 000 frames; one slot at a time is the expected
        # use; the limits exist to fail fast on misuse.
        self._upload_ttl_s: float = 300.0  # evict half-finished slots
        self._upload_max_active_slots: int = 4
        # Where uploaded audio files are written before GStreamer
        # plays them. Created lazily on first upload. Uses the
        # platform-native tempdir (``/tmp`` on Linux/macOS,
        # ``%TEMP%`` on Windows) so the daemon works whether it's
        # bundled inside the desktop app or running on the CM4.
        self._audio_temp_dir: str = os.path.join(
            tempfile.gettempdir(), "reachy-mini-uploads", "audio"
        )

        # Head wobbler speech offsets (x_m, y_m, z_m, roll_rad, pitch_rad, yaw_rad)
        self._speech_offsets: tuple[float, float, float, float, float, float] = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        self._tracking_enabled = False
        self._tracking_requested_weight = 1.0
        self._tracking_weight = 0.0
        # Aim servo: each control tick moves the aim angles by
        # clip(p * error, +/-max_step) toward the latest absolute target
        # angles published by the detector process. p sets responsiveness;
        # max_step is a hard per-tick velocity limit (~90 deg/s at 50 Hz).
        self._tracking_p = 0.15
        self._tracking_max_step_rad = float(np.radians(1.8))
        self._tracking_lost_timeout = 2.0
        self._tracking_aim: Annotated[NDArray[np.float64], (4, 4)] | None = None
        self._tracking_aim_rpy: tuple[float, float, float] | None = None
        self._tracking_target_rpy: tuple[float, float, float] | None = None
        self._tracking_seq = 0
        # Re-aim policy (see SetHeadTrackingCmd.mode). The detector keeps its
        # target mailbox fresh in every mode; the mode only decides when the
        # daemon latches a fresh target as its aim.
        self._tracking_mode: HeadTrackingMode = "continuous"
        self._tracking_glance_interval_s: tuple[float, float] = (5.0, 30.0)
        self._tracking_speaking_hold_s = 1.0
        self._tracking_next_glance_at: float | None = None
        self._tracking_last_speech_ts: float | None = None
        self._tracking_rng = random.Random()
        self._last_face_seen: float | None = None
        self._tracker: FaceTracker | None = None
        self._tracking_lock = threading.Lock()
        self._face_target = FaceTarget()
        self.T_head_cam = default_head_to_camera_transform()

        # WebRTC support
        self._send_message_to_webrtc: Optional[Callable[[Optional[str], str], None]] = (
            None
        )
        # JSON-RPC control surface. When set, DataChannel frames carrying a
        # ``"jsonrpc": "2.0"`` field are handed to this handler (the daemon's
        # app relay) instead of the legacy ``{"type": ...}`` command path.
        # Signature: ``handler(raw_message, reply)`` where ``reply`` sends a
        # response dict back to the originating peer. Wired by ``Daemon.start``.
        self._jsonrpc_handler: Optional[
            Callable[[str, Callable[[dict[str, Any]], None]], None]
        ] = None
        # WS broadcast callback. Set by WSServer.start() so the
        # backend can fan unsolicited events out to every WS client
        # using the same drop-oldest queues the state publishers use.
        # The WebRTC broadcast path goes through
        # ``_send_message_to_webrtc(None, ...)`` instead.
        self._ws_broadcast_callback: Optional[Callable[[str], None]] = None

        # Per-peer journalctl streaming tasks. Populated when a peer
        # sends `subscribe_logs`, cancelled on `unsubscribe_logs` or
        # peer disconnect (the latter is wired in setup_media_server).
        self._log_tasks: dict[str, "asyncio.Task[None]"] = {}
        # Monotonic sequence number stamped on every pushed pose frame so the
        # client can drop out-of-order/stale frames that arrive on the
        # unordered `pose` data channel (see `build_state_json`). Bumped only
        # from the media server's GLib thread (single writer), so a plain int
        # is safe.
        self._pose_seq = 0
        # Asyncio loop on which the log-streaming tasks run. Captured
        # in setup_media_server alongside the WebRTC handler loop, so
        # cross-thread cleanup (peer disconnect fires from gstreamer's
        # GLib thread) can schedule cancellation correctly.
        self._log_loop: Optional["asyncio.AbstractEventLoop"] = None

        # Synchronous callback that triggers a full daemon restart
        # (motor controller, kinematics, media server, ...). Wired in
        # by `Daemon` after `setup_media_server` so that the typed
        # `restart_daemon` DataChannel command can recover from a
        # broken backend (e.g. dead motor controller) without forcing
        # an out-of-band `systemctl restart` or REST round-trip.
        # Returns immediately - the actual restart runs on a fresh
        # thread so the data channel can flush its ack first.
        self._restart_daemon_callback: Optional[Callable[[], None]] = None

        # Synchronous callback that triggers a PyPI update of the daemon
        # followed by a restart. Wired in by `Daemon`, same fire-and-ack
        # contract as `_restart_daemon_callback`: the update ends with a
        # `systemctl restart` that tears the transport down, so the
        # callback must return promptly (it spawns its own thread). It runs
        # cheap pre-checks (wireless robot, update available, not already
        # running) synchronously and returns a refusal reason string when it
        # declines, or ``None`` once the update job has been accepted.
        self._start_update_callback: Optional[Callable[[bool], Optional[str]]] = None

        # Synchronous callback fired once every time wake_up() completes. Wired
        # in by `Daemon` to auto-start a configured startup app after the robot
        # wakes (however the wake was triggered: on-start, button, or REST).
        self._on_wake_up_callback: Optional[Callable[[], None]] = None

        # In-flight "return to a clean idle state" task, scheduled by
        # request_idle_reset() when the managed app slot becomes free. Kept so a
        # fast reconnect (or a local app grabbing the robot) can cancel it
        # instead of letting a stale goto_sleep fight the new session.
        self._idle_reset_task: Optional["asyncio.Task[None]"] = None
        # Synchronous callback fired after a `set_robot_name` command persists
        # a new name. Wired in by the app lifespan to apply the rename live
        # (daemon status + central relay + mDNS) so it takes effect without a
        # restart. Receives the stored (trimmed) name and must return promptly.
        self._set_robot_name_callback: Optional[Callable[[str], None]] = None

    # Life cycle methods
    def wrapped_run(self) -> None:
        """Run the backend in a try-except block to store errors."""
        try:
            self.run()
        except Exception as e:
            self.error = str(e)
            self.close()
            raise e

    def run(self) -> None:
        """Run the backend.

        This method is a placeholder and should be overridden by subclasses.
        """
        raise NotImplementedError("The method run should be overridden by subclasses.")

    def close(self) -> None:
        """Close the backend and release resources.

        Subclasses should override this method to add their own cleanup logic,
        and call super().close() at the end.

        Note: This base implementation handles common cleanup.
        Subclasses must still implement their own cleanup for backend-specific resources.
        """
        self.logger.debug("Backend.close() - cleaning up resources")
        # Stop the DoA poller and release the ReSpeaker USB handle.
        self._doa_stop.set()
        if self._doa_thread is not None:
            self._doa_thread.join(timeout=1.0)
            self._doa_thread = None
        # `join(timeout=...)` can return with a wedged USB read still in
        # flight, so only the lock proves the poller isn't mid-`ctrl_transfer`
        # when the device is disposed.
        with self._doa_usb_lock:
            if self.doa is not None:
                try:
                    self.doa.close()
                except Exception:  # noqa: BLE001 - best-effort teardown
                    self.logger.debug("DoA close failed", exc_info=True)
                self.doa = None
        self._media_server = None

    @property
    def is_move_running(self) -> bool:
        """Return True if a move is currently executing."""
        return self._active_move_depth > 0

    def _try_start_move(self) -> bool:
        """Attempt to acquire the move guard, returning False if another client already owns it."""
        if not self._play_move_lock.acquire(blocking=False):
            return False
        self._active_move_depth += 1
        return True

    def _end_move(self) -> None:
        """Release the move guard; paired with every successful _try_start_move()."""
        if self._active_move_depth > 0:
            self._active_move_depth -= 1
        self._play_move_lock.release()

    def get_status(
        self,
    ) -> "RobotBackendStatus | MujocoBackendStatus | MockupSimBackendStatus":
        """Return backend statistics.

        This method is a placeholder and should be overridden by subclasses.
        """
        raise NotImplementedError(
            "The method get_status should be overridden by subclasses."
        )

    # Present/Target joint positions
    def set_joint_positions_publisher(self, publisher: Publisher) -> None:
        """Set the publisher for joint positions.

        Args:
            publisher: A publisher object that will be used to publish joint positions.

        """
        self.joint_positions_publisher = publisher

    def set_pose_publisher(self, publisher: Publisher) -> None:
        """Set the publisher for head pose.

        Args:
            publisher: A publisher object that will be used to publish head pose.

        """
        self.pose_publisher = publisher

    def set_imu_publisher(self, publisher: Publisher) -> None:
        """Set the publisher for IMU data.

        Args:
            publisher: A publisher object that will be used to publish IMU data.

        """
        self.imu_publisher = publisher

    def get_imu_data(self) -> ImuDataMsg | None:
        """Latest IMU reading; None on backends without an IMU."""
        return None

    def update_target_head_joints_from_ik(
        self,
        pose: Annotated[NDArray[np.float64], (4, 4)] | None = None,
        body_yaw: float | None = None,
    ) -> None:
        """Update the target head joint positions from inverse kinematics.

        Args:
            pose (np.ndarray): 4x4 pose matrix representing the head pose.
            body_yaw (float): The yaw angle of the body, used to adjust the head pose.

        """
        if pose is None:
            pose = (
                self.target_head_pose
                if self.target_head_pose is not None
                else np.eye(4)
            )

        if body_yaw is None:
            body_yaw = self.target_body_yaw if self.target_body_yaw is not None else 0.0

        aim = self._tracking_aim
        if aim is not None and self._tracking_weight > 0.0:
            weight = min(max(self._tracking_weight, 0.0), 1.0)
            pose = linear_pose_interpolation(pose, aim, weight)

        # Compose speech wobbler offsets (if any) before IK
        if any(o != 0.0 for o in self._speech_offsets):
            x_m, y_m, z_m, roll_r, pitch_r, yaw_r = self._speech_offsets
            offset_pose = create_head_pose(
                x=x_m,
                y=y_m,
                z=z_m,
                roll=roll_r,
                pitch=pitch_r,
                yaw=yaw_r,
                degrees=False,
            )
            pose = compose_world_offset(pose, offset_pose)

        # Compute the inverse kinematics to get the head joint positions
        joints = self.head_kinematics.ik(pose, body_yaw=body_yaw)
        if joints is None or np.any(np.isnan(joints)):
            raise ValueError("WARNING: Collision detected or head pose not achievable!")

        # update the target head pose and body yaw
        self._last_target_head_pose = pose
        self._last_target_body_yaw = body_yaw

        self.target_head_joint_positions = joints

    def set_target_head_pose(
        self,
        pose: Annotated[NDArray[np.float64], (4, 4)],
    ) -> None:
        """Set the target head pose for the robot.

        Args:
            pose (np.ndarray): 4x4 pose matrix representing the head pose.

        """
        self.target_head_pose = pose
        if self._tracking_aim is not None and self._tracking_weight >= 1.0:
            return
        self.ik_required = True

    def set_target_body_yaw(self, body_yaw: float) -> None:
        """Set the target body yaw for the robot.

        Only used when doing a set_target() with a standalone body_yaw (no head pose).

        Args:
            body_yaw (float): The yaw angle of the body

        """
        if (
            self.target_body_yaw is not None
            and abs(self.target_body_yaw - body_yaw) <= 1e-9
        ):
            return
        self.target_body_yaw = body_yaw
        self.ik_required = True

    def set_target_head_joint_positions(
        self, positions: Annotated[NDArray[np.float64], (7,)] | None
    ) -> None:
        """Set the head joint positions.

        Args:
            positions (List[float]): A list of joint positions for the head.

        """
        self.target_head_joint_positions = positions
        self.ik_required = False

    def set_target(
        self,
        head: Annotated[NDArray[np.float64], (4, 4)] | None = None,  # 4x4 pose matrix
        antennas: Annotated[NDArray[np.float64], (2,)]
        | None = None,  # [right_angle, left_angle] (in rads)
        body_yaw: float | None = None,  # Body yaw angle in radians
    ) -> None:
        """Set the target head pose and/or antenna positions and/or body_yaw."""
        if head is not None:
            self.set_target_head_pose(head)

        if body_yaw is not None:
            self.set_target_body_yaw(body_yaw)

        if antennas is not None:
            self.set_target_antenna_joint_positions(antennas)

    def set_target_antenna_joint_positions(
        self,
        positions: Annotated[NDArray[np.float64], (2,)],
    ) -> None:
        """Set the antenna joint positions.

        Args:
            positions (List[float]): A list of joint positions for the antenna.

        """
        self.target_antenna_joint_positions = positions

    def set_speech_offsets(
        self,
        offsets: tuple[float, float, float, float, float, float],
    ) -> None:
        """Set head wobbler speech offsets, composed with target pose before IK.

        Args:
            offsets: ``(x_m, y_m, z_m, roll_rad, pitch_rad, yaw_rad)`` in
                world frame.  Zero tuple disables the offset.

        """
        self._speech_offsets = offsets
        if any(o != 0.0 for o in offsets):
            # Non-zero offsets mean the head wobbler is reacting to speech.
            self._tracking_last_speech_ts = time.monotonic()
        self.ik_required = True

    def enable_head_tracking(
        self,
        weight: float = 1.0,
        mode: HeadTrackingMode = "continuous",
        glance_interval_s: tuple[float, float] = (5.0, 30.0),
        speaking_hold_s: float = 1.0,
    ) -> bool:
        """Enable head tracking; ``weight`` 0 pauses the worker without stopping it.

        ``mode`` and its tunables select when the aim follows the face (see
        ``SetHeadTrackingCmd``); calling again while enabled changes them live.
        """
        with self._tracking_lock:
            self._tracking_requested_weight = min(max(float(weight), 0.0), 1.0)
            if mode != self._tracking_mode:
                # A mode change re-aims once right away, then applies the new policy.
                self._tracking_next_glance_at = None
            self._tracking_mode = mode
            self._tracking_glance_interval_s = (
                float(glance_interval_s[0]),
                float(glance_interval_s[1]),
            )
            self._tracking_speaking_hold_s = max(0.0, float(speaking_hold_s))
            if self._media_server is None:
                self.logger.warning("Cannot enable head tracking: no camera available")
                return False
            if self._tracking_requested_weight > 0.0:
                if self._tracker is None:
                    self._tracker = FaceTracker()
                self._tracker.start(self._media_server.camera_specs)
                self._tracker.set_active(True)
            else:
                if self._tracker is not None:
                    self._tracker.set_active(False)
                self.clear_tracking_aim()
            self._tracking_enabled = True
        return True

    def disable_head_tracking(self) -> None:
        """Disable daemon-side visual head tracking, stopping the detector thread."""
        with self._tracking_lock:
            self._tracking_enabled = False
            tracker = self._tracker
            self._tracker = None
            self.clear_tracking_aim()
        if tracker is not None:
            tracker.stop()

    def clear_tracking_aim(self) -> None:
        """Clear the tracking aim, latched target, and latest detected face."""
        self._tracking_aim = None
        self._tracking_aim_rpy = None
        self._tracking_target_rpy = None
        self._tracking_seq = 0
        self._tracking_next_glance_at = None
        self._tracking_weight = 0.0
        self._last_face_seen = None
        self._face_target = FaceTarget()
        self.ik_required = True

    @staticmethod
    def _pose_rpy(
        pose: Annotated[NDArray[np.float64], (4, 4)],
    ) -> tuple[float, float, float]:
        """Extract (roll, pitch, yaw), extrinsic x-y-z radians, from a pose."""
        roll, pitch, yaw = R.from_matrix(pose[:3, :3]).as_euler("xyz")
        return (float(roll), float(pitch), float(yaw))

    def _robot_is_speaking(self, now: float) -> bool:
        """Whether audio reached the speaker or speech offsets moved recently.

        Two daemon-side signals, no app cooperation needed: the media server
        stamps every output audio buffer above a small energy threshold, and
        ``set_speech_offsets`` stamps non-zero wobbler offsets (covers apps
        that play audio elsewhere but drive the wobbler).
        """
        last = self._tracking_last_speech_ts
        media_server = self._media_server
        audio_ts = (
            media_server.last_audio_output_time()
            if media_server is not None
            and hasattr(media_server, "last_audio_output_time")
            else None
        )
        if audio_ts is not None and (last is None or audio_ts > last):
            last = audio_ts
        return last is not None and now - last <= self._tracking_speaking_hold_s

    def _schedule_next_glance(self, now: float) -> None:
        low, high = self._tracking_glance_interval_s
        self._tracking_next_glance_at = now + self._tracking_rng.uniform(low, high)

    def _should_latch_target(self, now: float) -> bool:
        """Apply the re-aim policy to a fresh detected target."""
        if self._tracking_target_rpy is None:
            # The first target after (re)enabling always aims, in every mode.
            if self._tracking_mode == "periodic":
                self._schedule_next_glance(now)
            return True
        if self._tracking_mode == "continuous":
            return True
        if self._tracking_mode == "speaking":
            return self._robot_is_speaking(now)
        if (
            self._tracking_next_glance_at is None
            or now >= self._tracking_next_glance_at
        ):
            self._schedule_next_glance(now)
            return True
        return False

    def step_head_tracking(self) -> None:
        """Servo the aim one bounded step toward the latest target angles.

        The detector process publishes absolute head angles ("the face is at
        these angles in the robot frame"); this method treats the latest
        publication as where the person is now and moves the aim by
        ``clip(p * error, +/-max_step)`` per tick. Re-applying a stale absolute
        target converges and stops, so no timestamps are needed. A brief
        detection gap holds the last target; a sustained loss retargets the
        neutral pose so the robot recenters instead of freezing where the
        person left the frame. The tracking mode (continuous, periodic,
        speaking) decides which fresh targets are adopted as the aim.
        """
        with self._tracking_lock:
            if not self._tracking_enabled or self._tracking_requested_weight <= 0.0:
                return
            now = time.monotonic()
            if self._tracker is not None:
                # Share the current head orientation so the detector can turn
                # a face pixel into absolute angles.
                self._tracker.publish_head_pose(
                    *self._pose_rpy(self.get_current_head_pose())
                )
                target = self._tracker.latest()
                if target is not None and target.seq != self._tracking_seq:
                    self._tracking_seq = target.seq
                    if target.detected:
                        if self._should_latch_target(now):
                            self._tracking_target_rpy = (
                                target.roll,
                                target.pitch,
                                target.yaw,
                            )
                        self._tracking_weight = self._tracking_requested_weight
                        self._last_face_seen = now
                        self._face_target = FaceTarget(
                            detected=True,
                            x=target.x,
                            y=target.y,
                            roll=target.face_roll,
                            ts=now,
                        )
                    else:
                        # Hold the last target on a transient loss so the head
                        # doesn't lurch to neutral.
                        self._face_target = FaceTarget(detected=False, ts=now)
            if (
                self._last_face_seen is not None
                and now - self._last_face_seen >= self._tracking_lost_timeout
            ):
                self._tracking_target_rpy = (0.0, 0.0, 0.0)
            if self._tracking_target_rpy is None:
                return
            if self._tracking_aim_rpy is None:
                self._tracking_aim_rpy = self._pose_rpy(self.get_current_head_pose())
            max_step = self._tracking_max_step_rad
            roll, pitch, yaw = (
                current
                + min(
                    max(
                        self._tracking_p * math.remainder(target - current, math.tau),
                        -max_step,
                    ),
                    max_step,
                )
                for current, target in zip(
                    self._tracking_aim_rpy, self._tracking_target_rpy
                )
            )
            self._tracking_aim_rpy = (roll, pitch, yaw)
            self._tracking_aim = create_head_pose(
                roll=roll, pitch=pitch, yaw=yaw, degrees=False
            )
            self.ik_required = True

    def get_tracked_face(self) -> FaceTarget:
        """Return the latest face target observed by daemon-side tracking."""
        return self._face_target

    def set_target_head_joint_current(
        self,
        current: Annotated[NDArray[np.float64], (7,)],
    ) -> None:
        """Set the head joint current.

        Args:
            current (Annotated[NDArray[np.float64], (7,)]): A list of current values for the head motors.

        """
        self.target_head_joint_current = current
        self.ik_required = False

    async def play_move(
        self,
        move: Move,
        play_frequency: float = 100.0,
        initial_goto_duration: float = 0.0,
        audio_lead_s: float = 0.0,
        cancel_token: Optional[_PlaybackCancelToken] = None,
    ) -> None:
        """Asynchronously play a Move.

        Args:
            move (Move): The Move object to be played.
            play_frequency (float): The frequency at which to evaluate the move (in Hz).
            initial_goto_duration (float): Duration for an initial goto to the move's starting position. If 0.0, no initial goto is performed.
            audio_lead_s (float): How many seconds the audio (if any) starts BEFORE the motion. Positive values compensate for the constant GStreamer playbin latency on the robot so the audio reaches the speaker at the same moment the actuator starts moving. Negative values delay audio relative to motion. No-op when the move has no sound_path. Default 0.
            cancel_token (_PlaybackCancelToken, optional): If provided, the inner loop polls ``cancel_token.cancelled`` every tick and exits when flipped. Used by ``_async_play_uploaded_move`` to wire ``cancel_move`` to a specific upload_id; direct callers (goto_target, etc.) pass None. Independently of the token, every loop also polls the global ``_stop_move_requested`` flag flipped by ``StopMoveCmd``, so any move is interruptible by ``stop_move``.

        """
        if not self._try_start_move():
            self.logger.warning("Ignoring play_move request: another move is running.")
            return

        if self._active_move_depth == 1:
            # New top-level move: clear any stop request left over from a
            # previous run so a stale stop_move can't kill this one. Nested
            # play_moves (initial goto, goto_target inside a recorded move)
            # keep the flag so one stop interrupts the whole stack.
            self._stop_move_requested = False

        try:
            if initial_goto_duration > 0.0:
                start_head_pose, start_antennas_positions, start_body_yaw = (
                    move.evaluate(0.0)
                )
                await self.goto_target(
                    head=start_head_pose,
                    antennas=start_antennas_positions,
                    duration=initial_goto_duration,
                    body_yaw=start_body_yaw,
                )
            sleep_period = 1.0 / play_frequency

            # Sound handoff.  audio_lead_s shifts the audio start
            # relative to the motion loop:
            #   > 0: audio starts first, motion follows after the wait
            #   < 0: motion starts first, audio follows
            #   = 0: kick off audio just before entering the loop (legacy behaviour)
            if move.sound_path is not None and audio_lead_s > 0:
                self.play_sound(str(move.sound_path))
                await asyncio.sleep(audio_lead_s)
            elif move.sound_path is not None and audio_lead_s == 0:
                self.play_sound(str(move.sound_path))

            t0 = time.time()
            # Negative audio_lead_s: schedule sound to fire mid-loop,
            # |audio_lead_s| seconds after t0. We hold a reference to
            # the background task so the finally block can cancel it
            # if the motion loop exits before the sleep elapses (short
            # move + big negative lead would otherwise leak a task that
            # plays audio after the move has ended).
            delayed_sound_task: Optional["asyncio.Task[None]"] = None
            if move.sound_path is not None and audio_lead_s < 0:
                sound_path_str = str(move.sound_path)

                async def _delayed_sound() -> None:
                    try:
                        await asyncio.sleep(-audio_lead_s)
                    except asyncio.CancelledError:
                        return
                    if cancel_token is not None and cancel_token.cancelled:
                        return
                    if self._stop_move_requested:
                        return
                    self.play_sound(sound_path_str)

                delayed_sound_task = asyncio.create_task(_delayed_sound())
            interrupted = False
            try:
                while time.time() - t0 < move.duration:
                    if self._stop_move_requested or (
                        cancel_token is not None and cancel_token.cancelled
                    ):
                        self.logger.info("play_move cancelled, exiting playback loop")
                        interrupted = True
                        break
                    t = time.time() - t0

                    head, antennas, body_yaw = move.evaluate(t)
                    if head is not None:
                        self.set_target_head_pose(head)
                    if body_yaw is not None:
                        self.set_target_body_yaw(body_yaw)
                    if antennas is not None:
                        self.set_target_antenna_joint_positions(antennas)

                    elapsed = time.time() - t0 - t
                    if elapsed < sleep_period:
                        await asyncio.sleep(sleep_period - elapsed)
                    else:
                        await asyncio.sleep(0.001)
            finally:
                # Don't leak the delayed-sound task past the loop. Two
                # cases this matters: the move duration was shorter
                # than |audio_lead_s|, or the loop was cancelled before
                # the sleep elapsed.
                if delayed_sound_task is not None and not delayed_sound_task.done():
                    delayed_sound_task.cancel()
                # Interrupted mid-move: silence the sidecar sound too,
                # otherwise a recorded move's audio would keep playing to
                # the end of the WAV after the motion has stopped.
                if interrupted and move.sound_path is not None:
                    try:
                        self.stop_sound()
                    except Exception as e:
                        self.logger.warning(f"play_move: stop_sound failed: {e}")
        finally:
            self._end_move()

    async def goto_target(
        self,
        head: Annotated[NDArray[np.float64], (4, 4)] | None = None,  # 4x4 pose matrix
        antennas: Annotated[NDArray[np.float64], (2,)]
        | None = None,  # [right_angle, left_angle] (in rads)
        duration: float = 0.5,  # Duration in seconds for the movement, default is 0.5 seconds.
        method: InterpolationTechnique = InterpolationTechnique.MIN_JERK,  # can be "linear", "minjerk", "ease_in_out" or "cartoon", default is "minjerk"
        body_yaw: float | None = 0.0,  # Body yaw angle in radians
    ) -> None:
        """Asynchronously go to a target head pose and/or antennas position using task space interpolation, in "duration" seconds.

        Args:
            head (np.ndarray | None): 4x4 pose matrix representing the target head pose.
            antennas (np.ndarray | list[float] | None): 1D array with two elements representing the angles of the antennas in radians.
            duration (float): Duration of the movement in seconds.
            method (str): Interpolation method to use ("linear", "minjerk", "ease_in_out", "cartoon"). Default is "minjerk".
            body_yaw (float | None): Body yaw angle in radians.

        Raises:
            ValueError: If neither head nor antennas are provided, or if duration is not positive.

        """
        return await self.play_move(
            move=GotoMove(
                start_head_pose=self.get_present_head_pose(),
                target_head_pose=head,
                start_body_yaw=self.get_present_body_yaw(),
                target_body_yaw=body_yaw,
                start_antennas=np.array(self.get_present_antenna_joint_positions()),
                target_antennas=np.array(antennas) if antennas is not None else None,
                duration=duration,
                method=method,
            )
        )

    async def goto_joint_positions(
        self,
        head_joint_positions: list[float]
        | None = None,  # [yaw, stewart_platform x 6] length 7
        antennas_joint_positions: list[float]
        | None = None,  # [right_angle, left_angle] length 2
        duration: float = 0.5,  # Duration in seconds for the movement
        method: InterpolationTechnique = InterpolationTechnique.MIN_JERK,  # can be "linear", "minjerk", "ease_in_out" or "cartoon", default is "minjerk"
    ) -> None:
        """Asynchronously go to a target head joint positions and/or antennas joint positions using joint space interpolation, in "duration" seconds.

        Go to a target head joint positions and/or antennas joint positions using joint space interpolation, in "duration" seconds.

        Args:
            head_joint_positions (Optional[List[float]]): List of head joint positions in radians (length 7).
            antennas_joint_positions (Optional[List[float]]): List of antennas joint positions in radians (length 2).
            duration (float): Duration of the movement in seconds. Default is 0.5 seconds.
            method (str): Interpolation method to use ("linear", "minjerk", "ease_in_out", "cartoon"). Default is "minjerk".

        Raises:
            ValueError: If neither head_joint_positions nor antennas_joint_positions are provided, or if duration is not positive.

        """
        if duration <= 0.0:
            raise ValueError(
                "Duration must be positive and non-zero. Use set_target() for immediate position setting."
            )

        start_head = np.array(self.get_present_head_joint_positions())
        start_antennas = np.array(self.get_present_antenna_joint_positions())

        target_head = (
            np.array(head_joint_positions)
            if head_joint_positions is not None
            else start_head
        )
        target_antennas = (
            np.array(antennas_joint_positions)
            if antennas_joint_positions is not None
            else start_antennas
        )

        t0 = time.time()
        while time.time() - t0 < duration:
            t = time.time() - t0

            interp_time = time_trajectory(t / duration, method=method)

            head_joint = start_head + (target_head - start_head) * interp_time
            antennas_joint = (
                start_antennas + (target_antennas - start_antennas) * interp_time
            )

            self.set_target_head_joint_positions(head_joint)
            self.set_target_antenna_joint_positions(antennas_joint)
            await asyncio.sleep(0.01)

    def set_recording_publisher(self, publisher: Publisher) -> None:
        """Set the publisher for recording data.

        Args:
            publisher: A publisher object that will be used to publish recorded data.

        """
        self.recording_publisher = publisher

    def append_record(self, record: dict[str, Any]) -> None:
        """Append a record to the recorded data.

        Args:
            record (dict): A dictionary containing the record data to be appended.

        """
        if not self.is_recording:
            return
        # Double-check under lock to avoid race with stop_recording
        with self._rec_lock:
            if self.is_recording:
                self.recorded_data.append(record)

    def start_recording(self) -> None:
        """Start recording data."""
        with self._rec_lock:
            self.recorded_data = []
            self.is_recording = True

    def stop_recording(self) -> None:
        """Stop recording data and publish the recorded data."""
        # Swap buffer under lock so writers cannot touch the published list
        with self._rec_lock:
            self.is_recording = False
            recorded_data, self.recorded_data = self.recorded_data, []
        # Publish outside the lock
        if self.recording_publisher is not None:
            self.recording_publisher.put(RecordedDataMsg(data=recorded_data))
        else:
            self.logger.warning(
                "stop_recording called but recording_publisher is not set; dropping data."
            )

    def get_present_head_joint_positions(self) -> Annotated[NDArray[np.float64], (7,)]:
        """Return the present head joint positions.

        This method is a placeholder and should be overridden by subclasses.
        """
        raise NotImplementedError(
            "The method get_present_head_joint_positions should be overridden by subclasses."
        )

    def get_present_body_yaw(self) -> float:
        """Return the present body yaw."""
        yaw: float = self.get_present_head_joint_positions()[0]
        return yaw

    def get_present_head_pose(self) -> Annotated[NDArray[np.float64], (4, 4)]:
        """Return the present head pose as a 4x4 matrix."""
        assert self.current_head_pose is not None, (
            "The current head pose is not set. Please call the update_head_kinematics_model method first."
        )
        return self.current_head_pose

    def get_current_head_pose(self) -> Annotated[NDArray[np.float64], (4, 4)]:
        """Return the present head pose as a 4x4 matrix."""
        return self.get_present_head_pose()

    def get_present_antenna_joint_positions(
        self,
    ) -> Annotated[NDArray[np.float64], (2,)]:
        """Return the present antenna joint positions.

        This method is a placeholder and should be overridden by subclasses.
        """
        raise NotImplementedError(
            "The method get_present_antenna_joint_positions should be overridden by subclasses."
        )

    # Kinematics methods
    def update_head_kinematics_model(
        self,
        head_joint_positions: Annotated[NDArray[np.float64], (7,)] | None = None,
        antennas_joint_positions: Annotated[NDArray[np.float64], (2,)] | None = None,
    ) -> None:
        """Update the placo kinematics of the robot.

        Args:
            head_joint_positions (List[float] | None): The joint positions of the head.
            antennas_joint_positions (List[float] | None): The joint positions of the antennas.

        Returns:
            None: This method does not return anything.

        This method updates the head kinematics model with the given joint positions.
        - If the joint positions are not provided, it will use the current joint positions.
        - If the head joint positions have not changed, it will return without recomputing the forward kinematics.
        - If the head joint positions have changed, it will compute the forward kinematics to get the current head pose.
        - If the forward kinematics fails, it will raise an assertion error.
        - If the antennas joint positions are provided, it will update the current antenna joint positions.

        Note:
            This method will update the `current_head_pose` and `current_head_joint_positions`
            attributes of the backend instance with the computed values. And the `current_antenna_joint_positions` if provided.

        """
        if head_joint_positions is None:
            head_joint_positions = self.get_present_head_joint_positions()

        # Compute the forward kinematics to get the current head pose
        self.current_head_pose = self.head_kinematics.fk(head_joint_positions)

        # Check if the FK was successful
        assert self.current_head_pose is not None, (
            "FK failed to compute the current head pose."
        )

        # Store the last head joint positions
        self.current_head_joint_positions = head_joint_positions

        if antennas_joint_positions is not None:
            self.current_antenna_joint_positions = antennas_joint_positions

    def set_automatic_body_yaw(self, body_yaw: bool) -> None:
        """Set the automatic body yaw.

        Args:
            body_yaw (bool): The yaw angle of the body.

        """
        self.head_kinematics.set_automatic_body_yaw(automatic_body_yaw=body_yaw)

    def get_urdf(self) -> str:
        """Get the URDF representation of the robot."""
        urdf_path = Path(URDF_ROOT_PATH) / "robot.urdf"

        with open(urdf_path, "r") as f:
            return f.read()

    # Multimedia methods
    def play_sound(self, sound_file: str) -> None:
        """Play a sound file from the assets directory.

        Delegates to the media server's play_sound method.  If the server
        is not available (no_media mode), this is a no-op.

        Args:
            sound_file (str): The name of the sound file to play (e.g., "wake_up.wav").

        """
        if self._media_server is not None:
            self._media_server.play_sound(sound_file)

    def stop_sound(self) -> None:
        """Stop the currently playing sound file.

        Delegates to the media server's stop_sound method.  If the server
        is not available (no_media mode), this is a no-op.
        """
        if self._media_server is not None:
            self._media_server.stop_sound()

    def clear_incoming_audio(self) -> None:
        """Flush incoming WebRTC audio queued for the speaker (barge-in).

        Delegates to the media server.  If the server is not available
        (no_media mode), this is a no-op.
        """
        if self._media_server is not None:
            self._media_server.clear_incoming_audio()

    # Basic move definitions
    INIT_HEAD_POSE = np.eye(4)

    SLEEP_HEAD_JOINT_POSITIONS = [
        0,
        -0.9848156658225817,
        1.2624661884298831,
        -0.24390294527381684,
        0.20555342557667577,
        -1.2363885150358267,
        1.0032234352772091,
    ]

    INIT_ANTENNAS_JOINT_POSITIONS = np.array(
        (-0.1745, 0.1745)
    )  # ~10° offset to reduce shaking at vertical
    SLEEP_ANTENNAS_JOINT_POSITIONS = np.array((-3.05, 3.05))
    SLEEP_HEAD_POSE = np.array(
        [
            [0.911, 0.004, 0.413, -0.021],
            [-0.004, 1.0, -0.001, 0.001],
            [-0.413, -0.001, 0.911, -0.044],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    # Bounds the goto interpolation tail, not the hardware: play_move never
    # evaluates a move at exactly t=duration, so the last written target sits
    # a (velocity-eased) tick short of the commanded endpoint.
    INIT_TARGET_ATOL: float = 1e-2

    # Radius (magic units) around SLEEP_HEAD_POSE within which the robot counts
    # as already down. Shared by goto_sleep's "no travel needed" branch and
    # is_at_sleep_pose() so the trajectory and the skip can't drift apart.
    SLEEP_POSE_MAGIC_ATOL: float = 10.0

    def is_awake_at_init_pose(self) -> bool:
        """Motors on and init is the pose the controller is actively holding.

        Compares the commanded targets, not the measured pose: a head sagging
        a few mm under its own weight while commanded to init IS standing at
        init, and per-unit calibration residual must not flip the answer.
        """
        if self.target_head_pose is None or self.target_antenna_joint_positions is None:
            return False
        return (
            self.get_motor_control_mode() == MotorControlMode.Enabled
            and bool(
                np.allclose(
                    self.target_head_pose,
                    self.INIT_HEAD_POSE,
                    atol=self.INIT_TARGET_ATOL,
                )
            )
            and bool(
                np.allclose(
                    self.target_antenna_joint_positions,
                    self.INIT_ANTENNAS_JOINT_POSITIONS,
                    atol=self.INIT_TARGET_ATOL,
                )
            )
        )

    def is_at_sleep_pose(self) -> bool:
        """Head physically resting within the sleep-pose radius.

        Reads the measured pose, not the commanded target: a limp robot has no
        meaningful target, and the whole point is to tell a robot parked down by
        a clean leave sequence from one left limp mid-pose by a crashed app.
        """
        if self.current_head_pose is None:
            return False
        _, _, dist_to_sleep_pose = distance_between_poses(
            self.get_current_head_pose(), self.SLEEP_HEAD_POSE
        )
        return bool(dist_to_sleep_pose <= self.SLEEP_POSE_MAGIC_ATOL)

    async def wake_up(self, *, force: bool = False) -> None:
        """Wake up the robot - go to the initial head position and play the wake up emote and sound.

        Stands down (silently - no motion, no sound) when the robot is already
        awake at the init pose, so a boot-time ``wake_up`` after a session
        handoff doesn't replay the emote. The wake-completed callback still
        fires either way: the robot IS awake. ``force=True`` bypasses the
        stand-down, e.g. for a deliberate replay or a caller that just
        enabled the motors itself and knows a real wake is due.
        """
        await asyncio.sleep(0.1)

        if not force and self.is_awake_at_init_pose():
            if self._on_wake_up_callback is not None:
                self._on_wake_up_callback()
            return

        _, _, magic_distance = distance_between_poses(
            self.get_current_head_pose(), self.INIT_HEAD_POSE
        )
        await self.goto_target(
            self.INIT_HEAD_POSE,
            antennas=self.INIT_ANTENNAS_JOINT_POSITIONS,
            duration=magic_distance * 20 / 1000,  # ms_per_magic_mm = 10
        )
        await asyncio.sleep(0.1)

        # Toudoum
        self.play_sound("wake_up.wav")

        # Roll 20° to the left
        pose = self.INIT_HEAD_POSE.copy()
        pose[:3, :3] = R.from_euler("xyz", [20, 0, 0], degrees=True).as_matrix()
        await self.goto_target(pose, duration=0.2)

        # Go back to the initial position
        await self.goto_target(self.INIT_HEAD_POSE, duration=0.2)

        if self._on_wake_up_callback is not None:
            self._on_wake_up_callback()

    async def goto_sleep(self) -> None:
        """Put the robot to sleep by moving the head and antennas to a predefined sleep position.

        - If we are already very close to the sleep position, we do nothing.
        - If we are far from the sleep position:
            - If we are far from the initial position, we move there first.
            - If we are close to the initial position, we move directly to the sleep position.
        """
        self._quiesce_aim_sources()

        # Magic units
        _, _, dist_to_sleep_pose = distance_between_poses(
            self.get_current_head_pose(), self.SLEEP_HEAD_POSE
        )
        _, _, dist_to_init_pose = distance_between_poses(
            self.get_current_head_pose(), self.INIT_HEAD_POSE
        )
        sleep_time = 2.0

        # Thresholds found empirically.
        if dist_to_sleep_pose > self.SLEEP_POSE_MAGIC_ATOL:
            if dist_to_init_pose > 30:
                # Move to the initial position
                await self.goto_target(
                    self.INIT_HEAD_POSE,
                    antennas=self.INIT_ANTENNAS_JOINT_POSITIONS,
                    duration=1,
                )
                await asyncio.sleep(0.2)

            self.play_sound("go_sleep.wav")

            # Move to the sleep position
            await self.goto_target(
                self.SLEEP_HEAD_POSE,
                antennas=self.SLEEP_ANTENNAS_JOINT_POSITIONS,
                duration=2,
            )
        else:
            # The sound doesn't play fully if we don't wait enough
            self.play_sound("go_sleep.wav")
            sleep_time += 3

        self._last_head_pose = self.SLEEP_HEAD_POSE
        await asyncio.sleep(sleep_time)

        # Rest limp at the sleep pose, like a fresh boot.
        self.set_motor_control_mode(MotorControlMode.Disabled)

    def _quiesce_aim_sources(self) -> None:
        """Silence every secondary head-aim source before a scripted move.

        Wobbling and leftover speech offsets are added on top of the commanded
        target, and head tracking is a primary aim source in its own right, so
        any of them left running would fight the trajectory.
        """
        if self._media_server is not None:
            self._media_server.disable_wobbling()
        self.set_speech_offsets((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        self.disable_head_tracking()

    async def reset_to_sleep(self) -> None:
        """Put the robot to sleep from ANY motor state the last app left behind.

        ``goto_sleep()`` assumes it inherits a robot under position control.
        When that assumption is broken it degrades silently instead of failing:
        under gravity compensation the control loop ignores position targets
        outright, so the whole trajectory is written into the void and the final
        torque cut drops the head from wherever it happened to be. Motors left
        limp - globally or per-motor via ``set_torque(ids=...)`` - are just as
        bad, and none of these paths are hypothetical: a crashed app, a killed
        tab or a dropped Wi-Fi link all skip the client's own cleanup.

        So re-establish the assumption first, then reuse the normal sequence:
        torque on (``enable_motors`` pins every target to the measured pose, so
        nothing snaps), lift to the init pose, then ``goto_sleep()``, which ends
        limp at the sleep pose. The lift is what makes the result predictable -
        it collapses every possible inherited pose into the one starting point
        the sleep trajectory was tuned for.
        """
        # Ordered: quiesce first so tracking can't re-aim the head between the
        # torque coming back and the lift starting.
        self._quiesce_aim_sources()
        self.set_motor_control_mode(MotorControlMode.Enabled)

        _, _, magic_distance = distance_between_poses(
            self.get_current_head_pose(), self.INIT_HEAD_POSE
        )
        await self.goto_target(
            self.INIT_HEAD_POSE,
            antennas=self.INIT_ANTENNAS_JOINT_POSITIONS,
            # Same magic-unit scaling as wake_up, floored so a head already at
            # init still gets a real (if brief) move rather than a zero-length
            # one, and capped so a fully drooped head doesn't crawl back up.
            duration=float(np.clip(magic_distance * 20 / 1000, 0.3, 1.5)),
        )
        await asyncio.sleep(0.2)

        await self.goto_sleep()

    # Motor control modes
    @abstractmethod
    def get_motor_control_mode(self) -> MotorControlMode:
        """Get the motor control mode."""
        pass

    @abstractmethod
    def set_motor_control_mode(self, mode: MotorControlMode) -> None:
        """Set the motor control mode."""
        pass

    @abstractmethod
    def set_motor_torque_ids(self, ids: list[str], on: bool) -> None:
        """Set the motor torque for specific motor names."""
        pass

    def write_raw_packet(self, packet: bytes) -> bytes:
        """Write a raw packet to the motor controller and return the response.

        Args:
            packet (bytes): The raw packet to send to the motor controller.

        Returns:
            bytes: The raw response packet from the motor controller.

        """
        raise NotImplementedError(
            "The method write_raw_packet is only available for the real robot backend."
        )

    def get_present_passive_joint_positions(self) -> Optional[Dict[str, float]]:
        """Get the present passive joint positions.

        Requires the Placo kinematics engine.
        """
        # This is would be better, and fix mypy issues, but Placo is dynamically imported
        # if not isinstance(self.head_kinematics, PlacoKinematics):
        if self.kinematics_engine != "Placo":
            return None
        return {
            "passive_1_x": self.head_kinematics.get_joint("passive_1_x"),  # type: ignore [union-attr]
            "passive_1_y": self.head_kinematics.get_joint("passive_1_y"),  # type: ignore [union-attr]
            "passive_1_z": self.head_kinematics.get_joint("passive_1_z"),  # type: ignore [union-attr]
            "passive_2_x": self.head_kinematics.get_joint("passive_2_x"),  # type: ignore [union-attr]
            "passive_2_y": self.head_kinematics.get_joint("passive_2_y"),  # type: ignore [union-attr]
            "passive_2_z": self.head_kinematics.get_joint("passive_2_z"),  # type: ignore [union-attr]
            "passive_3_x": self.head_kinematics.get_joint("passive_3_x"),  # type: ignore [union-attr]
            "passive_3_y": self.head_kinematics.get_joint("passive_3_y"),  # type: ignore [union-attr]
            "passive_3_z": self.head_kinematics.get_joint("passive_3_z"),  # type: ignore [union-attr]
            "passive_4_x": self.head_kinematics.get_joint("passive_4_x"),  # type: ignore [union-attr]
            "passive_4_y": self.head_kinematics.get_joint("passive_4_y"),  # type: ignore [union-attr]
            "passive_4_z": self.head_kinematics.get_joint("passive_4_z"),  # type: ignore [union-attr]
            "passive_5_x": self.head_kinematics.get_joint("passive_5_x"),  # type: ignore [union-attr]
            "passive_5_y": self.head_kinematics.get_joint("passive_5_y"),  # type: ignore [union-attr]
            "passive_5_z": self.head_kinematics.get_joint("passive_5_z"),  # type: ignore [union-attr]
            "passive_6_x": self.head_kinematics.get_joint("passive_6_x"),  # type: ignore [union-attr]
            "passive_6_y": self.head_kinematics.get_joint("passive_6_y"),  # type: ignore [union-attr]
            "passive_6_z": self.head_kinematics.get_joint("passive_6_z"),  # type: ignore [union-attr]
            "passive_7_x": self.head_kinematics.get_joint("passive_7_x"),  # type: ignore [union-attr]
            "passive_7_y": self.head_kinematics.get_joint("passive_7_y"),  # type: ignore [union-attr]
            "passive_7_z": self.head_kinematics.get_joint("passive_7_z"),  # type: ignore [union-attr]
        }

    # ------------------------------------------------------------------
    # Direction of Arrival (cached, off the hot state path)
    # ------------------------------------------------------------------

    # ~10 Hz: fast enough to track a speaker turning, slow enough to keep the
    # blocking ReSpeaker USB reads off the ~30 Hz pose-push loop.
    DOA_POLL_INTERVAL_S = 0.1
    # The poller exits after this long without a cache consumer. DoA rides
    # the state snapshot, so "demand" means any connected client reading
    # state; the gain is that an idle daemon (no client at all - the common
    # state for an always-on robot) generates zero USB traffic. Must stay
    # comfortably above any plausible REST polling cadence: the idle exit
    # clears the cache, so a client polling slower than this would only
    # ever see the null it re-primed on its previous request.
    DOA_IDLE_STOP_S = 60.0

    def _note_doa_demand(self) -> None:
        """Record cache demand and make sure the poller is running.

        Spawning is guarded by ``_doa_lock`` so the ~30 Hz push loop and
        concurrent REST requests can't start two threads.
        """
        now = time.monotonic()
        with self._doa_lock:
            self._doa_demand_ts = now
            if self._doa_thread is not None and self._doa_thread.is_alive():
                return
            self._doa_thread = threading.Thread(
                target=self._doa_poll_loop, name="doa-poller", daemon=True
            )
            self._doa_thread.start()

    def _doa_poll_loop(self) -> None:
        """Cache the latest DoA reading until shutdown or idle timeout.

        ``get_DoA()`` is a blocking USB control transfer with no bounded
        latency (retry handshake of up to 100 attempts, 100 s worst-case
        transfer timeout), so it must never run on the ~30 Hz pose-push
        path or the HTTP event loop. This thread is the only place the
        daemon reads DoA; every consumer goes through the cache (see
        :meth:`read_doa`). Reads hold ``_doa_usb_lock`` so they never
        interleave with the audio-config commands' own ReSpeaker
        conversation, and every failure path is swallowed so a flaky USB
        read can never take the thread down.
        """
        while not self._doa_stop.is_set():
            with self._doa_lock:
                idle_for = time.monotonic() - self._doa_demand_ts
            if idle_for > self.DOA_IDLE_STOP_S:
                with self._doa_lock:
                    # Never serve a pre-idle reading to a late consumer.
                    self._last_doa = None
                return
            try:
                with self._doa_usb_lock:
                    reading = self.doa.get_DoA() if self.doa is not None else None
            except Exception:  # noqa: BLE001 - never kill the poller
                reading = None
            with self._doa_lock:
                self._last_doa = reading
            # `wait()` returns True once stopped, False on timeout - so this
            # both paces the poll and exits promptly on shutdown.
            if self._doa_stop.wait(self.DOA_POLL_INTERVAL_S):
                return

    def read_doa(self) -> DoaSnapshot | None:
        """Return the latest cached DoA reading, or ``None``.

        Never blocks: serves both the ~30 Hz pose push (via
        :meth:`build_state_snapshot`) and the REST surface (`/state/doa`,
        `/state/full?with_doa=true`). Registers demand so the poller
        (re)starts, and returns ``None`` when no ReSpeaker is present or
        the first poll hasn't landed yet (~one ``DOA_POLL_INTERVAL_S``
        after the first demand).
        """
        if self.doa is None or not self.doa.available:
            return None
        self._note_doa_demand()
        with self._doa_lock:
            reading = self._last_doa
        if reading is None:
            return None
        angle, speech_detected = reading
        return DoaSnapshot(angle=angle, speech_detected=speech_detected)

    # ------------------------------------------------------------------
    # State snapshot (shared by get_state replies and the pose push)
    # ------------------------------------------------------------------

    def build_state_snapshot(self) -> StateSnapshot:
        """Build the present-state snapshot sent to clients.

        Single source of truth for both the polled ``get_state`` reply and
        the pushed pose stream (see :meth:`build_state_json`). All getters
        read cached motor values (``get_last_position``), so this is cheap
        and safe to call from a thread other than the motor loop.
        """
        return StateSnapshot(
            head_pose=self.get_present_head_pose().tolist()
            if self.current_head_pose is not None
            else None,
            antennas=self.get_present_antenna_joint_positions().tolist()
            if self.current_antenna_joint_positions is not None
            else None,
            # Per-motor head values (7, body yaw at [0]) so WebRTC clients
            # can check the physical pose motor by motor without the LAN
            # /ws/sdk stream. The antennas need no twin field: `antennas`
            # already is the two motor values.
            head_joint_positions=self.get_present_head_joint_positions().tolist()
            if self.current_head_joint_positions is not None
            else None,
            body_yaw=self.get_present_body_yaw(),
            motor_mode=self.get_motor_control_mode(),
            is_recording=self.is_recording,
            is_move_running=self.is_move_running,
            face_target=self.get_tracked_face(),
            # Sound Direction of Arrival (ReSpeaker mic array), or None when
            # unavailable. Cached, never blocks (see _doa_poll_loop).
            doa=self.read_doa(),
        )

    def build_state_dict(self) -> dict[str, Any]:
        """JSON-safe dict view of :meth:`build_state_snapshot`.

        Used by the ``get_state`` reply, whose envelope is assembled by the
        transport as a plain dict.
        """
        return self.build_state_snapshot().model_dump(mode="json")

    def build_state_json(self) -> Optional[str]:
        """Serialize the present state as the client-facing envelope.

        Returns the ``{"state": ..., "seq": ...}`` payload, or ``None`` if
        it can't be built yet. Wired into the media server as the
        pose-stream provider (``set_pose_provider``) so the daemon can
        *push* pose over the unreliable/unordered ``pose`` data channel at
        a steady rate instead of the client round-tripping ``get_state``
        over Wi-Fi. Returning ``None`` (e.g. before the first kinematics
        update, or during shutdown) simply skips that tick.
        """
        try:
            snapshot = self.build_state_snapshot()
        except Exception:
            return None
        self._pose_seq += 1
        return PoseFrame(state=snapshot, seq=self._pose_seq).model_dump_json()

    # ------------------------------------------------------------------
    # Transport-agnostic command processing
    # ------------------------------------------------------------------

    def process_command(
        self,
        cmd: AnyCommand,
        send_response: Callable[[dict[str, Any]], None],
        peer_id: Optional[str] = None,
    ) -> None:
        """Process a command from any transport (WebRTC data channel, WebSocket, ...).

        Args:
            cmd: A validated command model (parsed via command_adapter).
            send_response: Callback to send a response dict back to the caller.
            peer_id: Optional caller identity for transports that multiplex
                multiple peers on a single backend (WebRTC). Used to scope
                per-peer state such as the journalctl log subscription so
                two peers can each have their own active stream.
                Non-multiplexed transports (HTTP/WebSocket) leave it as
                ``None`` — log subscription requires a peer id and quietly
                no-ops without one.

        """
        block_targets = self.is_move_running

        def _maybe_ignore(field: str) -> bool:
            if not block_targets:
                return False
            self.logger.warning(
                f"Ignoring {field} command: a move is currently running."
            )
            return True

        if isinstance(cmd, SetTargetCmd):
            if not _maybe_ignore("set_target"):
                self.set_target_head_pose(np.array(cmd.head).reshape(4, 4))
            send_response({"status": "ok", "command": "set_target"})

        elif isinstance(cmd, SetHeadJointsCmd):
            if not _maybe_ignore("set_head_joints"):
                self.set_target_head_joint_positions(np.array(cmd.joints))
            send_response({"status": "ok", "command": "set_head_joints"})

        elif isinstance(cmd, SetBodyYawCmd):
            if not _maybe_ignore("set_body_yaw"):
                self.set_target_body_yaw(cmd.body_yaw)
            send_response({"status": "ok", "command": "set_body_yaw"})

        elif isinstance(cmd, SetAntennasCmd):
            if not _maybe_ignore("set_antennas"):
                self.set_target_antenna_joint_positions(np.array(cmd.antennas))
            send_response({"status": "ok", "command": "set_antennas"})

        elif isinstance(cmd, SetFullTargetCmd):
            if not _maybe_ignore("set_full_target"):
                if cmd.head is not None:
                    self.set_target_head_pose(np.array(cmd.head).reshape(4, 4))
                if cmd.body_yaw is not None:
                    self.set_target_body_yaw(cmd.body_yaw)
                if cmd.antennas is not None:
                    self.set_target_antenna_joint_positions(np.array(cmd.antennas))
            send_response({"status": "ok", "command": "set_full_target"})

        elif isinstance(cmd, GotoTargetCmd):
            head = np.array(cmd.head).reshape(4, 4) if cmd.head else None
            antennas = np.array(cmd.antennas) if cmd.antennas else None
            asyncio.create_task(
                self._async_goto(
                    send_response, head, antennas, cmd.duration, cmd.body_yaw
                )
            )

        elif isinstance(cmd, WakeUpCmd):
            asyncio.create_task(self._async_wake_up(send_response))

        elif isinstance(cmd, GotoSleepCmd):
            asyncio.create_task(self._async_goto_sleep(send_response))

        elif isinstance(cmd, PlaySoundCmd):
            self.play_sound(cmd.file)
            send_response({"status": "ok", "command": "play_sound"})

        elif isinstance(cmd, PlayRecordedMoveCmd):
            asyncio.create_task(self._async_play_recorded_move(cmd, send_response))

        elif isinstance(cmd, PreloadDatasetCmd):
            asyncio.create_task(self._async_preload_dataset(cmd, send_response))

        elif isinstance(cmd, ClearIncomingAudioCmd):
            self.clear_incoming_audio()
            send_response({"status": "ok", "command": "clear_incoming_audio"})

        elif isinstance(cmd, SetSpeechOffsetsCmd):
            offsets = cmd.offsets
            if len(offsets) == 6:
                self.set_speech_offsets(
                    (
                        offsets[0],
                        offsets[1],
                        offsets[2],
                        offsets[3],
                        offsets[4],
                        offsets[5],
                    )
                )
            send_response({"status": "ok", "command": "set_speech_offsets"})

        elif isinstance(cmd, SetWobblingCmd):
            if self._media_server is not None:
                if cmd.enabled:
                    self._media_server.enable_wobbling(self.set_speech_offsets)
                else:
                    self._media_server.disable_wobbling()
                    self.set_speech_offsets((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            send_response({"status": "ok", "command": "set_wobbling"})

        elif isinstance(cmd, SetHeadTrackingCmd):
            if cmd.enabled:
                enabled = self.enable_head_tracking(
                    weight=cmd.weight,
                    mode=cmd.mode,
                    glance_interval_s=cmd.glance_interval_s,
                    speaking_hold_s=cmd.speaking_hold_s,
                )
                send_response(
                    {
                        "status": "ok" if enabled else "unavailable",
                        "command": "set_head_tracking",
                        "enabled": enabled,
                    }
                )
            else:
                self.disable_head_tracking()
                send_response(
                    {
                        "status": "ok",
                        "command": "set_head_tracking",
                        "enabled": False,
                    }
                )

        elif isinstance(cmd, GetTrackedFaceCmd):
            send_response(
                {
                    "command": "get_tracked_face",
                    "face_target": self.get_tracked_face().model_dump(),
                }
            )

        elif isinstance(cmd, SetMotorModeCmd):
            self.set_motor_control_mode(MotorControlMode(cmd.mode))
            send_response({"motor_mode": cmd.mode, "status": "ok"})

        elif isinstance(cmd, SetTorqueCmd):
            if cmd.ids is not None:
                self.set_motor_torque_ids(cmd.ids, cmd.on)
            elif cmd.on:
                self.set_motor_control_mode(MotorControlMode.Enabled)
            else:
                self.set_motor_control_mode(MotorControlMode.Disabled)
            send_response({"status": "ok", "command": "set_torque"})

        elif isinstance(cmd, GetMotorModeCmd):
            send_response({"motor_mode": self.get_motor_control_mode().value})

        elif isinstance(cmd, SetGravityCompensationCmd):
            try:
                if cmd.enabled:
                    self.set_motor_control_mode(MotorControlMode.GravityCompensation)
                else:
                    self.set_motor_control_mode(MotorControlMode.Enabled)
            except ValueError as e:
                send_response({"error": str(e), "command": "set_gravity_compensation"})
                return
            send_response({"status": "ok", "command": "set_gravity_compensation"})

        elif isinstance(cmd, SetAutomaticBodyYawCmd):
            self.set_automatic_body_yaw(cmd.enabled)
            send_response({"status": "ok", "command": "set_automatic_body_yaw"})

        elif isinstance(cmd, GetStateCmd):
            send_response({"state": self.build_state_dict()})

        elif isinstance(cmd, GetVersionCmd):
            from importlib.metadata import version

            send_response({"version": version("reachy_mini")})

        elif isinstance(cmd, GetHardwareIdCmd):
            from reachy_mini.utils.hardware_id import get_hardware_id

            send_response({"hardware_id": get_hardware_id()})

        elif isinstance(cmd, GetImuCmd):
            # `imu` is None on IMU-less hardware (Lite, simulation) so
            # clients can tell "no IMU" from "no reply" (old daemon).
            imu = self.get_imu_data()
            payload = imu.model_dump(exclude={"type"}) if imu is not None else None
            send_response({"command": "get_imu", "imu": payload})

        elif isinstance(cmd, (GetRobotNameCmd, SetRobotNameCmd)):
            # Robot display name is a persistent, robot-wide string stored on
            # disk. A rename is applied live below (status + central relay +
            # mDNS) via the set-robot-name callback, so no daemon restart is
            # needed; the persisted value also overrides the --robot-name
            # default on the next start.
            from reachy_mini.utils.robot_name import get_robot_name, set_robot_name

            if isinstance(cmd, SetRobotNameCmd):
                stored = set_robot_name(cmd.name)
                # Apply the rename live (status + central relay + mDNS) so it
                # takes effect without a daemon restart. Fail-safe: a wiring
                # error here must not break the persisted rename or the ack.
                if stored is not None and self._set_robot_name_callback is not None:
                    try:
                        self._set_robot_name_callback(stored)
                    except Exception as e:  # noqa: BLE001 - never break the cmd loop
                        self.logger.warning(f"set_robot_name live-apply failed: {e}")
                send_response(
                    {
                        "command": "set_robot_name",
                        "status": "ok" if stored is not None else "error",
                        "name": stored if stored is not None else get_robot_name(),
                    }
                )
            else:  # GetRobotNameCmd
                send_response(
                    {
                        "command": "get_robot_name",
                        "name": get_robot_name(),
                    }
                )

        elif isinstance(cmd, DeleteHfTokenCmd):
            # Sign the robot out of Hugging Face: clears the daemon's stored
            # token and notifies the central relay (drops it to
            # WAITING_FOR_TOKEN), so the robot de-registers and disappears
            # from its owner's list until it is set up again. Fail-safe:
            # delete_hf_token() never raises (returns False on failure), so
            # a storage/logout error can't break the command loop.
            from reachy_mini.apps.sources.hf_auth import delete_hf_token

            ok = delete_hf_token()
            send_response(
                {
                    "command": "delete_hf_token",
                    "status": "ok" if ok else "error",
                }
            )

        elif isinstance(cmd, (GetFirstWakeUpCmd, SetFirstWakeUpCmd)):
            # First wake-up wizard completion is a persistent, robot-wide
            # flag stored in the daemon config file (next to startup_app &
            # co) so the post-connection setup wizard only ever shows once,
            # whichever client connects. Read/write helpers are fail-safe
            # (never raise) so a storage error can't break the command loop.
            from reachy_mini.daemon.startup_app_config import (
                get_first_wake_up_completed,
                set_first_wake_up_completed,
            )

            if isinstance(cmd, SetFirstWakeUpCmd):
                ok = set_first_wake_up_completed(cmd.is_completed)
                send_response(
                    {
                        "command": "set_first_wake_up",
                        "status": "ok" if ok else "error",
                        "is_completed": cmd.is_completed
                        if ok
                        else get_first_wake_up_completed(),
                    }
                )
            else:  # GetFirstWakeUpCmd
                send_response(
                    {
                        "command": "get_first_wake_up",
                        "is_completed": get_first_wake_up_completed(),
                    }
                )

        elif isinstance(
            cmd,
            (
                SetVolumeCmd,
                GetVolumeCmd,
                SetMicrophoneVolumeCmd,
                GetMicrophoneVolumeCmd,
            ),
        ):
            # Volume is a global robot setting, not per-session: a remote
            # change persists for the next connection. This matches the
            # semantics of the local REST /api/volume endpoints, which
            # share the same VolumeControl singleton.
            from reachy_mini.daemon.app.routers.volume_control import (
                get_volume_control,
            )

            try:
                vc = get_volume_control()
            except Exception as e:
                # Unsupported platform or audio stack down — don't crash
                # the command loop, just report failure to the caller.
                self.logger.warning("Volume command failed (no control): %s", e)
                send_response(
                    {"error": f"Volume control unavailable: {e}", "command": cmd.type}
                )
            else:
                if isinstance(cmd, SetVolumeCmd):
                    ok = vc.set_output_volume(cmd.volume)
                    send_response(
                        {
                            "status": "ok" if ok else "error",
                            "command": "set_volume",
                            "volume": cmd.volume if ok else vc.get_output_volume(),
                        }
                    )
                elif isinstance(cmd, GetVolumeCmd):
                    send_response(
                        {"command": "get_volume", "volume": vc.get_output_volume()}
                    )
                elif isinstance(cmd, SetMicrophoneVolumeCmd):
                    ok = vc.set_input_volume(cmd.volume)
                    send_response(
                        {
                            "status": "ok" if ok else "error",
                            "command": "set_microphone_volume",
                            "volume": cmd.volume if ok else vc.get_input_volume(),
                        }
                    )
                else:  # GetMicrophoneVolumeCmd
                    send_response(
                        {
                            "command": "get_microphone_volume",
                            "volume": vc.get_input_volume(),
                        }
                    )

        elif isinstance(cmd, (ApplyAudioConfigCmd, ReadAudioParameterCmd)):
            from reachy_mini.media.audio_control_utils import init_respeaker_usb

            # This is a multi-transfer ReSpeaker conversation, so it holds
            # `_doa_usb_lock` end to end: it must never interleave with the
            # DoA poller (see _doa_poll_loop). Reuse the long-lived DoA
            # handle when the probe found a device; fall back to a transient
            # handle otherwise (no DoA handle also means no poller).
            with self._doa_usb_lock:
                transient = None
                if self.doa is not None and self.doa.available:
                    respeaker = self.doa.respeaker
                else:
                    try:
                        transient = init_respeaker_usb()
                    except Exception as e:
                        self.logger.warning("ReSpeaker init failed: %s", e)
                        send_response(
                            {
                                "error": f"ReSpeaker init failed: {e}",
                                "command": cmd.type,
                            }
                        )
                        return
                    respeaker = transient

                if respeaker is None:
                    send_response(
                        {
                            "error": "ReSpeaker audio board not available",
                            "command": cmd.type,
                        }
                    )
                    return

                try:
                    if isinstance(cmd, ApplyAudioConfigCmd):
                        config = [(p.name, p.values) for p in cmd.config]
                        applied = respeaker.apply_audio_config(
                            config, verify=cmd.verify
                        )
                        send_response(
                            {
                                "status": "ok" if applied else "error",
                                "command": "apply_audio_config",
                                "applied": applied,
                            }
                        )
                    else:  # ReadAudioParameterCmd
                        values = respeaker.read_values(cmd.name)
                        send_response(
                            {
                                "command": "read_audio_parameter",
                                "name": cmd.name,
                                "values": list(values) if values is not None else None,
                            }
                        )
                except Exception as e:
                    self.logger.warning(
                        "Audio config command %s failed: %s", cmd.type, e
                    )
                    send_response(
                        {
                            "error": f"Audio config command failed: {e}",
                            "command": cmd.type,
                        }
                    )
                finally:
                    if transient is not None:
                        transient.close()

        elif isinstance(cmd, StartRecordingCmd):
            self.start_recording()
            send_response(
                {"status": "ok", "command": "start_recording", "is_recording": True}
            )
        elif isinstance(cmd, StopRecordingCmd):
            self.stop_recording()
            send_response(
                {"status": "ok", "command": "stop_recording", "is_recording": False}
            )
        elif isinstance(cmd, AppendRecordCmd):
            self.append_record(cmd.record)
            send_response({"status": "ok", "command": "append_record"})

        elif isinstance(cmd, SubscribeLogsCmd):
            self._start_log_subscription(peer_id, send_response)
        elif isinstance(cmd, UnsubscribeLogsCmd):
            self._cancel_log_subscription(peer_id)

        elif isinstance(cmd, SubscribePoseCmd):
            self._set_pose_subscription(peer_id, True)
            send_response({"status": "ok", "command": "subscribe_pose"})
        elif isinstance(cmd, UnsubscribePoseCmd):
            self._set_pose_subscription(peer_id, False)
            send_response({"status": "ok", "command": "unsubscribe_pose"})

        elif isinstance(cmd, RestartDaemonCmd):
            # Ack BEFORE triggering the restart: the WebRTC transport
            # is torn down by `daemon.stop()` so any later send on
            # this peer's channel would silently drop. The callback
            # spawns its own thread and returns immediately.
            if self._restart_daemon_callback is None:
                send_response(
                    {
                        "error": "restart_daemon not supported by this backend host",
                        "command": "restart_daemon",
                    }
                )
                return
            send_response({"status": "ok", "command": "restart_daemon"})
            try:
                self._restart_daemon_callback()
            except Exception as e:
                self.logger.error(f"restart_daemon callback failed: {e}")

        elif isinstance(cmd, StartUpdateCmd):
            # Same fire-and-ack contract as `restart_daemon`: a successful
            # update ends with a `systemctl restart` that tears this
            # transport down. The callback runs cheap pre-checks (wireless
            # robot, an update is available, none already running)
            # synchronously and returns a refusal reason if it declined; we
            # only ack ok once the update job has actually been accepted, so
            # the consumer can surface a real error instead of waiting for a
            # reconnect that never comes.
            if self._start_update_callback is None:
                send_response(
                    {
                        "error": "start_update not supported by this backend host",
                        "command": "start_update",
                    }
                )
                return
            try:
                refusal = self._start_update_callback(cmd.pre_release)
            except Exception as e:
                self.logger.error(f"start_update callback failed: {e}")
                send_response(
                    {"error": f"start_update failed: {e}", "command": "start_update"}
                )
                return
            if refusal is not None:
                send_response({"error": refusal, "command": "start_update"})
                return
            send_response({"status": "ok", "command": "start_update"})

        elif isinstance(cmd, UploadMoveStartCmd):
            self._handle_upload_start(cmd)
        elif isinstance(cmd, UploadMoveChunkCmd):
            self._handle_upload_chunk(cmd)
        elif isinstance(cmd, UploadMoveFinishCmd):
            self._handle_upload_finish(cmd)
        elif isinstance(cmd, UploadAudioStartCmd):
            self._handle_audio_start(cmd)
        elif isinstance(cmd, UploadAudioChunkCmd):
            self._handle_audio_chunk(cmd)
        elif isinstance(cmd, UploadAudioFinishCmd):
            self._handle_audio_finish(cmd)
        elif isinstance(cmd, PlayUploadedMoveCmd):
            asyncio.create_task(self._async_play_uploaded_move(cmd))
        elif isinstance(cmd, CancelMoveCmd):
            self._handle_cancel_move(cmd)
        elif isinstance(cmd, StopMoveCmd):
            self._handle_stop_move(send_response)
        elif isinstance(cmd, PlayUploadedAudioCmd):
            self._handle_play_uploaded_audio(cmd)
        elif isinstance(cmd, CancelAudioCmd):
            self._handle_cancel_audio(cmd)

    # ------------------------------------------------------------------
    # Inline-move upload + daemon-side playback
    # ------------------------------------------------------------------

    def _evict_stale_uploads(self) -> None:
        """Drop in-progress upload slots older than ``_upload_ttl_s``.

        Called opportunistically on every new upload_move_start so we
        never accumulate orphans (client crashed mid-upload, dropped
        the data channel, etc.).
        """
        now = time.time()
        stale = [
            uid for uid, ts in self._upload_ts.items() if now - ts > self._upload_ttl_s
        ]
        for uid in stale:
            self._upload_chunks.pop(uid, None)
            self._upload_meta.pop(uid, None)
            self._upload_ts.pop(uid, None)
            self.logger.warning(f"upload_move: evicted stale slot {uid} (TTL exceeded)")

    # All upload_* handlers are fire-and-forget. The client pipelines
    # chunks at line rate (relying on SCTP's ordered, reliable delivery)
    # and the daemon silently drops failed slots. The eventual
    # play_uploaded_move broadcast will surface a "no such upload"
    # error if anything went wrong during upload.

    def _handle_upload_start(self, cmd: UploadMoveStartCmd) -> None:
        self._evict_stale_uploads()
        if len(self._upload_chunks) >= self._upload_max_active_slots:
            self.logger.warning(
                f"upload_move_start: refusing {cmd.upload_id}, too many active slots"
            )
            return
        # Sending start twice for the same id resets the slot, which
        # lets a client retry after a transient send failure without
        # needing to allocate a new id.
        self._upload_chunks[cmd.upload_id] = []
        self._upload_meta[cmd.upload_id] = {
            "total_chunks": cmd.total_chunks,
            "description": cmd.description,
            "estimated_duration_s": cmd.estimated_duration_s,
            "encoding": cmd.encoding,
        }
        self._upload_ts[cmd.upload_id] = time.time()

    def _handle_upload_chunk(self, cmd: UploadMoveChunkCmd) -> None:
        slot = self._upload_chunks.get(cmd.upload_id)
        meta = self._upload_meta.get(cmd.upload_id)
        if slot is None or meta is None:
            self.logger.warning(
                f"upload_move_chunk: no slot {cmd.upload_id}, dropping chunk {cmd.chunk_index}"
            )
            return
        expected_index = len(slot)
        if cmd.chunk_index != expected_index:
            # Out-of-order on an ordered SCTP transport means a client
            # bug. Drop the slot so the next start can recover cleanly.
            self.logger.warning(
                f"upload_move_chunk: out-of-order on {cmd.upload_id} "
                f"(expected {expected_index}, got {cmd.chunk_index}); dropping slot"
            )
            self._upload_chunks.pop(cmd.upload_id, None)
            self._upload_meta.pop(cmd.upload_id, None)
            self._upload_ts.pop(cmd.upload_id, None)
            return
        if cmd.chunk_index >= meta["total_chunks"]:
            self.logger.warning(
                f"upload_move_chunk: index {cmd.chunk_index} exceeds declared "
                f"total {meta['total_chunks']} for slot {cmd.upload_id}; dropping slot"
            )
            self._upload_chunks.pop(cmd.upload_id, None)
            self._upload_meta.pop(cmd.upload_id, None)
            self._upload_ts.pop(cmd.upload_id, None)
            return
        slot.append(cmd.chunk)
        self._upload_ts[cmd.upload_id] = time.time()

    def _evict_stale_audios(self) -> None:
        """Drop in-progress audio slots older than ``_upload_ttl_s``.

        Mirrors :meth:`_evict_stale_uploads` for the parallel audio
        path. Also nukes the on-disk WAV if it exists.
        """
        now = time.time()
        stale = [
            uid for uid, ts in self._audio_ts.items() if now - ts > self._upload_ttl_s
        ]
        for uid in stale:
            self._audio_chunks.pop(uid, None)
            self._audio_meta.pop(uid, None)
            self._audio_ts.pop(uid, None)
            self.logger.warning(
                f"upload_audio: evicted stale slot {uid} (TTL exceeded)"
            )
        # Also remove any orphaned finished audios past TTL: a client
        # may upload audio + never call play_uploaded_move.
        stale_files = []
        for uid, path in list(self._uploaded_audios.items()):
            try:
                age = time.time() - os.path.getmtime(path)
                if age > self._upload_ttl_s:
                    stale_files.append((uid, path))
            except OSError:
                stale_files.append((uid, path))
        for uid, path in stale_files:
            self._uploaded_audios.pop(uid, None)
            try:
                os.remove(path)
            except OSError:
                pass
            self.logger.warning(
                f"upload_audio: evicted orphaned audio {uid} (TTL exceeded)"
            )

    def _handle_audio_start(self, cmd: UploadAudioStartCmd) -> None:
        self._evict_stale_audios()
        if len(self._audio_chunks) >= self._upload_max_active_slots:
            self.logger.warning(
                f"upload_audio_start: refusing {cmd.upload_id}, too many active slots"
            )
            return
        # Restart-friendly: if the client retries with the same id we
        # just reset the slot.
        self._audio_chunks[cmd.upload_id] = []
        self._audio_meta[cmd.upload_id] = {
            "total_chunks": cmd.total_chunks,
            "encoding": cmd.encoding,
            "description": cmd.description,
        }
        self._audio_ts[cmd.upload_id] = time.time()

    def _handle_audio_chunk(self, cmd: UploadAudioChunkCmd) -> None:
        slot = self._audio_chunks.get(cmd.upload_id)
        meta = self._audio_meta.get(cmd.upload_id)
        if slot is None or meta is None:
            self.logger.warning(
                f"upload_audio_chunk: no slot {cmd.upload_id}, dropping chunk {cmd.chunk_index}"
            )
            return
        expected_index = len(slot)
        if cmd.chunk_index != expected_index:
            self.logger.warning(
                f"upload_audio_chunk: out-of-order on {cmd.upload_id} "
                f"(expected {expected_index}, got {cmd.chunk_index}); dropping slot"
            )
            self._audio_chunks.pop(cmd.upload_id, None)
            self._audio_meta.pop(cmd.upload_id, None)
            self._audio_ts.pop(cmd.upload_id, None)
            return
        if cmd.chunk_index >= meta["total_chunks"]:
            self.logger.warning(
                f"upload_audio_chunk: index {cmd.chunk_index} exceeds declared "
                f"total {meta['total_chunks']} for slot {cmd.upload_id}; dropping slot"
            )
            self._audio_chunks.pop(cmd.upload_id, None)
            self._audio_meta.pop(cmd.upload_id, None)
            self._audio_ts.pop(cmd.upload_id, None)
            return
        slot.append(cmd.chunk)
        self._audio_ts[cmd.upload_id] = time.time()

    def _handle_audio_finish(self, cmd: UploadAudioFinishCmd) -> None:
        slot = self._audio_chunks.pop(cmd.upload_id, None)
        meta = self._audio_meta.pop(cmd.upload_id, None)
        self._audio_ts.pop(cmd.upload_id, None)
        if slot is None or meta is None:
            self.logger.warning(f"upload_audio_finish: no such slot {cmd.upload_id}")
            return
        if len(slot) != meta["total_chunks"]:
            self.logger.warning(
                f"upload_audio_finish: chunk count mismatch on {cmd.upload_id} "
                f"(declared {meta['total_chunks']}, received {len(slot)})"
            )
            return
        payload = "".join(slot)
        try:
            import base64

            raw = base64.b64decode(payload, validate=False)
            os.makedirs(self._audio_temp_dir, exist_ok=True)
            # encoding "<container>-base64" → file extension; wav for legacy clients.
            ext = str(meta.get("encoding", "wav-base64")).split("-")[0] or "wav"
            path = os.path.join(self._audio_temp_dir, f"{cmd.upload_id}.{ext}")
            with open(path, "wb") as f:
                f.write(raw)
        except Exception as e:
            self.logger.warning(
                f"upload_audio_finish: write failed for slot {cmd.upload_id}: {e}"
            )
            return
        # If a previous audio was uploaded for this id, replace it.
        old = self._uploaded_audios.get(cmd.upload_id)
        if old and old != path:
            try:
                os.remove(old)
            except OSError:
                pass
        self._uploaded_audios[cmd.upload_id] = path
        self.logger.info(f"upload_audio_finish: stored {len(raw)} bytes at {path}")

    def _handle_play_uploaded_audio(self, cmd: PlayUploadedAudioCmd) -> None:
        """Play an uploaded audio standalone (no motion).

        Used during recording so the audio pipeline is identical to
        the eventual play_uploaded_move that will replay this audio.

        Broadcast ordering is deliberately symmetric to
        play_uploaded_move: broadcast STARTED first, then call
        play_sound.  Same order means the broadcast-vs-audio-at-speaker
        delta is identical in record and play, so the user's slider
        becomes a single per-robot constant (system latency, network
        RTT) rather than something that drifts between code paths.

        Records the upload_id on ``_active_audio_upload_id`` so the
        next ``cancel_audio`` knows which (if any) audio is live.
        """
        upload_id = cmd.upload_id
        audio_path = self._uploaded_audios.get(upload_id)
        if not audio_path:
            self.broadcast_to_all_clients(
                json.dumps(
                    {
                        "type": "play_uploaded_audio",
                        "upload_id": upload_id,
                        "error": "no such uploaded audio",
                    }
                )
            )
            return
        # Broadcast BEFORE play_sound, matching play_uploaded_move.
        self.broadcast_to_all_clients(
            json.dumps(
                {
                    "type": "play_uploaded_audio",
                    "upload_id": upload_id,
                    "started": True,
                }
            )
        )
        # Claim the active-audio slot before kicking off playback so a
        # cancel_audio arriving immediately after the start broadcast
        # finds the right id. Best-effort: GStreamer doesn't notify on
        # natural end, so this id may linger until either the next
        # play_uploaded_audio overwrites it or a cancel arrives.
        self._active_audio_upload_id = upload_id
        try:
            self.play_sound(audio_path)
        except Exception as e:
            self.logger.warning(f"play_uploaded_audio: play_sound failed: {e}")
            if self._active_audio_upload_id == upload_id:
                self._active_audio_upload_id = None
            # Broadcast a follow-up error so the client knows the
            # started event isn't actionable.
            self.broadcast_to_all_clients(
                json.dumps(
                    {
                        "type": "play_uploaded_audio",
                        "upload_id": upload_id,
                        "error": str(e),
                    }
                )
            )

    def _handle_cancel_audio(self, cmd: CancelAudioCmd) -> None:
        """Stop play_uploaded_audio iff its upload_id matches.

        Scoped so a stale cancel_audio against an already-finished id
        doesn't kill the audio attached to a play_uploaded_move that
        happens to be running. Idempotent.
        """
        active = self._active_audio_upload_id
        if active is None or active != cmd.upload_id:
            return
        self._active_audio_upload_id = None
        try:
            self.stop_sound()
        except Exception as e:
            self.logger.warning(f"cancel_audio: stop_sound failed: {e}")

    def _handle_cancel_move(self, cmd: CancelMoveCmd) -> None:
        """Cancel the active play_uploaded_move iff upload_id matches.

        Idempotent: a stale cancel against a no-longer-active id is a
        no-op, so two back-to-back plays can't cross-cancel each other.
        Acquires ``_play_move_lock`` so we never read a half-installed
        token from ``_async_play_uploaded_move``.
        """
        with self._play_move_lock:
            token = self._active_move_token
            if token is None or token.upload_id != cmd.upload_id:
                return
            token.cancelled = True

    def _handle_stop_move(
        self, send_response: Callable[[dict[str, Any]], None]
    ) -> None:
        """Stop whatever move is currently playing (recorded, uploaded, goto).

        The blunt client-facing counterpart of ``_handle_cancel_move``: no
        upload_id needed, it flips the global stop flag every ``play_move``
        loop polls, and also cancels the active uploaded-move token (if any)
        so that run's end broadcast reports ``cancelled`` instead of
        ``finished``. Idempotent: always acks ok, with ``stopped`` telling
        whether a move was actually interrupted.
        """
        if not self.is_move_running:
            send_response(
                {
                    "status": "ok",
                    "command": "stop_move",
                    "stopped": False,
                    "info": "no active move",
                }
            )
            return
        self._stop_move_requested = True
        # Also flip the per-run token (RLock is reentrant on the event-loop
        # thread, same as _handle_cancel_move) so an in-flight
        # play_uploaded_move bookkeeps this as a cancellation.
        with self._play_move_lock:
            token = self._active_move_token
            if token is not None:
                token.cancelled = True
        send_response({"status": "ok", "command": "stop_move", "stopped": True})

    def _handle_upload_finish(self, cmd: UploadMoveFinishCmd) -> None:
        slot = self._upload_chunks.pop(cmd.upload_id, None)
        meta = self._upload_meta.pop(cmd.upload_id, None)
        self._upload_ts.pop(cmd.upload_id, None)
        if slot is None or meta is None:
            self.logger.warning(f"upload_move_finish: no such slot {cmd.upload_id}")
            return
        if len(slot) != meta["total_chunks"]:
            self.logger.warning(
                f"upload_move_finish: chunk count mismatch on {cmd.upload_id} "
                f"(declared {meta['total_chunks']}, received {len(slot)})"
            )
            return
        payload = "".join(slot)
        encoding = meta.get("encoding", "json")
        # Cap on the assembled-then-decoded JSON size. Anything past
        # this means either a malicious gzip bomb (typical recorded
        # moves are < 10 MB JSON, < 5 MB gzipped) or a runaway
        # client; stop before allocating hundreds of MB on the CM4.
        max_decoded_bytes = 64 * 1024 * 1024
        try:
            if encoding == "gzip+base64":
                # Decompressing the assembled base64+gzip payload back
                # to UTF-8 JSON. Compute is small even for multi-MB
                # moves (gzip is fast on the CM4).
                import base64
                import gzip

                raw = base64.b64decode(payload, validate=False)
                # gzip.decompress reads the whole stream into RAM; use
                # GzipFile.read(max_decoded_bytes + 1) so a bomb can't
                # exhaust memory before we notice it's oversize.
                import io

                with gzip.GzipFile(fileobj=io.BytesIO(raw), mode="rb") as gz:
                    raw_text = gz.read(max_decoded_bytes + 1)
                if len(raw_text) > max_decoded_bytes:
                    self.logger.warning(
                        f"upload_move_finish: decoded payload exceeds "
                        f"{max_decoded_bytes} bytes on {cmd.upload_id}; dropping"
                    )
                    return
                payload_text = raw_text.decode("utf-8")
            elif encoding == "json":
                if len(payload) > max_decoded_bytes:
                    self.logger.warning(
                        f"upload_move_finish: payload exceeds "
                        f"{max_decoded_bytes} bytes on {cmd.upload_id}; dropping"
                    )
                    return
                payload_text = payload
            else:
                self.logger.warning(
                    f"upload_move_finish: unknown encoding {encoding!r} on {cmd.upload_id}"
                )
                return
            move_dict = json.loads(payload_text)
            # Reuse the RecordedMove parser; same JSON shape as the
            # HF dance/emotion datasets, no on-disk sound path.
            from reachy_mini.motion.recorded_move import RecordedMove

            parsed = RecordedMove(move_dict, sound_path=None)
        except Exception as e:
            self.logger.warning(
                f"upload_move_finish: parse failed for slot {cmd.upload_id}: {e}"
            )
            return
        self._uploaded_moves[cmd.upload_id] = parsed

    # ------------------------------------------------------------------
    # journalctl log streaming over the typed transport (subscribe_logs)
    # ------------------------------------------------------------------

    def _start_log_subscription(
        self,
        peer_id: Optional[str],
        send_response: Callable[[dict[str, Any]], None],
    ) -> None:
        if peer_id is None:
            send_response(
                LogStreamErrorMsg(
                    error="subscribe_logs requires a peer-aware transport"
                ).model_dump()
            )
            return
        # Cancel any pre-existing task for this peer; subscribing twice
        # is a no-op error rather than an exception so the consumer can
        # blindly call `subscribeLogs` on every reconnect.
        self._cancel_log_subscription(peer_id)
        task = asyncio.create_task(self._async_subscribe_logs(peer_id, send_response))
        self._log_tasks[peer_id] = task

    def _cancel_log_subscription(self, peer_id: Optional[str]) -> None:
        if peer_id is None:
            return
        task = self._log_tasks.pop(peer_id, None)
        if task is not None and not task.done():
            task.cancel()

    def _set_pose_subscription(self, peer_id: Optional[str], enabled: bool) -> None:
        """Enable/disable the pushed pose stream for a peer.

        Delegated to the media server, which owns the per-peer ``pose`` data
        channels and the push timer. No-op without a peer id (non-multiplexed
        transports) or on an older media server lacking the hook.
        """
        if peer_id is None or self._media_server is None:
            return
        if hasattr(self._media_server, "set_pose_subscription"):
            self._media_server.set_pose_subscription(peer_id, enabled)

    def _on_peer_disconnect(self, peer_id: str) -> None:
        """Cancel any per-peer state when the WebRTC peer goes away.

        Called from the media server's GStreamer/GLib thread, so we hop
        back onto the asyncio loop before touching task state.
        """
        loop = self._log_loop
        if loop is None or peer_id not in self._log_tasks:
            return
        loop.call_soon_threadsafe(self._cancel_log_subscription, peer_id)

    async def _async_subscribe_logs(
        self,
        peer_id: str,
        send_response: Callable[[dict[str, Any]], None],
    ) -> None:
        """Stream `journalctl -u reachy-mini-daemon` lines until cancelled.

        Mirrors the flags used by the WS route at
        ``daemon/app/routers/logs.py`` (and, modulo ``-n``, the BT
        pull-model path) so all three transports surface the same lines.
        """
        process: Optional[asyncio.subprocess.Process] = None
        try:
            try:
                process = await asyncio.create_subprocess_exec(
                    "journalctl",
                    "-u",
                    "reachy-mini-daemon",
                    "-b",
                    "-f",
                    "-n",
                    "100",
                    "--output",
                    "short-iso",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
            except FileNotFoundError:
                send_response(
                    LogStreamErrorMsg(error="journalctl not found").model_dump()
                )
                return

            assert process.stdout is not None
            while True:
                raw = await process.stdout.readline()
                if not raw:
                    # journalctl -f shouldn't EOF except on shutdown;
                    # surface it so the consumer can decide to retry.
                    send_response(
                        LogStreamErrorMsg(error="journalctl stream ended").model_dump()
                    )
                    return
                line = raw.decode("utf-8", errors="replace").rstrip("\n")
                # short-iso prefixes each line with an ISO-8601 timestamp
                # followed by a single space: split once so consumers
                # can render them separately without re-parsing.
                ts, sep, rest = line.partition(" ")
                if not sep:
                    ts, rest = "", line
                send_response(LogLineMsg(timestamp=ts, line=rest).model_dump())
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.logger.warning("[subscribe_logs] %s: %s", peer_id, e)
            try:
                send_response(LogStreamErrorMsg(error=str(e)).model_dump())
            except Exception:
                pass
        finally:
            if process is not None and process.returncode is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(process.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()

    async def _async_goto(
        self,
        send_response: Callable[[dict[str, Any]], None],
        head: Any,
        antennas: Any,
        duration: float,
        body_yaw: float | None,
    ) -> None:
        """Execute goto_target and send response when done."""
        try:
            await self.goto_target(
                head=head, antennas=antennas, duration=duration, body_yaw=body_yaw
            )
            send_response({"status": "ok", "command": "goto_target", "completed": True})
        except Exception as e:
            send_response({"error": str(e), "command": "goto_target"})

    async def _async_wake_up(
        self, send_response: Callable[[dict[str, Any]], None]
    ) -> None:
        """Execute wake_up and send response when done."""
        try:
            await self.wake_up()
            send_response({"status": "ok", "command": "wake_up", "completed": True})
        except Exception as e:
            send_response({"error": str(e), "command": "wake_up"})

    async def _async_goto_sleep(
        self, send_response: Callable[[dict[str, Any]], None]
    ) -> None:
        """Execute goto_sleep and send response when done."""
        try:
            await self.goto_sleep()
            send_response({"status": "ok", "command": "goto_sleep", "completed": True})
        except Exception as e:
            send_response({"error": str(e), "command": "goto_sleep"})

    async def _async_play_recorded_move(
        self,
        cmd: PlayRecordedMoveCmd,
        send_response: Callable[[dict[str, Any]], None],
    ) -> None:
        """Load a named move from a dataset and play it (motion + sound).

        Mirrors the ``/api/move/play/recorded-move-dataset`` route but over the
        data channel so it works on a remote WebRTC session. Fire-and-forget:
        the ack reports dispatch success/failure (unknown move, missing
        dataset), not playback completion. ``RecordedMoves`` reads the local
        HF cache first, so on a pre-downloaded robot this stays off the
        network.
        """
        from reachy_mini.motion.recorded_move import (
            DEFAULT_EMOTIONS_DATASET,
            RecordedMoves,
        )

        dataset = cmd.dataset_name or DEFAULT_EMOTIONS_DATASET
        try:
            # On a cold cache RecordedMoves() triggers a blocking
            # snapshot_download: run it in a worker thread so it cannot stall
            # the daemon's event loop (and its pose stream) mid-session.
            move = await asyncio.to_thread(
                lambda: RecordedMoves(dataset).get(cmd.move_name)
            )
        except Exception as e:
            self.logger.warning(
                f"play_recorded_move: {cmd.move_name!r} from {dataset!r} "
                f"failed to load: {e}"
            )
            send_response(
                {
                    "command": "play_recorded_move",
                    "status": "error",
                    "error": str(e),
                }
            )
            return

        # Dispatched: ack now (fire-and-forget), then run the move. play_move
        # self-guards against a concurrent move, so a double-tap is a no-op.
        send_response(
            {
                "command": "play_recorded_move",
                "status": "ok",
                "move_name": cmd.move_name,
            }
        )
        try:
            await self.play_move(move, initial_goto_duration=cmd.initial_goto_duration)
        except Exception as e:
            self.logger.warning(f"play_recorded_move: playback failed: {e}")

    async def _async_preload_dataset(
        self,
        cmd: PreloadDatasetCmd,
        send_response: Callable[[dict[str, Any]], None],
    ) -> None:
        """Warm the local HF cache for a recorded-move dataset.

        Runs ``snapshot_download`` (cache-first, so a no-op when already
        cached) in a worker thread: it does blocking network + disk IO and
        must not stall the daemon's event loop mid-session. The ack carries
        the cached path on success so clients can log/verify; a failure is
        reported but deliberately not fatal - ``play_recorded_move`` will
        retry the download on demand at playback time.
        """
        from reachy_mini.motion.recorded_move import preload_dataset

        local_path = await asyncio.to_thread(preload_dataset, cmd.dataset_name)
        if local_path is None:
            send_response(
                {
                    "command": "preload_dataset",
                    "status": "error",
                    "dataset_name": cmd.dataset_name,
                    "error": "download failed (see daemon logs)",
                }
            )
            return
        send_response(
            {
                "command": "preload_dataset",
                "status": "ok",
                "dataset_name": cmd.dataset_name,
                "local_path": local_path,
            }
        )

    async def _async_play_uploaded_move(self, cmd: PlayUploadedMoveCmd) -> None:
        """Run Backend.play_move on a previously-uploaded move slot.

        If an audio payload was uploaded with the same upload_id, it
        is attached as the move's sound_path so GStreamer playbin
        plays it on the robot speaker in lockstep with the motion
        loop (single-clock sync, no cross-network drift).

        Emits two unsolicited broadcast messages of type
        ``"play_uploaded_move"``: one when the playback loop actually
        starts (with ``has_audio`` indicating whether the daemon will
        also be emitting audio), one when it ends (``finished`` /
        ``cancelled`` / ``error``).  Clients filter by ``upload_id``.

        Going through broadcast (not per-call send_response) keeps the
        existing fire-and-forget semantics of process_command; the
        broadcast reaches both WS and WebRTC peers through the same
        path the daemon already uses for state messages.

        Cleans up the on-disk audio file after playback ends so the
        temp dir doesn't accumulate.
        """
        upload_id = cmd.upload_id
        move = self._uploaded_moves.pop(upload_id, None)
        audio_path = self._uploaded_audios.pop(upload_id, None)
        if move is None:
            # Don't leave the audio orphaned on disk if the move side
            # failed.
            if audio_path:
                try:
                    os.remove(audio_path)
                except OSError:
                    pass
            self.broadcast_to_all_clients(
                json.dumps(
                    {
                        "type": "play_uploaded_move",
                        "upload_id": upload_id,
                        "error": "no such uploaded move (upload first)",
                    }
                )
            )
            return

        # Attach the uploaded audio to the move so Backend.play_move
        # picks it up via the existing sound_path path.  RecordedMove
        # stores the path on a private attribute; we mutate it
        # directly because the slot is single-use and about to be
        # discarded.
        if audio_path:
            move._sound_path = audio_path

        # Broadcast start with the declared duration so any client
        # waiting on the started event knows the loop is live.
        self.broadcast_to_all_clients(
            json.dumps(
                {
                    "type": "play_uploaded_move",
                    "upload_id": upload_id,
                    "started": True,
                    "duration_s": move.duration,
                    "has_audio": audio_path is not None,
                }
            )
        )

        result: dict[str, Any] = {
            "type": "play_uploaded_move",
            "upload_id": upload_id,
            "has_audio": audio_path is not None,
        }
        # Install the per-run cancel token under the lock so a
        # cancel_move arriving in this exact window can't see a stale
        # (or absent) token. The token is removed in finally.
        token = _PlaybackCancelToken(upload_id)
        with self._play_move_lock:
            self._active_move_token = token
        try:
            await self.play_move(
                move,
                play_frequency=cmd.play_frequency,
                initial_goto_duration=cmd.initial_goto_duration,
                audio_lead_s=cmd.audio_lead_ms / 1000.0,
                cancel_token=token,
            )
            # play_move exits its loop in two cases: natural end or
            # cancel. The token preserves which one happened.
            if token.cancelled:
                result["cancelled"] = True
            else:
                result["finished"] = True
        except Exception as e:
            self.logger.exception(f"play_uploaded_move failed: {e}")
            result["error"] = str(e)
        finally:
            with self._play_move_lock:
                if self._active_move_token is token:
                    self._active_move_token = None
            # Always stop any sound that is still running and remove
            # the temp audio file.  Without stop_sound a cancelled
            # play would leave music playing until the WAV ends.
            if audio_path:
                try:
                    self.stop_sound()
                except Exception:
                    pass
                try:
                    os.remove(audio_path)
                except OSError:
                    pass
        try:
            self.broadcast_to_all_clients(json.dumps(result))
        except Exception as e:
            self.logger.warning(f"broadcast of play_uploaded_move end failed: {e}")

    # ------------------------------------------------------------------
    # Unsolicited broadcast (fan out to every connected client across
    # all transports). Used by the async play_uploaded_move handler to
    # report start / end events without going through the per-call
    # send_response (which is fire-and-forget on WS).
    # ------------------------------------------------------------------

    def set_ws_broadcast_callback(self, cb: Callable[[str], None]) -> None:
        """Register the WS broadcast hook. Called by WSServer.start()."""
        self._ws_broadcast_callback = cb

    def broadcast_to_all_clients(self, payload: str) -> None:
        """Send a JSON string to every connected client.

        Goes out on the WebRTC data channel (peer_id=None broadcasts)
        and on every WS client queue. Older clients that don't
        recognize the payload's type field just ignore it.
        """
        if self._send_message_to_webrtc is not None:
            try:
                self._send_message_to_webrtc(None, payload)
            except Exception as e:
                self.logger.warning(f"broadcast: WebRTC send failed: {e}")
        if self._ws_broadcast_callback is not None:
            try:
                self._ws_broadcast_callback(payload)
            except Exception as e:
                self.logger.warning(f"broadcast: WS broadcast failed: {e}")

    # ------------------------------------------------------------------
    # WebRTC data channel interface (delegates to process_command)
    # ------------------------------------------------------------------

    def set_restart_daemon_callback(self, callback: Callable[[], None]) -> None:
        """Wire the synchronous trigger used by the ``restart_daemon`` cmd.

        ``Daemon`` injects a callback that schedules ``daemon.restart()``
        on a fresh background thread (mirroring what
        ``bg_job_register.run_command`` does for the REST endpoint).
        The callback MUST return promptly so ``process_command`` can
        flush its ack on the about-to-be-torn-down DataChannel.
        """
        self._restart_daemon_callback = callback

    def set_start_update_callback(
        self, callback: Callable[[bool], Optional[str]]
    ) -> None:
        """Wire the trigger used by the ``start_update`` DataChannel cmd.

        ``Daemon`` injects a callback that runs ``update_reachy_mini`` on
        a fresh background thread (mirroring ``set_restart_daemon_callback``).
        Takes the ``pre_release`` flag and MUST return promptly so the ack
        is flushed before the update's ``systemctl restart`` tears the
        DataChannel down. It returns a refusal reason string when it declines
        the update (non-wireless robot, no update available, or one already
        running), or ``None`` once the job has been accepted.
        """
        self._start_update_callback = callback

    def set_on_wake_up_callback(self, callback: Callable[[], None]) -> None:
        """Wire a callback fired once each time ``wake_up()`` completes.

        ``Daemon`` injects this to auto-start a configured startup app after
        the robot wakes, regardless of how the wake was triggered. The callback
        runs inside the event loop, so it must return promptly (it schedules
        any async work as a task).
        """
        self._on_wake_up_callback = callback

    def set_robot_name_callback(self, callback: Callable[[str], None]) -> None:
        """Wire the live-apply hook for the ``set_robot_name`` DataChannel cmd.

        The app lifespan injects a callback that refreshes the advertised
        name in place (daemon status + central relay + mDNS) so a rename
        takes effect without a daemon restart. It receives the persisted
        (trimmed) name and MUST return promptly (any slow work is
        thread-offloaded on its side).
        """
        self._set_robot_name_callback = callback

    def setup_media_server(self, media_server: Any) -> None:
        """Connect the backend to the media server.

        Stores a reference to the ``GstMediaServer`` for:
        - WebRTC data channel message handling (robot control)
        - Sound playback delegation (play_sound)

        Args:
            media_server: The ``GstMediaServer`` instance.

        """
        self._media_server = media_server

        _loop = asyncio.new_event_loop()
        threading.Thread(target=_loop.run_forever, daemon=True).start()
        # Capture the loop so peer-disconnect cleanup (which fires on
        # gstreamer's GLib thread) can schedule task cancellation
        # threadsafely. Same loop used by `_handle_webrtc_message`.
        self._log_loop = _loop

        def _threadsafe_handler(peer_id: str, message: str) -> None:
            _loop.call_soon_threadsafe(self._handle_webrtc_message, peer_id, message)

        media_server.set_message_handler(_threadsafe_handler)
        # Ask the media server to notify us when a peer goes away so we
        # can cancel any pending journalctl subprocess for it. No-op
        # if the running media server doesn't support the hook (older
        # builds): log subscription cleanup just falls back to manual
        # `unsubscribe_logs` from the consumer side.
        if hasattr(media_server, "set_peer_disconnect_handler"):
            media_server.set_peer_disconnect_handler(self._on_peer_disconnect)
        self._send_message_to_webrtc = media_server.send_data_message
        # Feed the media server the present-state snapshot so it can *push*
        # pose over the unreliable/unordered `pose` data channel at a steady
        # rate (immune to Wi-Fi jitter), instead of the client polling
        # `get_state` on the reliable-ordered channel. No-op on older media
        # servers that don't expose the hook (falls back to polling).
        if hasattr(media_server, "set_pose_provider"):
            media_server.set_pose_provider(self.build_state_json)

    def set_jsonrpc_handler(
        self, handler: Callable[[str, Callable[[dict[str, Any]], None]], None]
    ) -> None:
        """Wire the JSON-RPC control handler (the daemon app relay)."""
        self._jsonrpc_handler = handler

    def _handle_webrtc_message(self, peer_id: str, message: str) -> None:
        # A fresh command means someone owns the robot again: cancel any pending
        # idle-reset goto_sleep so it doesn't fight the new session. Runs on the
        # same loop as the task, so the cancel is race-free.
        self._cancel_idle_reset()

        def send(resp: dict[str, Any]) -> None:
            self._send_webrtc_response(peer_id, resp)

        # JSON-RPC frames go to the app relay; legacy {"type": ...} commands
        # fall through to command_adapter. Both share the DataChannel.
        if self._jsonrpc_handler is not None:
            try:
                obj = json.loads(message)
            except (json.JSONDecodeError, ValueError):
                obj = None
            if looks_like_jsonrpc(obj):
                self._jsonrpc_handler(message, send)
                return

        try:
            cmd = command_adapter.validate_json(message)
        except Exception as e:
            self.logger.error(f"WebRTC invalid command: {e}")
            send({"error": f"Invalid command: {e}"})
            return
        try:
            self.process_command(cmd, send_response=send, peer_id=peer_id)
        except Exception as e:
            self.logger.error(f"WebRTC command error: {e}")
            send({"error": str(e)})

    def _send_webrtc_response(self, peer_id: str, response: dict[str, Any]) -> None:
        if self._send_message_to_webrtc:
            self._send_message_to_webrtc(peer_id, json.dumps(response))

    # ------------------------------------------------------------------
    # Idle reset (clean state when no managed app owns the robot)
    # ------------------------------------------------------------------

    # Grace period before an idle reset actually moves the robot. A transient
    # drop (Wi-Fi blip, fast session hand-off, quick app switch) frees the lock
    # and immediately reschedules a new owner; waiting this long before starting
    # goto_sleep lets that reconnect cancel the reset *before* any motion, so we
    # don't start a trajectory only to yank it mid-flight and leave the robot in
    # an intermediate pose.
    IDLE_RESET_DEBOUNCE_S: float = 1.5

    # Longer grace used when the slot was released on purpose for a successor
    # (covers a Space cold-start). Also bounds how long a client that died on
    # the endSession path keeps the robot needlessly awake.
    IDLE_RESET_HANDOFF_GRACE_S: float = 15.0

    def request_idle_reset(self, *, expect_handoff: bool = False) -> None:
        """Return the robot to a clean idle state when no managed app owns it.

        Called by the daemon when the shared ``RobotAppLock`` transitions to
        ``free`` (a remote WebRTC session ended or dropped, or a local app
        exited). The daemon never resets motor state on session teardown, and
        clients are not guaranteed to run a clean leave sequence: a crash, a
        buggy app, or a lost Wi-Fi link all skip it. Without this, whatever
        motor mode the last app left behind - notably ``gravity_compensation`` -
        survives into the next session, where the client's ``ensureAwake()``
        sees ``isAwake() == true`` and skips the wake animation, leaving the
        robot parked in a weird state.

        Threadsafe and non-blocking: hops onto the backend's own event loop (the
        one that also handles data-channel commands) and returns immediately.
        No-op when that loop isn't up yet (e.g. ``--no-media`` daemons, which
        have no remote sessions anyway).

        Args:
            expect_handoff: True when the slot was released on purpose for a
                successor session, which earns the longer
                ``IDLE_RESET_HANDOFF_GRACE_S`` instead of the default debounce.

        """
        loop = getattr(self, "_log_loop", None)
        if loop is None:
            return
        grace_s = (
            self.IDLE_RESET_HANDOFF_GRACE_S
            if expect_handoff
            else self.IDLE_RESET_DEBOUNCE_S
        )
        loop.call_soon_threadsafe(self._maybe_start_idle_reset, grace_s)

    def cancel_idle_reset(self) -> None:
        """Cancel a pending/in-flight idle reset from any thread.

        Called when a new owner grabs the robot from a path that doesn't go
        through the data channel - notably a local Python app acquiring the app
        slot (``AppManager.start_app``) - so the daemon's teardown goto_sleep
        doesn't fight the app's own wake/motion. Threadsafe: hops onto the
        backend loop that owns the reset task. No-op when that loop isn't up yet.
        """
        loop = getattr(self, "_log_loop", None)
        if loop is None:
            return
        loop.call_soon_threadsafe(self._cancel_idle_reset)

    def _cancel_idle_reset(self) -> None:
        """Cancel an in-flight idle reset. Must run on the backend loop."""
        task = self._idle_reset_task
        if task is not None and not task.done():
            task.cancel()
        self._idle_reset_task = None

    def _already_idle(self) -> bool:
        """Nothing left to do: the robot is limp AND physically at the sleep pose.

        Both halves matter. A well-behaved client (e.g. the mobile app, which
        sleeps then disables on leave) satisfies them and must not get a
        redundant trajectory + sound. Limp *anywhere else* is the crashed-app
        signature - torque cut mid-pose, head drooped - and does need the reset,
        which is why the motor mode alone is not a sufficient test.
        """
        return (
            self.get_motor_control_mode() == MotorControlMode.Disabled
            and self.is_at_sleep_pose()
        )

    def _maybe_start_idle_reset(self, grace_s: float) -> None:
        """Kick off a sleep reset if the robot isn't already idle. Runs on the loop."""
        try:
            if not self.ready.is_set() or self.is_shutting_down:
                return
            if self._already_idle():
                return
            self._cancel_idle_reset()
            self._idle_reset_task = asyncio.create_task(self._async_idle_reset(grace_s))
        except Exception:
            self.logger.warning("Idle reset scheduling failed", exc_info=True)

    async def _async_idle_reset(self, grace_s: float) -> None:
        """Graceful return to the sleep pose, which ends with motors disabled.

        Starts with a debounce (``grace_s``): a fast reconnect or a local app
        grabbing the robot cancels the task during that window, so no motion
        happens at all on transient drops. Only if the slot stays free past the
        grace period do we actually reset.

        Goes through ``reset_to_sleep()`` rather than ``goto_sleep()`` directly:
        the client that just vanished may have left the motors limp or in
        gravity compensation, and a bare ``goto_sleep()`` degrades silently in
        both cases. ``reset_to_sleep()`` finishes with
        ``set_motor_control_mode(Disabled)``, which also clears
        ``gravity_compensation_mode`` - so the next session's ``ensureAwake()``
        correctly triggers a fresh wake.
        """
        try:
            await asyncio.sleep(grace_s)
            # Conditions may have changed during the grace period (shutdown
            # started, or the robot was already put to sleep by whoever briefly
            # held the slot). Re-check before committing to the trajectory.
            if self.is_shutting_down:
                return
            if self._already_idle():
                return
            await self.reset_to_sleep()
        except asyncio.CancelledError:
            # A new session (or local app) grabbed the robot before/mid-reset:
            # let it take over without noise.
            raise
        except Exception:
            self.logger.warning("Idle reset to sleep failed", exc_info=True)
        finally:
            # Only clear the handle if it still points at *this* task: a
            # concurrent _cancel_idle_reset()+reschedule may have already
            # installed a newer task, and blindly nulling would orphan it
            # (a later cancel would then miss it).
            if self._idle_reset_task is asyncio.current_task():
                self._idle_reset_task = None
