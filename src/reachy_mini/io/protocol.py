"""Protocol definitions for Reachy Mini client/server communication.

All messages use a {"type": "...", ...payload} envelope.

Client->Server command types:
    set_target, set_head_joints, set_body_yaw, set_antennas, set_full_target,
    goto_target, wake_up, goto_sleep, play_sound,
    set_motor_mode, set_torque, get_motor_mode,
    set_gravity_compensation, set_automatic_body_yaw,
    get_state, get_version, start_recording, stop_recording, append_record,
    subscribe_logs, unsubscribe_logs, restart_daemon,
    upload_move_start, upload_move_chunk, upload_move_finish,
    play_uploaded_move, cancel_move

Server->Client message types:
    joint_positions, head_pose, imu_data, recorded_data,
    daemon_status, task_progress
"""

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field, TypeAdapter

from reachy_mini.utils.interpolation import InterpolationTechnique

# ------------------------------------------------------------------
# Shared enums
# ------------------------------------------------------------------


class MotorControlMode(str, Enum):
    """Enum for motor control modes."""

    Enabled = "enabled"
    Disabled = "disabled"
    GravityCompensation = "gravity_compensation"


class DaemonState(str, Enum):
    """Enum representing the state of the Reachy Mini daemon."""

    NOT_INITIALIZED = "not_initialized"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


# ------------------------------------------------------------------
# Backend status models
# ------------------------------------------------------------------


class RobotBackendStatus(BaseModel):
    """Status of the Robot Backend."""

    ready: bool
    motor_control_mode: MotorControlMode
    last_alive: float | None
    control_loop_stats: dict[str, Any]
    error: str | None = None


class MujocoBackendStatus(BaseModel):
    """Status of the Mujoco backend."""

    motor_control_mode: MotorControlMode
    error: str | None = None


class MockupSimBackendStatus(BaseModel):
    """Status of the MockupSim backend."""

    motor_control_mode: MotorControlMode
    error: str | None = None


class DaemonStatus(BaseModel):
    """Status of the Reachy Mini daemon."""

    type: Literal["daemon_status"] = "daemon_status"
    robot_name: str
    state: DaemonState
    wireless_version: bool
    desktop_app_daemon: bool
    simulation_enabled: Optional[bool]
    mockup_sim_enabled: Optional[bool]
    no_media: bool = False
    media_released: bool = False
    camera_specs_name: str = ""
    backend_status: Optional[
        RobotBackendStatus | MujocoBackendStatus | MockupSimBackendStatus
    ]
    error: Optional[str] = None
    wlan_ip: Optional[str] = None
    version: Optional[str] = None
    hardware_id: Optional[str] = None


# ------------------------------------------------------------------
# Client -> Server commands
# ------------------------------------------------------------------


class SetTargetCmd(BaseModel):
    """Set the target head pose (4x4 matrix, flattened)."""

    type: Literal["set_target"] = "set_target"
    head: list[float]


class SetHeadJointsCmd(BaseModel):
    """Set the target head joint positions (7 values)."""

    type: Literal["set_head_joints"] = "set_head_joints"
    joints: list[float]


class SetBodyYawCmd(BaseModel):
    """Set the target body yaw angle (radians)."""

    type: Literal["set_body_yaw"] = "set_body_yaw"
    body_yaw: float


class SetAntennasCmd(BaseModel):
    """Set the target antenna positions [right, left] (radians)."""

    type: Literal["set_antennas"] = "set_antennas"
    antennas: list[float]


class SetFullTargetCmd(BaseModel):
    """Set head, antennas, and body_yaw in a single message.

    All fields are optional so callers can send any subset.
    This avoids the overhead of three separate WebSocket messages
    when updating head + antennas + body_yaw together.
    """

    type: Literal["set_full_target"] = "set_full_target"
    head: list[float] | None = None
    antennas: list[float] | None = None
    body_yaw: float | None = None


class GotoTargetCmd(BaseModel):
    """Smooth interpolated goto with optional head, antennas, and body yaw."""

    type: Literal["goto_target"] = "goto_target"
    head: list[float] | None = None
    antennas: list[float] | None = None
    duration: float = 0.5
    body_yaw: float | None = None


class WakeUpCmd(BaseModel):
    """Wake up the robot."""

    type: Literal["wake_up"] = "wake_up"


class GotoSleepCmd(BaseModel):
    """Put the robot to sleep."""

    type: Literal["goto_sleep"] = "goto_sleep"


class PlaySoundCmd(BaseModel):
    """Play a sound file."""

    type: Literal["play_sound"] = "play_sound"
    file: str


class SetMotorModeCmd(BaseModel):
    """Set the motor control mode (enabled, disabled, gravity_compensation)."""

    type: Literal["set_motor_mode"] = "set_motor_mode"
    mode: str


class SetTorqueCmd(BaseModel):
    """Set torque on/off, optionally for specific motor IDs."""

    type: Literal["set_torque"] = "set_torque"
    on: bool
    ids: list[str] | None = None


class GetMotorModeCmd(BaseModel):
    """Query the current motor control mode."""

    type: Literal["get_motor_mode"] = "get_motor_mode"


class SetGravityCompensationCmd(BaseModel):
    """Enable or disable gravity compensation mode."""

    type: Literal["set_gravity_compensation"] = "set_gravity_compensation"
    enabled: bool


class SetAutomaticBodyYawCmd(BaseModel):
    """Enable or disable automatic body yaw."""

    type: Literal["set_automatic_body_yaw"] = "set_automatic_body_yaw"
    enabled: bool


class GetStateCmd(BaseModel):
    """Query the full robot state."""

    type: Literal["get_state"] = "get_state"


class GetVersionCmd(BaseModel):
    """Query the version."""

    type: Literal["get_version"] = "get_version"


class GetHardwareIdCmd(BaseModel):
    """Query the robot's unique hardware ID (Pollen audio device serial)."""

    type: Literal["get_hardware_id"] = "get_hardware_id"


class StartRecordingCmd(BaseModel):
    """Start recording joint data."""

    type: Literal["start_recording"] = "start_recording"


class StopRecordingCmd(BaseModel):
    """Stop recording and publish recorded data."""

    type: Literal["stop_recording"] = "stop_recording"


class AppendRecordCmd(BaseModel):
    """Append a single record to the recording buffer."""

    type: Literal["append_record"] = "append_record"
    record: dict[str, Any]


# Volume / microphone commands. Volume is a global robot setting (not
# per-session), so a remote client's change persists after they
# disconnect — same semantics as the local REST /api/volume endpoints.
class SetVolumeCmd(BaseModel):
    """Set the output (speaker) volume, 0-100."""

    type: Literal["set_volume"] = "set_volume"
    volume: int = Field(..., ge=0, le=100)


class GetVolumeCmd(BaseModel):
    """Query the current output (speaker) volume."""

    type: Literal["get_volume"] = "get_volume"


class SetMicrophoneVolumeCmd(BaseModel):
    """Set the input (microphone) volume, 0-100."""

    type: Literal["set_microphone_volume"] = "set_microphone_volume"
    volume: int = Field(..., ge=0, le=100)


class GetMicrophoneVolumeCmd(BaseModel):
    """Query the current input (microphone) volume."""

    type: Literal["get_microphone_volume"] = "get_microphone_volume"


# ------------------------------------------------------------------
# Daemon log streaming over the DataChannel.
#
# Push-based stream of `journalctl -u reachy-mini-daemon` lines. Same
# unit and same flags as the existing /logs/ws/daemon WebSocket
# (`routers/logs.py`); exposed over the typed transport so remote
# (Central-routed) peers can consume daemon logs without an LAN HTTP
# path. The unit is hard-coded — this is not a generic
# system-introspection primitive.
#
# Idempotent: re-subscribing while a stream is already running on
# the same peer cancels the previous subprocess and restarts. Stream
# auto-terminates on peer disconnect (cleanup wired in
# `daemon/backend/abstract.py`).
# ------------------------------------------------------------------


class SubscribeLogsCmd(BaseModel):
    """Subscribe the calling peer to the daemon's journalctl stream."""

    type: Literal["subscribe_logs"] = "subscribe_logs"


class UnsubscribeLogsCmd(BaseModel):
    """Stop the calling peer's log subscription. No-op if no stream."""

    type: Literal["unsubscribe_logs"] = "unsubscribe_logs"


# ------------------------------------------------------------------
# Daemon restart over the DataChannel.
#
# Mirrors `POST /api/daemon/restart`: rebuilds the backend (motor
# controller, kinematics, media server, ...). The DataChannel itself
# is torn down by the restart, so the daemon sends an ack BEFORE
# kicking off the actual restart and the client is expected to
# reconnect afterwards. Idempotent at the daemon level (the restart
# coroutine already no-ops if the daemon is STOPPED).
# ------------------------------------------------------------------


class RestartDaemonCmd(BaseModel):
    """Restart the daemon (rebuilds backend, motor controller, media server).

    The WebRTC transport is torn down by the restart, so the daemon
    sends a single ack response immediately (``{"status": "ok",
    "command": "restart_daemon"}``) and the consumer is expected to
    reconnect once the daemon is back up. There is no completion
    message - the data channel is gone before the restart finishes.
    """

    type: Literal["restart_daemon"] = "restart_daemon"


# ------------------------------------------------------------------
# Inline-move upload + daemon-side playback.
#
# Streaming control over the data channel (one set_target per tick)
# is jittery on wireless links because every frame has to make a
# WebRTC round trip. The fix is to upload the whole move to the
# daemon up front and let Backend.play_move run the inner loop
# locally on the robot.
#
# Flow:
#   1. UploadMoveStartCmd  → opens a slot, daemon acks with the slot id
#   2. UploadMoveChunkCmd  → sends a fragment of the move JSON
#                            (chunked because WebRTC data-channel
#                             messages are ~16 KB safe / 64 KB risky)
#   3. UploadMoveFinishCmd → daemon assembles + parses the move,
#                            evicts the slot if anything fails
#   4. PlayUploadedMoveCmd → spawns Backend.play_move on the slot's
#                            move, ack on start, finished/error on
#                            completion
#   5. CancelMoveCmd       → flips backend._move_cancelled so the
#                            playback loop exits at the next tick
#
# Slots are in-memory only; evicted on play-finish, cancel, or TTL.
# Audio is NOT included — Marionette keeps audio in the browser
# (WebRTC audio track) so the synchronization handshake is the
# data-channel "play" ack and the client's own audio start.
# ------------------------------------------------------------------


class UploadMoveStartCmd(BaseModel):
    """Open an upload slot for a new move.

    ``upload_id`` is chosen by the client (use a UUID).  ``total_chunks``
    lets the daemon allocate / validate the chunk count.  The body
    field is optional metadata for diagnostics; the actual move
    payload arrives in :class:`UploadMoveChunkCmd` messages.
    """

    type: Literal["upload_move_start"] = "upload_move_start"
    upload_id: str
    total_chunks: int = Field(..., ge=1, le=4096)
    description: str = ""
    estimated_duration_s: float = Field(default=0.0, ge=0.0)


class UploadMoveChunkCmd(BaseModel):
    """One fragment of a move payload.

    ``chunk`` is a slice of the JSON-serialized move (UTF-8). Chunks
    must arrive in order; out-of-order delivery is a protocol error
    and discards the slot.
    """

    type: Literal["upload_move_chunk"] = "upload_move_chunk"
    upload_id: str
    chunk_index: int = Field(..., ge=0)
    chunk: str


class UploadMoveFinishCmd(BaseModel):
    """Close an upload slot. Daemon assembles+parses the move JSON.

    The ack carries ``{"status": "ok", "command": "upload_move_finish",
    "upload_id": ..., "frames": N, "duration_s": D}`` on success.  If
    parsing fails the slot is evicted and the ack carries an ``error``
    field instead.
    """

    type: Literal["upload_move_finish"] = "upload_move_finish"
    upload_id: str


class PlayUploadedMoveCmd(BaseModel):
    """Play a previously-uploaded move on the daemon.

    Mirrors the parameters of :meth:`Backend.play_move`.  The daemon
    spawns the playback as a background task and sends two acks:
    ``{"status": "ok", "command": "play_uploaded_move", "started": True}``
    immediately, and a ``{"command": "play_uploaded_move", "finished":
    true | "cancelled": true | "error": "..."}`` message once the
    task completes.

    ``initial_goto_duration`` works the same as in ``Backend.play_move``:
    if non-zero, the robot smoothly interpolates to the move's first
    frame before the streamed playback starts.  Marionette typically
    handles this client-side via ``Motor.prepForPlayback`` and leaves
    this 0.
    """

    type: Literal["play_uploaded_move"] = "play_uploaded_move"
    upload_id: str
    play_frequency: float = Field(default=100.0, gt=0.0, le=200.0)
    initial_goto_duration: float = Field(default=0.0, ge=0.0)


class CancelMoveCmd(BaseModel):
    """Cancel any currently-running play_move / goto on the backend.

    Flips ``backend._move_cancelled`` so the playback loop exits at
    its next tick.  No-op if nothing is running; idempotent on
    repeated sends.
    """

    type: Literal["cancel_move"] = "cancel_move"


AnyCommand = Annotated[
    SetTargetCmd
    | SetHeadJointsCmd
    | SetBodyYawCmd
    | SetAntennasCmd
    | SetFullTargetCmd
    | GotoTargetCmd
    | WakeUpCmd
    | GotoSleepCmd
    | PlaySoundCmd
    | SetMotorModeCmd
    | SetTorqueCmd
    | GetMotorModeCmd
    | SetGravityCompensationCmd
    | SetAutomaticBodyYawCmd
    | GetStateCmd
    | GetVersionCmd
    | GetHardwareIdCmd
    | StartRecordingCmd
    | StopRecordingCmd
    | AppendRecordCmd
    | SetVolumeCmd
    | GetVolumeCmd
    | SetMicrophoneVolumeCmd
    | GetMicrophoneVolumeCmd
    | SubscribeLogsCmd
    | UnsubscribeLogsCmd
    | RestartDaemonCmd
    | UploadMoveStartCmd
    | UploadMoveChunkCmd
    | UploadMoveFinishCmd
    | PlayUploadedMoveCmd
    | CancelMoveCmd,
    Field(discriminator="type"),
]

command_adapter: TypeAdapter[AnyCommand] = TypeAdapter(AnyCommand)


# ------------------------------------------------------------------
# Server -> Client state messages (published by backend control loops)
# ------------------------------------------------------------------


class JointPositionsMsg(BaseModel):
    """Head and antenna joint positions (published at 50 Hz)."""

    type: Literal["joint_positions"] = "joint_positions"
    head_joint_positions: list[float]
    antennas_joint_positions: list[float]


class HeadPoseMsg(BaseModel):
    """Head pose as a 4x4 transformation matrix (published at 50 Hz)."""

    type: Literal["head_pose"] = "head_pose"
    head_pose: list[list[float]]


class ImuDataMsg(BaseModel):
    """IMU sensor data (published at 50 Hz on wireless version)."""

    type: Literal["imu_data"] = "imu_data"
    accelerometer: list[float]
    gyroscope: list[float]
    quaternion: list[float]
    temperature: float


class RecordedDataMsg(BaseModel):
    """Recorded joint data (published once when recording stops)."""

    type: Literal["recorded_data"] = "recorded_data"
    data: list[dict[str, Any]]


class LogLineMsg(BaseModel):
    """A single journalctl line for the active log subscriber.

    `timestamp` is the ISO-formatted prefix from
    `journalctl --output short-iso`; `line` is the rest of the
    record (everything after the timestamp). Consumers that want a
    severity tag should parse it from the line text — the daemon
    deliberately does not classify, since clients already have a
    parser (e.g. desktop app's `parseDaemonLogLevel`).
    """

    type: Literal["log_line"] = "log_line"
    timestamp: str
    line: str


class LogStreamErrorMsg(BaseModel):
    """The log subscription failed and is now terminated.

    Most common cause: `journalctl` is unavailable on the host
    (development macOS, non-systemd Linux). The subscription is
    over after this message; the consumer must re-`subscribe_logs`
    to retry.
    """

    type: Literal["log_stream_error"] = "log_stream_error"
    error: str


# ------------------------------------------------------------------
# Task protocol
# ------------------------------------------------------------------


class GotoTaskRequest(BaseModel):
    """A goto target task."""

    head: list[float] | None  # 4x4 flatten pose matrix
    antennas: list[float] | None  # [right_angle, left_angle] (in rads)
    duration: float
    method: InterpolationTechnique
    body_yaw: float | None


class PlayMoveTaskRequest(BaseModel):
    """A play move task."""

    move_name: str


AnyTaskRequest = GotoTaskRequest | PlayMoveTaskRequest


class TaskRequest(BaseModel):
    """Any task request (sent by client with type="task")."""

    type: Literal["task"] = "task"
    uuid: UUID
    req: AnyTaskRequest
    timestamp: datetime


AnyMessage = Annotated[AnyCommand | TaskRequest, Field(discriminator="type")]
message_adapter: TypeAdapter[AnyMessage] = TypeAdapter(AnyMessage)


class TaskProgress(BaseModel):
    """Task progress (broadcast to all clients)."""

    type: Literal["task_progress"] = "task_progress"
    uuid: UUID
    finished: bool = False
    error: str | None = None
    timestamp: datetime


AnyServerMsg = Annotated[
    JointPositionsMsg
    | HeadPoseMsg
    | ImuDataMsg
    | RecordedDataMsg
    | DaemonStatus
    | TaskProgress
    | LogLineMsg
    | LogStreamErrorMsg,
    Field(discriminator="type"),
]
server_msg_adapter: TypeAdapter[AnyServerMsg] = TypeAdapter(AnyServerMsg)
