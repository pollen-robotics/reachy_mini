"""Protocol definitions for Reachy Mini client/server communication.

All messages use a {"type": "...", ...payload} envelope.

Client->Server command types:
    set_target, set_head_joints, set_body_yaw, set_antennas, set_full_target,
    goto_target, wake_up, goto_sleep, play_sound,
    set_motor_mode, set_torque, get_motor_mode,
    set_gravity_compensation, set_automatic_body_yaw,
    get_state, get_version, start_recording, stop_recording, append_record

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
# Generic HTTP proxy command.
#
# Lets a remote WebRTC peer call any /api/... route on the daemon
# that does not yet have a typed DataChannel command (e.g.
# /api/hf-auth/*, /api/apps/*, /api/daemon/status, ...). The daemon
# forwards (method, path, body, headers) to its own
# http://127.0.0.1:8000 server and returns the response carrying
# the same `request_id` so the client can correlate.
#
# This is a transitional, generic transport while we migrate hot
# paths to typed *Cmd schemas. Existing typed commands (head pose,
# audio, motors, ...) should be preferred when they exist - the
# proxy is not type-checked and bypasses pydantic validation of
# the inner payloads.
#
# Trust boundary: enforced upstream by whoever can establish a
# WebRTC session with this daemon (HF central producer auth in
# remote, GStreamer signaling on LAN). The proxy does not add
# attack surface beyond what the existing typed commands already
# expose to the same caller. Endpoints that mutate the daemon's
# auth state (e.g. /api/hf-auth/save-token) are still callable
# - by design, since this is the whole point of the unified
# transport.
# ------------------------------------------------------------------


class HttpProxyCmd(BaseModel):
    """Proxy an HTTP request to the daemon's own /api/... route.

    The daemon forwards `method path body headers` to
    http://127.0.0.1:8000 and returns a `http_proxy_response`
    message echoing `request_id`.
    """

    type: Literal["http_proxy"] = "http_proxy"
    request_id: str
    method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"]
    path: str = Field(
        ..., description="Path starting with '/', e.g. '/api/daemon/status'."
    )
    body: Any | None = None
    headers: dict[str, str] | None = None
    timeout_s: float = Field(30.0, ge=0.1, le=300.0)


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
    | StartRecordingCmd
    | StopRecordingCmd
    | AppendRecordCmd
    | SetVolumeCmd
    | GetVolumeCmd
    | SetMicrophoneVolumeCmd
    | GetMicrophoneVolumeCmd
    | HttpProxyCmd,
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
    | TaskProgress,
    Field(discriminator="type"),
]
server_msg_adapter: TypeAdapter[AnyServerMsg] = TypeAdapter(AnyServerMsg)
