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
    upload_audio_start, upload_audio_chunk, upload_audio_finish,
    play_uploaded_move, cancel_move,
    play_uploaded_audio, cancel_audio,
    apply_audio_config, read_audio_parameter

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

class SetSpeechOffsetsCmd(BaseModel):
    """Set head-wobbler speech offsets (composed with target pose before IK)."""

    type: Literal["set_speech_offsets"] = "set_speech_offsets"
    offsets: list[float]  # [x_m, y_m, z_m, roll_rad, pitch_rad, yaw_rad]


class SetWobblingCmd(BaseModel):
    """Enable or disable daemon-side audio-reactive head wobbling."""

    type: Literal["set_wobbling"] = "set_wobbling"
    enabled: bool

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


# XVF3800 audio-board configuration over the DataChannel.
#
# Remote counterparts of `AudioBase.apply_audio_config()` and
# `ReSpeaker.read_values()` (see `media/audio_control_utils.py`).
# Both operations open a short-lived ReSpeaker USB handle on the
# daemon side; values flow as plain numbers (the protocol does not
# distinguish int vs float — see `AudioControlValue`).


class AudioParamPair(BaseModel):
    """One ``(parameter_name, values)`` pair in an audio config payload."""

    name: str
    values: list[float]


class ApplyAudioConfigCmd(BaseModel):
    """Write a batch of XVF3800 parameters and (optionally) verify them."""

    type: Literal["apply_audio_config"] = "apply_audio_config"
    config: list[AudioParamPair]
    verify: bool = True


class ReadAudioParameterCmd(BaseModel):
    """Read a single XVF3800 parameter by name."""

    type: Literal["read_audio_parameter"] = "read_audio_parameter"
    name: str


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
# round trip. The fix is to upload the whole move to the daemon up
# front and let Backend.play_move run the inner loop locally on the
# robot.
#
# Wire flow (all five commands are fire-and-forget at the transport
# level; the daemon never sends a per-command ack):
#
#   1. UploadMoveStartCmd  opens a slot
#   2. UploadMoveChunkCmd  sends a fragment of the move JSON. Chunks
#                          must arrive in order; WebRTC SCTP is
#                          ordered+reliable so the client can pipeline
#                          freely without per-chunk acks
#   3. UploadMoveFinishCmd assembles + parses; bad slots are silently
#                          dropped
#   4. PlayUploadedMoveCmd spawns Backend.play_move. Surfaces two
#                          UNSOLICITED broadcast messages tagged with
#                          the slot id (see "play_uploaded_move"
#                          server-to-client shape below): one at the
#                          moment the playback loop starts, one when
#                          it ends (finished / cancelled / error)
#   5. CancelMoveCmd       cancels the play_uploaded_move that
#                          owns the given upload_id, via a per-run
#                          cancellation token. Idempotent; no-op for
#                          stale ids that don't match the active run
#
# Slots are in-memory only; evicted on play-finish, cancel, or TTL.
#
# AUDIO: optional. A second parallel upload (UploadAudioStartCmd /
# ChunkCmd / FinishCmd) uses the SAME upload_id to attach a WAV file
# to the move slot.  When present, play_uploaded_move passes the
# audio path to Backend.play_move's sound_path, which triggers
# GStreamer playbin on the robot.  Both motion and audio then run
# on the same clock (the daemon's), eliminating cross-machine drift
# entirely.  Clients that prefer to keep audio in the browser (e.g.
# user picked "Device" output) simply skip the audio upload.
# ------------------------------------------------------------------


class UploadMoveStartCmd(BaseModel):
    """Open an upload slot for a new move. Fire-and-forget.

    ``upload_id`` is chosen by the client (use a UUID). ``total_chunks``
    lets the daemon validate completeness at finish time. ``description``
    and ``estimated_duration_s`` are diagnostics surfaced in daemon
    logs only.

    ``encoding`` controls how the daemon decodes the assembled payload
    at finish time:
      - "json"        (default): chunks are UTF-8 JSON text, concatenated
                       and parsed directly. Simplest, works without any
                       extra deps on the client.
      - "gzip+base64": chunks are base64-encoded fragments of a
                       gzipped UTF-8 JSON payload. Roughly 3x smaller
                       on the wire for typical recorded moves (floats
                       compress extremely well). Recommended for moves
                       longer than a few seconds.
    """

    type: Literal["upload_move_start"] = "upload_move_start"
    upload_id: str
    # 4096 chunks × ~16 KB = 64 MB max wire payload. A 10-minute 100 Hz
    # gzipped-base64 move is ~5 MB / ~300 chunks; the cap exists to
    # bound RAM on the CM4 if a misbehaving client over-declares.
    total_chunks: int = Field(..., ge=1, le=4096)
    description: str = ""
    estimated_duration_s: float = Field(default=0.0, ge=0.0)
    encoding: Literal["json", "gzip+base64"] = "json"


class UploadMoveChunkCmd(BaseModel):
    """One fragment of a move payload. Fire-and-forget.

    ``chunk`` is a slice of the JSON-serialized move (UTF-8). Chunks
    must arrive in order: SCTP guarantees that on a healthy data
    channel, and a single misordered chunk discards the slot
    server-side. Pipelining at line rate is fine.
    """

    type: Literal["upload_move_chunk"] = "upload_move_chunk"
    upload_id: str
    chunk_index: int = Field(..., ge=0)
    # The JS SDK slices at 12 KB; the 16 KB ceiling leaves headroom for
    # base64/JSON envelope overhead while still blocking pathological
    # multi-MB single-chunk sends from a misbehaving client.
    chunk: str = Field(..., max_length=16 * 1024)


class UploadMoveFinishCmd(BaseModel):
    """Close an upload slot. Fire-and-forget.

    Daemon assembles the fragments, parses them as a recorded move,
    and stores the result keyed by ``upload_id``. If anything fails
    (chunk count mismatch, JSON parse error, malformed move shape)
    the slot is silently dropped: the next :class:`PlayUploadedMoveCmd`
    will broadcast a ``"no such uploaded move"`` error so the client
    learns about the failure at play time.
    """

    type: Literal["upload_move_finish"] = "upload_move_finish"
    upload_id: str


class UploadAudioStartCmd(BaseModel):
    """Open an audio slot keyed by the same upload_id as a move.

    Fire-and-forget.  ``upload_id`` must match the move's id; at
    play time the daemon pairs the two.  Audio uploaded without a
    matching move is held until either a matching move arrives or
    the TTL expires.

    ``encoding`` is currently always ``"wav-base64"``: raw PCM WAV
    bytes (any container the GStreamer playbin can decode) sliced
    into chunks and base64-encoded.  Defined as an enum so a future
    encoding (raw binary frames, opus, ...) can be added without
    breaking older clients.
    """

    type: Literal["upload_audio_start"] = "upload_audio_start"
    upload_id: str
    # 16384 chunks × ~16 KB = 256 MB max wire payload. A 5-minute
    # 16 kHz mono PCM song base64-encodes to ~13 MB / ~850 chunks,
    # so this cap gives roughly 1.5 hours of headroom without
    # exposing the CM4 to multi-GB allocations on a misbehaving
    # client.
    total_chunks: int = Field(..., ge=1, le=16384)
    encoding: Literal["wav-base64"] = "wav-base64"
    description: str = ""


class UploadAudioChunkCmd(BaseModel):
    """One fragment of an audio payload. Fire-and-forget.

    ``chunk`` is a slice of the base64-encoded audio bytes.  Chunks
    must arrive in order on a healthy SCTP data channel; a single
    misordered chunk discards the slot server-side.
    """

    type: Literal["upload_audio_chunk"] = "upload_audio_chunk"
    upload_id: str
    chunk_index: int = Field(..., ge=0)
    # Matches UploadMoveChunkCmd.chunk: see comment there.
    chunk: str = Field(..., max_length=16 * 1024)


class UploadAudioFinishCmd(BaseModel):
    """Close an audio slot. Fire-and-forget.

    Daemon decodes the base64-assembled payload and writes the
    resulting WAV bytes to a temp file under
    ``<platform-tempdir>/reachy-mini-uploads/audio/{upload_id}.wav``
    (``/tmp`` on Linux/macOS, ``%TEMP%`` on Windows).  At
    play_uploaded_move time the path is attached as ``sound_path`` on
    the move object so Backend.play_move starts GStreamer playbin in
    lockstep with the motion loop.  The temp file is deleted after
    playback ends.

    If the audio failed to assemble (chunk mismatch, bad base64), the
    slot is silently dropped: the move plays without audio rather
    than blocking.
    """

    type: Literal["upload_audio_finish"] = "upload_audio_finish"
    upload_id: str


class PlayUploadedMoveCmd(BaseModel):
    """Play a previously-uploaded move on the daemon.

    Fire-and-forget at the transport level; progress comes back as
    broadcast events.

    The daemon spawns Backend.play_move as a background task and emits
    two unsolicited messages tagged ``type="play_uploaded_move"`` and
    ``upload_id=<this id>``:

    - ``{"started": true, "duration_s": D}`` when the inner loop is
      about to tick for the first time.  When the same upload_id
      also has an uploaded audio attached, the daemon-side GStreamer
      playbin has already started by this point too: clients should
      NOT trigger a second audio source from their own side.
    - ``{"finished": true}`` / ``{"cancelled": true}`` / ``{"error":
      "..."}`` exactly once when the task ends.

    Both messages go out on every transport the daemon serves (WS
    broadcast + WebRTC data channel broadcast).  Clients filter by
    ``upload_id``.

    ``initial_goto_duration`` works the same as in Backend.play_move:
    if non-zero, the robot smoothly interpolates to the move's first
    frame before the playback loop starts.  Callers that want to
    handle that approach themselves leave this at 0.

    ``audio_lead_ms`` shifts the daemon-side audio start relative to
    the motion start.  Positive means audio plays N ms BEFORE motion
    (compensates for the constant GStreamer playbin latency on the
    robot; typical values: 0-100 ms).  Negative is supported but
    rarely useful.  Only applied when an audio is attached.
    """

    type: Literal["play_uploaded_move"] = "play_uploaded_move"
    upload_id: str
    play_frequency: float = Field(default=100.0, gt=0.0, le=200.0)
    initial_goto_duration: float = Field(default=0.0, ge=0.0)
    audio_lead_ms: float = Field(default=0.0, ge=-2000.0, le=2000.0)


class CancelMoveCmd(BaseModel):
    """Cancel the play_uploaded_move identified by ``upload_id``.

    Fire-and-forget. The backend cancels only if the currently-running
    uploaded move's id matches ``upload_id`` — a stale cancel arriving
    after the targeted move ended (or against a never-started id) is a
    no-op. Scoped to uploaded moves; direct ``Backend.play_move`` calls
    (e.g. via ``goto_target``) are never cancelled by this command.

    Goes through the per-run cancellation token created by
    ``_async_play_uploaded_move`` so two back-to-back plays can't
    cross-cancel each other.
    """

    type: Literal["cancel_move"] = "cancel_move"
    upload_id: str


class PlayUploadedAudioCmd(BaseModel):
    """Play a previously-uploaded audio standalone (no motion).

    Used by clients during recording to keep the audio pipeline
    identical between record time and play time.  Same upload_id as
    the audio attached.  Daemon plays the WAV via the same
    GStreamer playbin path used by play_uploaded_move, and emits a
    broadcast ``{"type":"play_uploaded_audio","upload_id":...,
    "started":true}`` event the moment ``set_state(PLAYING)`` is
    called.  Clients use this event as the t=0 reference for
    motion capture, so the eventual play_uploaded_move that uses
    the same audio reproduces the recording-time alignment
    (pipeline latency cancels).

    Stop via CancelAudioCmd.  No finished event is emitted; the
    daemon doesn't track playback duration -- callers know it from
    the WAV header and stop on their own.
    """

    type: Literal["play_uploaded_audio"] = "play_uploaded_audio"
    upload_id: str


class CancelAudioCmd(BaseModel):
    """Stop the play_uploaded_audio identified by ``upload_id``. Fire-and-forget.

    The backend cancels only if the currently-playing standalone audio's
    id matches ``upload_id``; a stale cancel against a different (or
    already-finished) id is a no-op. Won't touch audio attached to an
    in-flight play_uploaded_move — that audio is cancelled together with
    the move via :class:`CancelMoveCmd`.
    """

    type: Literal["cancel_audio"] = "cancel_audio"
    upload_id: str


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
    | SetSpeechOffsetsCmd
    | SetWobblingCmd
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
    | UploadAudioStartCmd
    | UploadAudioChunkCmd
    | UploadAudioFinishCmd
    | PlayUploadedMoveCmd
    | CancelMoveCmd
    | PlayUploadedAudioCmd
    | CancelAudioCmd
    | ApplyAudioConfigCmd
    | ReadAudioParameterCmd,
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
