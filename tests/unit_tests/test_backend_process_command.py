"""Tests for Backend.process_command dispatch across command families.

Covers the transport-agnostic command dispatcher in
``daemon/backend/abstract.py`` for the Set*/Get*/audio/recording/log/
lifecycle command families. The GotoTargetCmd reshape and the WebRTC
message entry point are covered separately in ``test_process_command.py``
and are not duplicated here.

All tests run headless: no hardware, no network. Media, volume, respeaker,
and hardware-id seams are patched with in-memory stubs.
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock

import numpy as np
import pytest

from reachy_mini.io.protocol import (
    AppendRecordCmd,
    ApplyAudioConfigCmd,
    AudioParamPair,
    ClearIncomingAudioCmd,
    GetHardwareIdCmd,
    GetMicrophoneVolumeCmd,
    GetMotorModeCmd,
    GetStateCmd,
    GetTrackedFaceCmd,
    GetVersionCmd,
    GetVolumeCmd,
    GotoSleepCmd,
    PlaySoundCmd,
    ReadAudioParameterCmd,
    RestartDaemonCmd,
    SetAntennasCmd,
    SetAutomaticBodyYawCmd,
    SetBodyYawCmd,
    SetFullTargetCmd,
    SetGravityCompensationCmd,
    SetHeadJointsCmd,
    SetHeadTrackingCmd,
    SetMicrophoneVolumeCmd,
    SetMotorModeCmd,
    SetSpeechOffsetsCmd,
    SetTargetCmd,
    SetTorqueCmd,
    SetVolumeCmd,
    SetWobblingCmd,
    StartRecordingCmd,
    StartUpdateCmd,
    StopRecordingCmd,
    SubscribeLogsCmd,
    UnsubscribeLogsCmd,
    WakeUpCmd,
)


def _dispatch(backend: Any, cmd: Any, peer_id: str | None = None) -> list[dict]:
    """Dispatch a command, collecting responses into a list."""
    responses: list[dict] = []
    backend.process_command(cmd, send_response=responses.append, peer_id=peer_id)
    return responses


# ------------------------------------------------------------------
# Set* target commands
# ------------------------------------------------------------------


def test_set_target_updates_head_pose(sim_backend: Any) -> None:
    """SetTargetCmd reshapes the flat head and updates target_head_pose."""
    pose = np.eye(4)
    pose[:3, 3] = [0.1, 0.2, 0.3]
    responses = _dispatch(sim_backend, SetTargetCmd(head=pose.flatten().tolist()))
    assert responses == [{"status": "ok", "command": "set_target"}]
    np.testing.assert_array_equal(sim_backend.target_head_pose, pose)


def test_set_head_joints_updates_target(sim_backend: Any) -> None:
    """SetHeadJointsCmd updates target_head_joint_positions."""
    joints = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    responses = _dispatch(sim_backend, SetHeadJointsCmd(joints=joints))
    assert responses == [{"status": "ok", "command": "set_head_joints"}]
    np.testing.assert_array_equal(
        sim_backend.target_head_joint_positions, np.array(joints)
    )


def test_set_body_yaw_updates_target(sim_backend: Any) -> None:
    """SetBodyYawCmd updates target_body_yaw."""
    responses = _dispatch(sim_backend, SetBodyYawCmd(body_yaw=0.42))
    assert responses == [{"status": "ok", "command": "set_body_yaw"}]
    assert sim_backend.target_body_yaw == 0.42


def test_set_antennas_updates_target(sim_backend: Any) -> None:
    """SetAntennasCmd updates target_antenna_joint_positions."""
    responses = _dispatch(sim_backend, SetAntennasCmd(antennas=[0.3, -0.3]))
    assert responses == [{"status": "ok", "command": "set_antennas"}]
    np.testing.assert_array_equal(
        sim_backend.target_antenna_joint_positions, np.array([0.3, -0.3])
    )


def test_set_full_target_partial_fields(sim_backend: Any) -> None:
    """SetFullTargetCmd applies only the fields provided, leaving head untouched."""
    sim_backend.target_head_pose = None
    responses = _dispatch(
        sim_backend, SetFullTargetCmd(body_yaw=0.15, antennas=[0.1, 0.2])
    )
    assert responses == [{"status": "ok", "command": "set_full_target"}]
    assert sim_backend.target_head_pose is None  # head omitted -> untouched
    assert sim_backend.target_body_yaw == 0.15
    np.testing.assert_array_equal(
        sim_backend.target_antenna_joint_positions, np.array([0.1, 0.2])
    )


def test_set_full_target_all_fields(sim_backend: Any) -> None:
    """SetFullTargetCmd with head set reshapes it to 4x4."""
    pose = np.eye(4)
    pose[0, 3] = 0.5
    responses = _dispatch(
        sim_backend, SetFullTargetCmd(head=pose.flatten().tolist(), body_yaw=0.2)
    )
    assert responses == [{"status": "ok", "command": "set_full_target"}]
    np.testing.assert_array_equal(sim_backend.target_head_pose, pose)


def test_set_target_ignored_when_move_running(sim_backend: Any) -> None:
    """A running move blocks target updates but the command still acks ok."""
    sim_backend.target_head_pose = None
    sim_backend._active_move_depth = 1  # makes is_move_running True
    assert sim_backend.is_move_running
    responses = _dispatch(sim_backend, SetTargetCmd(head=np.eye(4).flatten().tolist()))
    assert responses == [{"status": "ok", "command": "set_target"}]
    assert sim_backend.target_head_pose is None  # target left unchanged


# ------------------------------------------------------------------
# Lifecycle: wake_up / goto_sleep (create asyncio tasks)
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wake_up_completes(sim_backend: Any) -> None:
    """WakeUpCmd awaits wake_up and acks a completed response."""
    sim_backend.wake_up = AsyncMock()
    responses = _dispatch(sim_backend, WakeUpCmd())
    await asyncio.sleep(0)
    sim_backend.wake_up.assert_awaited_once()
    assert responses == [{"status": "ok", "command": "wake_up", "completed": True}]


@pytest.mark.asyncio
async def test_goto_sleep_completes(sim_backend: Any) -> None:
    """GotoSleepCmd awaits goto_sleep and acks a completed response."""
    sim_backend.goto_sleep = AsyncMock()
    responses = _dispatch(sim_backend, GotoSleepCmd())
    await asyncio.sleep(0)
    sim_backend.goto_sleep.assert_awaited_once()
    assert responses == [{"status": "ok", "command": "goto_sleep", "completed": True}]


# ------------------------------------------------------------------
# Audio playback commands
# ------------------------------------------------------------------


def test_play_sound_delegates(sim_backend: Any) -> None:
    """PlaySoundCmd forwards the file name to play_sound."""
    sim_backend.play_sound = Mock()
    responses = _dispatch(sim_backend, PlaySoundCmd(file="wake_up.wav"))
    sim_backend.play_sound.assert_called_once_with("wake_up.wav")
    assert responses == [{"status": "ok", "command": "play_sound"}]


def test_clear_incoming_audio_delegates(sim_backend: Any) -> None:
    """ClearIncomingAudioCmd forwards to clear_incoming_audio."""
    sim_backend.clear_incoming_audio = Mock()
    responses = _dispatch(sim_backend, ClearIncomingAudioCmd())
    sim_backend.clear_incoming_audio.assert_called_once_with()
    assert responses == [{"status": "ok", "command": "clear_incoming_audio"}]


# ------------------------------------------------------------------
# Speech offsets / wobbling / head tracking
# ------------------------------------------------------------------


def test_set_speech_offsets_six_values(sim_backend: Any) -> None:
    """Six offsets update _speech_offsets."""
    offsets = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    responses = _dispatch(sim_backend, SetSpeechOffsetsCmd(offsets=offsets))
    assert responses == [{"status": "ok", "command": "set_speech_offsets"}]
    assert sim_backend._speech_offsets == tuple(offsets)


def test_set_speech_offsets_wrong_length_ignored(sim_backend: Any) -> None:
    """A non-6 offsets payload is ignored but still acks ok."""
    before = sim_backend._speech_offsets
    responses = _dispatch(sim_backend, SetSpeechOffsetsCmd(offsets=[0.1, 0.2, 0.3]))
    assert responses == [{"status": "ok", "command": "set_speech_offsets"}]
    assert sim_backend._speech_offsets == before


def test_set_wobbling_enabled(sim_backend: Any) -> None:
    """SetWobblingCmd enabled calls the media server's enable_wobbling."""
    responses = _dispatch(sim_backend, SetWobblingCmd(enabled=True))
    sim_backend._media_server.enable_wobbling.assert_called_once()
    assert responses == [{"status": "ok", "command": "set_wobbling"}]


def test_set_wobbling_disabled(sim_backend: Any) -> None:
    """SetWobblingCmd disabled calls disable_wobbling and zeroes offsets."""
    sim_backend._speech_offsets = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    responses = _dispatch(sim_backend, SetWobblingCmd(enabled=False))
    sim_backend._media_server.disable_wobbling.assert_called_once()
    assert sim_backend._speech_offsets == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert responses == [{"status": "ok", "command": "set_wobbling"}]


def test_set_head_tracking_enabled_available(sim_backend: Any) -> None:
    """Enabling head tracking reports enabled=True when the backend accepts it."""
    sim_backend.enable_head_tracking = Mock(return_value=True)
    responses = _dispatch(sim_backend, SetHeadTrackingCmd(enabled=True, weight=0.5))
    sim_backend.enable_head_tracking.assert_called_once_with(weight=0.5)
    assert responses == [
        {"status": "ok", "command": "set_head_tracking", "enabled": True}
    ]


def test_set_head_tracking_enabled_unavailable(sim_backend: Any) -> None:
    """Enabling head tracking reports unavailable when the backend declines."""
    sim_backend.enable_head_tracking = Mock(return_value=False)
    responses = _dispatch(sim_backend, SetHeadTrackingCmd(enabled=True))
    assert responses == [
        {"status": "unavailable", "command": "set_head_tracking", "enabled": False}
    ]


def test_set_head_tracking_disabled(sim_backend: Any) -> None:
    """Disabling head tracking calls disable_head_tracking and reports False."""
    sim_backend.disable_head_tracking = Mock()
    responses = _dispatch(sim_backend, SetHeadTrackingCmd(enabled=False))
    sim_backend.disable_head_tracking.assert_called_once_with()
    assert responses == [
        {"status": "ok", "command": "set_head_tracking", "enabled": False}
    ]


def test_get_tracked_face(sim_backend: Any) -> None:
    """GetTrackedFaceCmd returns the serialized face target."""
    responses = _dispatch(sim_backend, GetTrackedFaceCmd())
    assert len(responses) == 1
    assert responses[0]["command"] == "get_tracked_face"
    assert "face_target" in responses[0]
    assert "detected" in responses[0]["face_target"]


# ------------------------------------------------------------------
# Motor mode / torque / gravity compensation
# ------------------------------------------------------------------


def test_set_motor_mode(sim_backend: Any) -> None:
    """SetMotorModeCmd applies the requested control mode."""
    responses = _dispatch(sim_backend, SetMotorModeCmd(mode="disabled"))
    assert responses == [{"motor_mode": "disabled", "status": "ok"}]
    assert sim_backend.get_motor_control_mode().value == "disabled"


def test_get_motor_mode(sim_backend: Any) -> None:
    """GetMotorModeCmd returns the current control mode value."""
    responses = _dispatch(sim_backend, GetMotorModeCmd())
    assert responses == [{"motor_mode": "enabled"}]


def test_set_torque_with_ids(sim_backend: Any) -> None:
    """SetTorqueCmd with ids delegates to set_motor_torque_ids."""
    sim_backend.set_motor_torque_ids = Mock()
    responses = _dispatch(sim_backend, SetTorqueCmd(on=True, ids=["a", "b"]))
    sim_backend.set_motor_torque_ids.assert_called_once_with(["a", "b"], True)
    assert responses == [{"status": "ok", "command": "set_torque"}]


def test_set_torque_on_global(sim_backend: Any) -> None:
    """SetTorqueCmd on without ids enables all motors."""
    responses = _dispatch(sim_backend, SetTorqueCmd(on=True))
    assert responses == [{"status": "ok", "command": "set_torque"}]
    assert sim_backend.get_motor_control_mode().value == "enabled"


def test_set_torque_off_global(sim_backend: Any) -> None:
    """SetTorqueCmd off without ids disables all motors."""
    responses = _dispatch(sim_backend, SetTorqueCmd(on=False))
    assert responses == [{"status": "ok", "command": "set_torque"}]
    assert sim_backend.get_motor_control_mode().value == "disabled"


def test_set_gravity_compensation_enabled(sim_backend: Any) -> None:
    """SetGravityCompensationCmd enabled switches to gravity-comp mode."""
    responses = _dispatch(sim_backend, SetGravityCompensationCmd(enabled=True))
    assert responses == [{"status": "ok", "command": "set_gravity_compensation"}]
    assert sim_backend.get_motor_control_mode().value == "gravity_compensation"


def test_set_gravity_compensation_disabled(sim_backend: Any) -> None:
    """SetGravityCompensationCmd disabled switches back to enabled mode."""
    responses = _dispatch(sim_backend, SetGravityCompensationCmd(enabled=False))
    assert responses == [{"status": "ok", "command": "set_gravity_compensation"}]
    assert sim_backend.get_motor_control_mode().value == "enabled"


def test_set_gravity_compensation_value_error(sim_backend: Any) -> None:
    """A ValueError from mode switching is surfaced as an error response."""
    sim_backend.set_motor_control_mode = Mock(side_effect=ValueError("no grav comp"))
    responses = _dispatch(sim_backend, SetGravityCompensationCmd(enabled=True))
    assert responses == [
        {"error": "no grav comp", "command": "set_gravity_compensation"}
    ]


def test_set_automatic_body_yaw(sim_backend: Any) -> None:
    """SetAutomaticBodyYawCmd forwards the flag to set_automatic_body_yaw."""
    sim_backend.set_automatic_body_yaw = Mock()
    responses = _dispatch(sim_backend, SetAutomaticBodyYawCmd(enabled=True))
    sim_backend.set_automatic_body_yaw.assert_called_once_with(True)
    assert responses == [{"status": "ok", "command": "set_automatic_body_yaw"}]


# ------------------------------------------------------------------
# State / version / hardware id
# ------------------------------------------------------------------


def test_get_state(sim_backend: Any) -> None:
    """GetStateCmd returns a state dict with the expected keys."""
    responses = _dispatch(sim_backend, GetStateCmd())
    assert len(responses) == 1
    state = responses[0]["state"]
    for key in (
        "head_pose",
        "antennas",
        "body_yaw",
        "motor_mode",
        "is_recording",
        "is_move_running",
        "face_target",
    ):
        assert key in state
    assert state["is_recording"] is False
    assert state["is_move_running"] is False


def test_get_version(sim_backend: Any) -> None:
    """GetVersionCmd returns the installed reachy_mini version string."""
    responses = _dispatch(sim_backend, GetVersionCmd())
    assert len(responses) == 1
    assert isinstance(responses[0]["version"], str)


def test_get_hardware_id(sim_backend: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """GetHardwareIdCmd returns the value from get_hardware_id."""
    import reachy_mini.utils.hardware_id as hw

    monkeypatch.setattr(hw, "get_hardware_id", lambda: "HW-1234")
    responses = _dispatch(sim_backend, GetHardwareIdCmd())
    assert responses == [{"hardware_id": "HW-1234"}]


# ------------------------------------------------------------------
# Volume / microphone commands
# ------------------------------------------------------------------


def _patch_volume_control(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Install a stub VolumeControl and return it."""
    import reachy_mini.daemon.app.routers.volume_control as vcmod

    vc = MagicMock()
    monkeypatch.setattr(vcmod, "get_volume_control", lambda: vc)
    return vc


def test_set_volume_ok(sim_backend: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """SetVolumeCmd echoes the requested volume when the control accepts it."""
    vc = _patch_volume_control(monkeypatch)
    vc.set_output_volume.return_value = True
    responses = _dispatch(sim_backend, SetVolumeCmd(volume=42))
    vc.set_output_volume.assert_called_once_with(42)
    assert responses == [{"status": "ok", "command": "set_volume", "volume": 42}]


def test_set_volume_failure_returns_current(
    sim_backend: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rejected SetVolumeCmd reports error and the actual current volume."""
    vc = _patch_volume_control(monkeypatch)
    vc.set_output_volume.return_value = False
    vc.get_output_volume.return_value = 10
    responses = _dispatch(sim_backend, SetVolumeCmd(volume=42))
    assert responses == [{"status": "error", "command": "set_volume", "volume": 10}]


def test_get_volume(sim_backend: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """GetVolumeCmd returns the current output volume."""
    vc = _patch_volume_control(monkeypatch)
    vc.get_output_volume.return_value = 55
    responses = _dispatch(sim_backend, GetVolumeCmd())
    assert responses == [{"command": "get_volume", "volume": 55}]


def test_set_microphone_volume_ok(
    sim_backend: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SetMicrophoneVolumeCmd echoes the requested input volume on success."""
    vc = _patch_volume_control(monkeypatch)
    vc.set_input_volume.return_value = True
    responses = _dispatch(sim_backend, SetMicrophoneVolumeCmd(volume=30))
    vc.set_input_volume.assert_called_once_with(30)
    assert responses == [
        {"status": "ok", "command": "set_microphone_volume", "volume": 30}
    ]


def test_get_microphone_volume(
    sim_backend: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GetMicrophoneVolumeCmd returns the current input volume."""
    vc = _patch_volume_control(monkeypatch)
    vc.get_input_volume.return_value = 20
    responses = _dispatch(sim_backend, GetMicrophoneVolumeCmd())
    assert responses == [{"command": "get_microphone_volume", "volume": 20}]


def test_volume_control_unavailable(
    sim_backend: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing get_volume_control yields an error response, not a crash."""
    import reachy_mini.daemon.app.routers.volume_control as vcmod

    def _raise() -> None:
        raise RuntimeError("no audio stack")

    monkeypatch.setattr(vcmod, "get_volume_control", _raise)
    responses = _dispatch(sim_backend, GetVolumeCmd())
    assert len(responses) == 1
    assert responses[0]["command"] == "get_volume"
    assert "no audio stack" in responses[0]["error"]


# ------------------------------------------------------------------
# ReSpeaker audio config commands
# ------------------------------------------------------------------


def test_apply_audio_config_ok(
    sim_backend: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ApplyAudioConfigCmd writes config through a respeaker and reports applied."""
    import reachy_mini.media.audio_control_utils as acu

    respeaker = MagicMock()
    respeaker.apply_audio_config.return_value = True
    monkeypatch.setattr(acu, "init_respeaker_usb", lambda: respeaker)
    cmd = ApplyAudioConfigCmd(
        config=[AudioParamPair(name="AGCGAIN", values=[1.0])], verify=False
    )
    responses = _dispatch(sim_backend, cmd)
    respeaker.apply_audio_config.assert_called_once_with([("AGCGAIN", [1.0])], verify=False)
    respeaker.close.assert_called_once()
    assert responses == [
        {"status": "ok", "command": "apply_audio_config", "applied": True}
    ]


def test_read_audio_parameter_ok(
    sim_backend: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ReadAudioParameterCmd returns the values read from the respeaker."""
    import reachy_mini.media.audio_control_utils as acu

    respeaker = MagicMock()
    respeaker.read_values.return_value = [1.0, 2.0]
    monkeypatch.setattr(acu, "init_respeaker_usb", lambda: respeaker)
    responses = _dispatch(sim_backend, ReadAudioParameterCmd(name="AGCGAIN"))
    respeaker.close.assert_called_once()
    assert responses == [
        {"command": "read_audio_parameter", "name": "AGCGAIN", "values": [1.0, 2.0]}
    ]


def test_audio_config_board_not_available(
    sim_backend: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A None respeaker yields a 'not available' error response."""
    import reachy_mini.media.audio_control_utils as acu

    monkeypatch.setattr(acu, "init_respeaker_usb", lambda: None)
    responses = _dispatch(sim_backend, ReadAudioParameterCmd(name="AGCGAIN"))
    assert len(responses) == 1
    assert responses[0]["command"] == "read_audio_parameter"
    assert responses[0]["error"] == "ReSpeaker audio board not available"


def test_audio_config_init_raises(
    sim_backend: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing respeaker init yields an init-failed error response."""
    import reachy_mini.media.audio_control_utils as acu

    def _raise() -> None:
        raise RuntimeError("usb down")

    monkeypatch.setattr(acu, "init_respeaker_usb", _raise)
    responses = _dispatch(sim_backend, ReadAudioParameterCmd(name="AGCGAIN"))
    assert len(responses) == 1
    assert "usb down" in responses[0]["error"]


# ------------------------------------------------------------------
# Recording commands
# ------------------------------------------------------------------


def test_start_recording(sim_backend: Any) -> None:
    """StartRecordingCmd flips is_recording on and acks."""
    responses = _dispatch(sim_backend, StartRecordingCmd())
    assert sim_backend.is_recording is True
    assert responses == [
        {"status": "ok", "command": "start_recording", "is_recording": True}
    ]


def test_stop_recording(sim_backend: Any) -> None:
    """StopRecordingCmd flips is_recording off and publishes recorded data."""
    sim_backend.recording_publisher = MagicMock()
    sim_backend.start_recording()
    responses = _dispatch(sim_backend, StopRecordingCmd())
    assert sim_backend.is_recording is False
    sim_backend.recording_publisher.put.assert_called_once()
    assert responses == [
        {"status": "ok", "command": "stop_recording", "is_recording": False}
    ]


def test_append_record(sim_backend: Any) -> None:
    """AppendRecordCmd appends to the buffer while recording."""
    sim_backend.start_recording()
    responses = _dispatch(sim_backend, AppendRecordCmd(record={"t": 0.0}))
    assert responses == [{"status": "ok", "command": "append_record"}]
    assert sim_backend.recorded_data == [{"t": 0.0}]


# ------------------------------------------------------------------
# Log subscription
# ------------------------------------------------------------------


def test_subscribe_logs_without_peer(sim_backend: Any) -> None:
    """Subscribing without a peer id fails: log streams are peer-scoped."""
    responses = _dispatch(sim_backend, SubscribeLogsCmd(), peer_id=None)
    assert len(responses) == 1
    assert responses[0]["type"] == "log_stream_error"
    assert "peer-aware" in responses[0]["error"]


def test_unsubscribe_logs_without_peer(sim_backend: Any) -> None:
    """Unsubscribing without a peer id is a silent no-op."""
    responses = _dispatch(sim_backend, UnsubscribeLogsCmd(), peer_id=None)
    assert responses == []


# ------------------------------------------------------------------
# Restart / update lifecycle
# ------------------------------------------------------------------


def test_restart_daemon_unsupported(sim_backend: Any) -> None:
    """RestartDaemonCmd errors when no restart callback is registered."""
    sim_backend._restart_daemon_callback = None
    responses = _dispatch(sim_backend, RestartDaemonCmd())
    assert len(responses) == 1
    assert responses[0]["command"] == "restart_daemon"
    assert "not supported" in responses[0]["error"]


def test_restart_daemon_ok(sim_backend: Any) -> None:
    """RestartDaemonCmd acks ok and invokes the restart callback."""
    callback = Mock()
    sim_backend._restart_daemon_callback = callback
    responses = _dispatch(sim_backend, RestartDaemonCmd())
    callback.assert_called_once_with()
    assert responses == [{"status": "ok", "command": "restart_daemon"}]


def test_start_update_unsupported(sim_backend: Any) -> None:
    """StartUpdateCmd errors when no update callback is registered."""
    sim_backend._start_update_callback = None
    responses = _dispatch(sim_backend, StartUpdateCmd())
    assert len(responses) == 1
    assert responses[0]["command"] == "start_update"
    assert "not supported" in responses[0]["error"]


def test_start_update_accepted(sim_backend: Any) -> None:
    """StartUpdateCmd acks ok when the callback accepts (returns None)."""
    sim_backend._start_update_callback = Mock(return_value=None)
    responses = _dispatch(sim_backend, StartUpdateCmd(pre_release=True))
    sim_backend._start_update_callback.assert_called_once_with(True)
    assert responses == [{"status": "ok", "command": "start_update"}]


def test_start_update_refused(sim_backend: Any) -> None:
    """A refusal string from the callback is surfaced as an error."""
    sim_backend._start_update_callback = Mock(return_value="already running")
    responses = _dispatch(sim_backend, StartUpdateCmd())
    assert responses == [{"error": "already running", "command": "start_update"}]


def test_start_update_raises(sim_backend: Any) -> None:
    """An exception from the update callback is surfaced as an error."""
    sim_backend._start_update_callback = Mock(side_effect=RuntimeError("boom"))
    responses = _dispatch(sim_backend, StartUpdateCmd())
    assert len(responses) == 1
    assert responses[0]["command"] == "start_update"
    assert "boom" in responses[0]["error"]
