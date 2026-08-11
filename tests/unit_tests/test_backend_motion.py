"""Hardware-free tests for Backend target setters, IK, recording and async wrappers."""

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from reachy_mini.io.protocol import RecordedDataMsg


# Target setters


def test_set_target_head_pose_sets_ik_required(sim_backend) -> None:
    """set_target_head_pose stores the pose and requests IK."""
    sim_backend.ik_required = False
    pose = np.eye(4)
    sim_backend.set_target_head_pose(pose)
    assert np.array_equal(sim_backend.target_head_pose, pose)
    assert sim_backend.ik_required is True


def test_set_target_head_pose_full_tracking_weight_skips_ik(sim_backend) -> None:
    """Full-weight tracking short-circuits IK: ik_required stays False."""
    sim_backend.ik_required = False
    sim_backend._tracking_aim = np.eye(4)
    sim_backend._tracking_weight = 1.0
    sim_backend.set_target_head_pose(np.eye(4))
    assert sim_backend.ik_required is False


def test_set_target_body_yaw_sets_ik_required(sim_backend) -> None:
    """set_target_body_yaw stores the yaw and requests IK."""
    sim_backend.ik_required = False
    sim_backend.set_target_body_yaw(0.5)
    assert sim_backend.target_body_yaw == 0.5
    assert sim_backend.ik_required is True


def test_set_target_body_yaw_noop_when_unchanged(sim_backend) -> None:
    """A yaw within 1e-9 of the current value is a no-op (no IK)."""
    sim_backend.set_target_body_yaw(0.5)
    sim_backend.ik_required = False
    sim_backend.set_target_body_yaw(0.5 + 1e-12)
    assert sim_backend.ik_required is False


def test_set_target_head_joint_positions_clears_ik(sim_backend) -> None:
    """Setting joints directly bypasses IK."""
    sim_backend.ik_required = True
    positions = np.zeros(7)
    sim_backend.set_target_head_joint_positions(positions)
    assert np.array_equal(sim_backend.target_head_joint_positions, positions)
    assert sim_backend.ik_required is False


def test_set_target_antenna_joint_positions(sim_backend) -> None:
    """Antenna setter stores the positions."""
    positions = np.array([0.1, -0.2])
    sim_backend.set_target_antenna_joint_positions(positions)
    assert np.array_equal(sim_backend.target_antenna_joint_positions, positions)


def test_set_target_composite_dispatch(sim_backend) -> None:
    """set_target dispatches to head, antenna and body-yaw setters."""
    sim_backend.ik_required = False
    head = np.eye(4)
    antennas = np.array([0.3, 0.4])
    sim_backend.set_target(head=head, antennas=antennas, body_yaw=0.7)
    assert np.array_equal(sim_backend.target_head_pose, head)
    assert np.array_equal(sim_backend.target_antenna_joint_positions, antennas)
    assert sim_backend.target_body_yaw == 0.7
    assert sim_backend.ik_required is True


def test_set_speech_offsets_sets_ik_required(sim_backend) -> None:
    """set_speech_offsets stores the tuple and requests IK."""
    sim_backend.ik_required = False
    offsets = (0.01, 0.0, 0.0, 0.0, 0.0, 0.0)
    sim_backend.set_speech_offsets(offsets)
    assert sim_backend._speech_offsets == offsets
    assert sim_backend.ik_required is True


# IK


def test_update_target_head_joints_from_ik(sim_backend) -> None:
    """IK on identity pose yields joints and records the last target pose."""
    sim_backend.update_target_head_joints_from_ik(pose=np.eye(4))
    assert sim_backend.target_head_joint_positions is not None
    assert np.array_equal(sim_backend._last_target_head_pose, np.eye(4))


def test_update_target_head_joints_from_ik_with_speech_offsets(sim_backend) -> None:
    """Non-zero speech offsets are composed into the pose before IK."""
    sim_backend._speech_offsets = (0.005, 0.0, 0.0, 0.0, 0.0, 0.1)
    sim_backend.update_target_head_joints_from_ik(pose=np.eye(4))
    assert sim_backend.target_head_joint_positions is not None
    # Composed pose differs from the raw identity input.
    assert not np.array_equal(sim_backend._last_target_head_pose, np.eye(4))


def test_update_target_head_joints_from_ik_unreachable_raises(sim_backend) -> None:
    """A None IK result (collision/unreachable) raises ValueError."""
    sim_backend.head_kinematics.ik = MagicMock(return_value=None)
    with pytest.raises(ValueError):
        sim_backend.update_target_head_joints_from_ik(pose=np.eye(4))


# Recording


def test_start_recording(sim_backend) -> None:
    """start_recording arms recording with an empty buffer."""
    sim_backend.start_recording()
    assert sim_backend.is_recording is True
    assert sim_backend.recorded_data == []


def test_append_record_while_recording(sim_backend) -> None:
    """append_record stores records only while recording is active."""
    sim_backend.start_recording()
    sim_backend.append_record({"t": 1})
    assert sim_backend.recorded_data == [{"t": 1}]


def test_append_record_noop_when_stopped(sim_backend) -> None:
    """append_record is a no-op when not recording."""
    sim_backend.is_recording = False
    sim_backend.recorded_data = []
    sim_backend.append_record({"t": 1})
    assert sim_backend.recorded_data == []


def test_stop_recording_publishes(sim_backend) -> None:
    """stop_recording swaps the buffer and publishes a RecordedDataMsg."""
    publisher = MagicMock()
    sim_backend.set_recording_publisher(publisher)
    sim_backend.start_recording()
    sim_backend.append_record({"t": 1})
    sim_backend.stop_recording()
    assert sim_backend.is_recording is False
    assert sim_backend.recorded_data == []
    publisher.put.assert_called_once()
    msg = publisher.put.call_args.args[0]
    assert isinstance(msg, RecordedDataMsg)
    assert msg.data == [{"t": 1}]


def test_stop_recording_without_publisher_warns(sim_backend) -> None:
    """stop_recording with no publisher warns instead of crashing."""
    sim_backend.recording_publisher = None
    sim_backend.start_recording()
    sim_backend.append_record({"t": 1})
    sim_backend.stop_recording()  # must not raise
    assert sim_backend.is_recording is False


# Publisher setters


def test_publisher_setters_store_object(sim_backend) -> None:
    """Each publisher setter stores the given object on the backend."""
    joints, pose, imu, rec = (MagicMock() for _ in range(4))
    sim_backend.set_joint_positions_publisher(joints)
    sim_backend.set_pose_publisher(pose)
    sim_backend.set_imu_publisher(imu)
    sim_backend.set_recording_publisher(rec)
    assert sim_backend.joint_positions_publisher is joints
    assert sim_backend.pose_publisher is pose
    assert sim_backend.imu_publisher is imu
    assert sim_backend.recording_publisher is rec


# Present-pose getters


def test_get_present_body_yaw(sim_backend) -> None:
    """Present body yaw is the first head joint position."""
    expected = sim_backend.get_present_head_joint_positions()[0]
    assert sim_backend.get_present_body_yaw() == expected


def test_get_present_head_pose(sim_backend) -> None:
    """Present head pose returns the seeded current pose."""
    assert np.array_equal(sim_backend.get_present_head_pose(), np.eye(4))


def test_get_current_head_pose(sim_backend) -> None:
    """get_current_head_pose delegates to get_present_head_pose."""
    assert np.array_equal(sim_backend.get_current_head_pose(), np.eye(4))


# Async wrappers.
# The wake_up / goto_sleep success paths run through these wrappers via the
# process_command WakeUp/GotoSleep dispatch tests, so here we only add the
# error branches (and the _async_goto pair, which nothing else exercises).


@pytest.mark.asyncio
async def test_async_goto_ok(sim_backend) -> None:
    """_async_goto reports completion when goto_target succeeds."""
    sim_backend.goto_target = AsyncMock()
    send_response = MagicMock()
    await sim_backend._async_goto(send_response, np.eye(4), None, 1.0, None)
    resp = send_response.call_args.args[0]
    assert resp == {"status": "ok", "command": "goto_target", "completed": True}


@pytest.mark.asyncio
async def test_async_goto_error(sim_backend) -> None:
    """_async_goto reports the error when goto_target raises."""
    sim_backend.goto_target = AsyncMock(side_effect=RuntimeError("boom"))
    send_response = MagicMock()
    await sim_backend._async_goto(send_response, np.eye(4), None, 1.0, None)
    resp = send_response.call_args.args[0]
    assert resp["command"] == "goto_target"
    assert resp["error"] == "boom"


@pytest.mark.asyncio
async def test_async_wake_up_error(sim_backend) -> None:
    """_async_wake_up reports the error when wake_up raises."""
    sim_backend.wake_up = AsyncMock(side_effect=RuntimeError("boom"))
    send_response = MagicMock()
    await sim_backend._async_wake_up(send_response)
    resp = send_response.call_args.args[0]
    assert resp["command"] == "wake_up"
    assert resp["error"] == "boom"


@pytest.mark.asyncio
async def test_async_goto_sleep_error(sim_backend) -> None:
    """_async_goto_sleep reports the error when goto_sleep raises."""
    sim_backend.goto_sleep = AsyncMock(side_effect=RuntimeError("boom"))
    send_response = MagicMock()
    await sim_backend._async_goto_sleep(send_response)
    resp = send_response.call_args.args[0]
    assert resp["command"] == "goto_sleep"
    assert resp["error"] == "boom"
