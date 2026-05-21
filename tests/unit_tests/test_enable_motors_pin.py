"""Regression test for the brutal-snap bug in enable_motors.

If enable_motors() leaves the daemon with a stale target_head_pose and
ik_required=True, the next 50 Hz control-loop tick runs IK against the
stale pose and overwrites target_head_joint_positions, snapping the
head when torque comes on. The fix pins all target representations to
the present physical pose before flipping torque on.

This test exercises RobotBackend.enable_motors() directly with a mocked
motor controller, asserting all the post-conditions that must hold to
prevent the snap. Hardware-free.
"""

from unittest.mock import MagicMock

import numpy as np

from reachy_mini.daemon.backend.robot.backend import RobotBackend


def _build_backend_with_stale_state() -> tuple[RobotBackend, dict]:
    """Bypass __init__ (which talks to hardware) and set up the state
    that exists right before a buggy enable_motors call: stale
    target_head_pose, stale target_head_joint_positions, ik_required
    True, and a different present pose on the motor controller."""
    b = RobotBackend.__new__(RobotBackend)

    b.c = MagicMock()
    b.logger = MagicMock()
    b._current_head_operation_mode = 3  # position control
    b._current_antennas_operation_mode = 3
    b._torque_enabled = False

    # Stale Cartesian + joint state (what would be there after a goto-base
    # followed by disable_motors + manual hand movement).
    stale_pose = np.eye(4, dtype=np.float64)
    stale_pose[:3, 3] = [0.5, 0.5, 0.5]
    b.target_head_pose = stale_pose
    b.target_head_joint_positions = np.array([0.5] * 7, dtype=np.float64)
    b.target_antenna_joint_positions = np.array([0.5, -0.5], dtype=np.float64)
    b.target_body_yaw = 0.5
    b.ik_required = True

    # current_head_pose is FK'd every control-loop tick; assume it's
    # already been updated to reflect the post-hand-movement pose.
    present_pose = np.eye(4, dtype=np.float64)
    b.current_head_pose = present_pose

    # Motor controller returns the present physical pose (also post-
    # hand-movement).
    present = MagicMock()
    present.body_yaw = 0.0
    present.stewart = [0.1, 0.2, -0.3, 0.4, 0.5, -0.6]
    present.antennas = [-0.1, 0.2]
    b.c.get_last_position.return_value = present

    expected = {
        "head_joints": np.array([0.0] + present.stewart, dtype=np.float64),
        "antennas": np.array(present.antennas, dtype=np.float64),
        "body_yaw": 0.0,
        "head_pose": present_pose,
    }
    return b, expected


def test_enable_motors_pins_targets_to_present() -> None:
    b, expected = _build_backend_with_stale_state()

    b.enable_motors()

    np.testing.assert_array_equal(
        b.target_head_joint_positions, expected["head_joints"]
    )
    np.testing.assert_array_equal(
        b.target_antenna_joint_positions, expected["antennas"]
    )
    assert b.target_body_yaw == expected["body_yaw"]
    np.testing.assert_array_equal(b.target_head_pose, expected["head_pose"])


def test_enable_motors_clears_ik_required() -> None:
    """Without this, the next IK tick would re-derive joints from
    target_head_pose and overwrite our pin."""
    b, _ = _build_backend_with_stale_state()
    assert b.ik_required is True

    b.enable_motors()

    assert b.ik_required is False


def test_enable_motors_writes_hardware_before_enable_torque() -> None:
    """The motor goal-position registers must be written before
    enable_torque so the brief window between torque-on and the next
    control-loop tick doesn't drive toward a stale hardware goal."""
    b, expected = _build_backend_with_stale_state()
    calls: list[str] = []
    b.c.set_stewart_platform_position.side_effect = lambda _: calls.append("stewart")
    b.c.set_body_rotation.side_effect = lambda _: calls.append("body")
    b.c.set_antennas_positions.side_effect = lambda _: calls.append("antennas")
    b.c.enable_torque.side_effect = lambda: calls.append("enable_torque")

    b.enable_motors()

    assert calls.index("enable_torque") > calls.index("stewart")
    assert calls.index("enable_torque") > calls.index("body")
    assert calls.index("enable_torque") > calls.index("antennas")
    assert b._torque_enabled is True


def test_enable_motors_skips_position_writes_in_torque_control_mode() -> None:
    """Operation mode 0 is torque/current control (used for gravity
    compensation); the goal-position register is irrelevant there."""
    b, _ = _build_backend_with_stale_state()
    b._current_head_operation_mode = 0
    b._current_antennas_operation_mode = 0

    b.enable_motors()

    b.c.set_stewart_platform_position.assert_not_called()
    b.c.set_body_rotation.assert_not_called()
    b.c.set_antennas_positions.assert_not_called()
    b.c.enable_torque.assert_called_once()
