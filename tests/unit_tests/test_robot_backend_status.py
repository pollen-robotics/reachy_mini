"""Regression tests for robot backend status reporting."""

from threading import Event

from reachy_mini.daemon.backend.robot.backend import RobotBackend
from reachy_mini.io.protocol import MotorControlMode, RobotBackendStatus


def test_robot_backend_status_reflects_live_readiness() -> None:
    """Cached status should reflect the live backend lifecycle events."""
    backend = object.__new__(RobotBackend)
    backend.ready = Event()
    backend.ready.set()
    backend.should_stop = Event()
    backend.last_alive = 123.5
    backend.error = None
    backend.motor_control_mode = MotorControlMode.Enabled
    backend._status = RobotBackendStatus(
        ready=False,
        motor_control_mode=MotorControlMode.Disabled,
        last_alive=None,
        control_loop_stats={},
    )

    status = RobotBackend.get_status(backend)

    assert status.ready is True
    assert status.last_alive == 123.5
    assert status.motor_control_mode == MotorControlMode.Enabled

    backend.should_stop.set()

    assert RobotBackend.get_status(backend).ready is False
