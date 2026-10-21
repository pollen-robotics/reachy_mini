"""Tests for the motors router."""

from types import SimpleNamespace

from reachy_mini.daemon.app.routers import motors
from reachy_mini.daemon.backend.robot.backend import RobotBackend
from reachy_mini.io.protocol import MotorControlMode


def test_gravity_compensation_without_placo_returns_409(router_app):
    """Non-Placo engine: a 409 naming the cause, pending override kept."""
    backend = SimpleNamespace(
        kinematics_engine="NN",
        motor_control_mode=MotorControlMode.Enabled,
        _partial_torque_override=True,
    )
    backend.set_motor_control_mode = lambda mode: RobotBackend.set_motor_control_mode(
        backend, mode
    )
    client = router_app(motors.router, backend=backend)

    r = client.post("/motors/set_mode/gravity_compensation")

    assert r.status_code == 409
    assert "Placo" in r.json()["detail"]
    assert backend._partial_torque_override is True
    assert backend.motor_control_mode == MotorControlMode.Enabled
