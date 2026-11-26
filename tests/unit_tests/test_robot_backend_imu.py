"""Tests for the robot backend's IMU read."""

from typing import Any

from reachy_mini.daemon.backend.robot.backend import RobotBackend


class _FakeBMI088:
    def __init__(self) -> None:
        self.quat_args: list[dict[str, Any]] = []
        self.temperature_reads = 0

    def read_accelerometer(self, m_per_s2: bool = False) -> tuple[float, ...]:
        return (0.0, 0.0, 9.8)

    def read_gyroscope(self, deg_per_s: bool = False) -> tuple[float, ...]:
        return (0.1, 0.2, 0.3)

    def get_quat(self, dt: float, **readings: Any) -> list[float]:
        self.quat_args.append(readings)
        return [1.0, 0.0, 0.0, 0.0]

    def read_temperature(self) -> float:
        self.temperature_reads += 1
        return 40.0


def test_read_imu_reuses_readings_and_caches_temperature() -> None:
    """The filter gets the loop's own readings; temperature is read once a second."""
    backend = object.__new__(RobotBackend)
    backend.bmi088 = _FakeBMI088()
    backend.control_loop_frequency = 50.0
    backend._imu_temperature = None

    first = RobotBackend._read_imu(backend)
    second = RobotBackend._read_imu(backend)

    assert (
        backend.bmi088.quat_args
        == [{"acc": (0.0, 0.0, 9.8), "gyr": (0.1, 0.2, 0.3)}] * 2
    )
    assert backend.bmi088.temperature_reads == 1
    assert first is not None and second is not None
    assert second.temperature == 40.0
