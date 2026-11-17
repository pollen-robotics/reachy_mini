"""The daemon caps the average motor voltage through the servos' Goal PWM. Hardware-free."""

import struct
from unittest.mock import MagicMock

import pytest

from reachy_mini.daemon.backend.robot.backend import (
    GOAL_PWM_ADDR,
    PRESENT_INPUT_VOLTAGE_ADDR,
    RobotBackend,
)

IDS = {"body_rotation": 10, "stewart_1": 11, "right_antenna": 17}


def _backend(supply_volts: float) -> RobotBackend:
    """Bypass __init__ (which talks to hardware); every servo reports ``supply_volts``."""
    b = RobotBackend.__new__(RobotBackend)
    b.c = MagicMock()
    b.logger = MagicMock()
    b.c.get_motor_name_id.return_value = IDS
    b.c.async_read_raw_bytes.return_value = list(
        struct.pack("<H", round(supply_volts * 10))
    )
    return b


def _goal_pwm_writes(b: RobotBackend) -> list[tuple[int, int, int]]:
    return [
        (call.args[0], call.args[1], struct.unpack("<H", bytes(call.args[2]))[0])
        for call in b.c.async_write_raw_bytes.call_args_list
    ]


@pytest.mark.parametrize(
    "supply_volts, goal_pwm",
    [(7.6, 699), (7.0, 759), (6.0, 885), (5.0, 885)],  # 6 V cap; full duty below it
)
def test_goal_pwm_holds_every_motor_at_six_volts(
    supply_volts: float, goal_pwm: int
) -> None:
    b = _backend(supply_volts)

    b._cap_motor_voltage()

    b.c.async_read_raw_bytes.assert_any_call(10, PRESENT_INPUT_VOLTAGE_ADDR, 2)
    assert _goal_pwm_writes(b) == [(id, GOAL_PWM_ADDR, goal_pwm) for id in IDS.values()]


def test_without_a_voltage_reading_the_cap_assumes_the_nominal_supply() -> None:
    """No answer, or a reading of 0: cap for 7.6 V rather than leave the servo uncapped."""
    b = _backend(7.0)
    b.c.async_read_raw_bytes.side_effect = [
        RuntimeError("no answer"),
        list(struct.pack("<H", 0)),
        list(struct.pack("<H", 70)),
    ]

    b._cap_motor_voltage()

    assert _goal_pwm_writes(b) == [
        (10, GOAL_PWM_ADDR, 699),
        (11, GOAL_PWM_ADDR, 699),
        (17, GOAL_PWM_ADDR, 759),
    ]
