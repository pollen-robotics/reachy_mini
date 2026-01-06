"""Module to parse Reachy Mini hardware configuration from a YAML file."""

from dataclasses import dataclass

import yaml


@dataclass
class MotorConfig:
    """Motor configuration."""

    id: int
    offset: int
    angle_limit_min: int
    angle_limit_max: int
    return_delay_time: int
    shutdown_error: int
    operating_mode: int
    pid: tuple[int, int, int] | None = None


@dataclass
class SerialConfig:
    """Serial configuration."""

    baudrate: int


@dataclass
class ReachyMiniConfig:
    """Reachy Mini configuration."""

    version: str
    serial: SerialConfig
    motors: dict[str, MotorConfig]


def parse_yaml_config(filename: str) -> ReachyMiniConfig:
    """Parse the YAML configuration file and return a ReachyMiniConfig."""
    with open(filename, "r") as file:
        conf = yaml.load(file, Loader=yaml.FullLoader)

    version = conf["version"]

    motor_ids = {}
    for motor in conf["motors"]:
        for name, params in motor.items():
            motor_ids[name] = MotorConfig(
                id=params["id"],
                offset=params["offset"],
                angle_limit_min=params["lower_limit"],
                angle_limit_max=params["upper_limit"],
                return_delay_time=params["return_delay_time"],
                shutdown_error=params["shutdown_error"],
                operating_mode=params["operating_mode"],
                pid=params.get("pid"),
            )

    serial = SerialConfig(baudrate=conf["serial"]["baudrate"])

    return ReachyMiniConfig(
        version=version,
        serial=serial,
        motors=motor_ids,
    )
