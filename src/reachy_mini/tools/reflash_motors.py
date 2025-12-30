"""Script to reflash Reachy Mini's motors firmware."""

import argparse
from importlib.resources import files
from typing import Optional

import questionary
from rich.console import Console

import reachy_mini
from reachy_mini.daemon.utils import find_serial_port
from reachy_mini.tools.setup_motor import (
    check_configuration,
    light_led_down,
    light_led_up,
    setup_motor,
)
from reachy_mini.utils.hardware_config.parser import parse_yaml_config

BAUDRATE = 1000000


def main() -> None:
    """Entry point for the reflash_motors script."""
    parser = argparse.ArgumentParser(
        description="Reflash Reachy Mini motors' firmware.",
    )
    parser.add_argument(
        "--serialport",
        type=str,
        required=False,
        default=None,
        help="Serial port of the Reachy Mini (e.g. /dev/ttyUSB0 or COM3). "
        "If not specified, the script will try to automatically find it.",
    )
    args = parser.parse_args()
    reflash_motors(args.serialport)


def reflash_motors(
    serialport: Optional[str] = None, dont_light_up: bool = False
) -> None:
    """Reflash Reachy Mini's motors."""
    console = Console()

    config_file_path = str(
        files(reachy_mini).joinpath("assets/config/hardware_config.yaml")
    )
    config = parse_yaml_config(config_file_path)
    motors = list(config.motors.keys())
    if serialport is None:
        console.print(
            "Which version of Reachy Mini are you using?",
        )
        wireless_choice = questionary.select(
            ">",
            [
                questionary.Choice("Lite", value=False),
                questionary.Choice("Wireless", value=True),
            ],
        ).ask()
        ports = find_serial_port(wireless_version=wireless_choice)

        if len(ports) == 0:
            raise RuntimeError(
                "No Reachy Mini serial port found. "
                "Check USB connection and permissions. "
                "Or directly specify the serial port using --serialport."
            )
        elif len(ports) > 1:
            raise RuntimeError(
                f"Multiple Reachy Mini serial ports found {ports}."
                "Please specify the serial port using --serialport."
            )

        serialport = ports[0]
        console.print(f"Found Reachy Mini serial port: {serialport}", style="green")

    for motor_name in motors:
        motor_config = config.motors[motor_name]

        from_id = motor_config.id

        setup_motor(
            motor_config,
            serialport,
            from_id=from_id,
            from_baudrate=config.serial.baudrate,
            target_baudrate=config.serial.baudrate,
        )

        try:
            check_configuration(
                motor_config,
                serialport,
                baudrate=config.serial.baudrate,
            )
        except RuntimeError as e:
            console.print(
                f"[FAIL] Configuration check failed for motor '{motor_name}': {e}",
                style="red",
            )
            return

        light_led_up(
            serialport,
            motor_config.id,
            baudrate=config.serial.baudrate,
        )

        if dont_light_up:
            light_led_down(
                serialport,
                motor_config.id,
                baudrate=config.serial.baudrate,
            )
