"""Motor setup script for the Reachy Mini robot.

This script allows to configure the motors of the Reachy Mini robot by setting their ID, baudrate, offset, angle limits, return delay time, and removing the input voltage error.

The motor needs to be configured one by one, so you will need to connect only one motor at a time to the serial port. You can specify which motor to configure by passing its name as an argument.

If not specified, it assumes the motor is in the factory settings (ID 1 and baudrate 57600). If it's not the case, you will need to use a tool like Dynamixel Wizard to first reset it or manually specify the ID and baudrate.

Please note that all values given in the configuration file are in the motor's raw units.
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

from rustypot import Xl330PyController


@dataclass
class MotorConfig:
    """Motor configuration."""

    id: int
    offset: int
    angle_limit_min: int
    angle_limit_max: int
    return_delay_time: int
    shutdown_error: int


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
    import yaml

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
            )

    serial = SerialConfig(baudrate=conf["serial"]["baudrate"])

    return ReachyMiniConfig(
        version=version,
        serial=serial,
        motors=motor_ids,
    )


FACTORY_DEFAULT_ID = 1
FACTORY_DEFAULT_BAUDRATE = 57600
SERIAL_TIMEOUT = 0.002  # seconds
MOTOR_SETUP_DELAY = 0.1  # seconds

XL_BAUDRATE_CONV_TABLE = {
    9600: 0,
    57600: 1,
    115200: 2,
    1000000: 3,
    2000000: 4,
    3000000: 5,
    4000000: 6,
}


def setup_motor(
    motor_config: MotorConfig,
    serial_port: str,
    from_baudrate: int,
    target_baudrate: int,
    from_id: int,
):
    """Set up the motor with the given configuration."""
    if not lookup_for_motor(
        serial_port,
        from_id,
        from_baudrate,
    ):
        raise RuntimeError(
            f"No motor found on port {serial_port}. "
            f"Make sure the motor is in factory settings (ID {from_id} and baudrate {from_baudrate}) and connected to the specified port."
        )

    # Make sure the torque is disabled to be able to write EEPROM
    disable_torque(serial_port, from_id, from_baudrate)

    if from_baudrate != target_baudrate:
        change_baudrate(
            serial_port,
            id=from_id,
            base_baudrate=from_baudrate,
            target_baudrate=target_baudrate,
        )
        time.sleep(MOTOR_SETUP_DELAY)

    if from_id != motor_config.id:
        change_id(
            serial_port,
            current_id=from_id,
            new_id=motor_config.id,
            baudrate=target_baudrate,
        )
        time.sleep(MOTOR_SETUP_DELAY)

    change_offset(
        serial_port,
        id=motor_config.id,
        offset=motor_config.offset,
        baudrate=target_baudrate,
    )

    time.sleep(MOTOR_SETUP_DELAY)

    change_angle_limits(
        serial_port,
        id=motor_config.id,
        angle_limit_min=motor_config.angle_limit_min,
        angle_limit_max=motor_config.angle_limit_max,
        baudrate=target_baudrate,
    )

    time.sleep(MOTOR_SETUP_DELAY)

    change_shutdown_error(
        serial_port,
        id=motor_config.id,
        baudrate=target_baudrate,
        shutdown_error=motor_config.shutdown_error,
    )

    time.sleep(MOTOR_SETUP_DELAY)

    change_return_delay_time(
        serial_port,
        id=motor_config.id,
        return_delay_time=motor_config.return_delay_time,
        baudrate=target_baudrate,
    )

    time.sleep(MOTOR_SETUP_DELAY)


def lookup_for_motor(
    serial_port: str, id: int, baudrate: int, silent: bool = False
) -> bool:
    """Check if a motor with the given ID is reachable on the specified serial port."""
    if not silent:
        print(
            f"Looking for motor with ID {id} on port {serial_port}...",
            end="",
            flush=True,
        )
    c = Xl330PyController(serial_port, baudrate=baudrate, timeout=SERIAL_TIMEOUT)
    ret = c.ping(id)
    if not silent:
        print(f"{'✅' if ret else '❌'}")
    return ret


def disable_torque(serial_port: str, id: int, baudrate: int):
    """Disable the torque of the motor with the given ID on the specified serial port."""
    print(f"Disabling torque for motor with ID {id}...", end="", flush=True)
    c = Xl330PyController(serial_port, baudrate=baudrate, timeout=SERIAL_TIMEOUT)
    c.write_torque_enable(id, False)
    print("✅")


def change_baudrate(
    serial_port: str, id: int, base_baudrate: int, target_baudrate: int
):
    """Change the baudrate of the motor with the given ID on the specified serial port."""
    print(f"Changing baudrate to {target_baudrate}...", end="", flush=True)
    c = Xl330PyController(serial_port, baudrate=base_baudrate, timeout=SERIAL_TIMEOUT)
    c.write_baud_rate(id, XL_BAUDRATE_CONV_TABLE[target_baudrate])
    print("✅")


def change_id(serial_port: str, current_id: int, new_id: int, baudrate: int):
    """Change the ID of the motor with the given current ID on the specified serial port."""
    print(f"Changing ID from {current_id} to {new_id}...", end="", flush=True)
    c = Xl330PyController(serial_port, baudrate=baudrate, timeout=SERIAL_TIMEOUT)
    c.write_id(current_id, new_id)
    print("✅")


def change_offset(serial_port: str, id: int, offset: int, baudrate: int):
    """Change the offset of the motor with the given ID on the specified serial port."""
    print(f"Changing offset for motor with ID {id} to {offset}...", end="", flush=True)
    c = Xl330PyController(serial_port, baudrate=baudrate, timeout=SERIAL_TIMEOUT)
    c.write_homing_offset(id, offset)
    print("✅")


def change_angle_limits(
    serial_port: str,
    id: int,
    angle_limit_min: int,
    angle_limit_max: int,
    baudrate: int,
):
    """Change the angle limits of the motor with the given ID on the specified serial port."""
    print(
        f"Changing angle limits for motor with ID {id} to [{angle_limit_min}, {angle_limit_max}]...",
        end="",
        flush=True,
    )
    c = Xl330PyController(serial_port, baudrate=baudrate, timeout=SERIAL_TIMEOUT)
    c.write_raw_min_position_limit(id, angle_limit_min)
    c.write_raw_max_position_limit(id, angle_limit_max)
    print("✅")


def change_shutdown_error(
    serial_port: str, id: int, baudrate: int, shutdown_error: int
):
    """Change the shutdown error of the motor with the given ID on the specified serial port."""
    print(
        f"Changing shutdown error for motor with ID {id} to {shutdown_error}...",
        end="",
        flush=True,
    )
    c = Xl330PyController(serial_port, baudrate=baudrate, timeout=SERIAL_TIMEOUT)
    c.write_shutdown(id, shutdown_error)
    print("✅")


def change_return_delay_time(
    serial_port: str, id: int, return_delay_time: int, baudrate: int
):
    """Change the return delay time of the motor with the given ID on the specified serial port."""
    print(
        f"Changing return delay time for motor with ID {id} to {return_delay_time}...",
        end="",
        flush=True,
    )
    c = Xl330PyController(serial_port, baudrate=baudrate, timeout=SERIAL_TIMEOUT)
    c.write_return_delay_time(id, return_delay_time)
    print("✅")


def light_led_up(serial_port: str, id: int, baudrate: int):
    """Light the LED of the motor with the given ID on the specified serial port."""
    c = Xl330PyController(serial_port, baudrate=baudrate, timeout=SERIAL_TIMEOUT)
    c.write_led(id, 1)


def check_configuration(motor_config: MotorConfig, serial_port: str, baudrate: int):
    """Check the configuration of the motor with the given ID on the specified serial port."""
    c = Xl330PyController(serial_port, baudrate=baudrate, timeout=SERIAL_TIMEOUT)

    print("Checking configuration...")

    # Check if there is a motor with the desired ID
    if not c.ping(motor_config.id):
        raise RuntimeError(f"No motor with ID {motor_config.id} found, cannot proceed")
    print(f"Found motor with ID {motor_config.id} ✅.")

    # Read return delay time
    return_delay = c.read_return_delay_time(motor_config.id)[0]
    if return_delay != motor_config.return_delay_time:
        raise RuntimeError(
            f"Return delay time is {return_delay}, expected {motor_config.return_delay_time}"
        )
    print(f"Return delay time is correct: {return_delay} ✅.")

    # Read angle limits
    angle_limit_min = c.read_raw_min_position_limit(motor_config.id)[0]
    angle_limit_max = c.read_raw_max_position_limit(motor_config.id)[0]
    if angle_limit_min != motor_config.angle_limit_min:
        raise RuntimeError(
            f"Angle limit min is {angle_limit_min}, expected {motor_config.angle_limit_min}"
        )
    if angle_limit_max != motor_config.angle_limit_max:
        raise RuntimeError(
            f"Angle limit max is {angle_limit_max}, expected {motor_config.angle_limit_max}"
        )
    print(
        f"Angle limits are correct: [{motor_config.angle_limit_min}, {motor_config.angle_limit_max}] ✅."
    )

    # Read homing offset
    offset = c.read_homing_offset(motor_config.id)[0]
    if offset != motor_config.offset:
        raise RuntimeError(f"Homing offset is {offset}, expected {motor_config.offset}")
    print(f"Homing offset is correct: {offset} ✅.")

    # Read shutdown
    shutdown = c.read_shutdown(motor_config.id)[0]
    if shutdown != motor_config.shutdown_error:
        raise RuntimeError(
            f"Shutdown is {shutdown}, expected {motor_config.shutdown_error}"
        )
    print(f"Shutdown error is correct: {shutdown} ✅.")

    print("Configuration is correct ✅!")


def run(args):
    """Entry point for the Reachy Mini motor configuration tool."""
    config = parse_yaml_config(args.config_file)

    if args.motor_name == "all":
        motors = list(config.motors.keys())
    else:
        motors = [args.motor_name]

    for motor_name in motors:
        motor_config = config.motors[motor_name]

        if args.update_config:
            args.from_id = motor_config.id
            args.from_baudrate = config.serial.baudrate

        if not args.check_only:
            setup_motor(
                motor_config,
                args.serialport,
                from_id=args.from_id,
                from_baudrate=args.from_baudrate,
                target_baudrate=config.serial.baudrate,
            )

        try:
            check_configuration(
                motor_config,
                args.serialport,
                baudrate=config.serial.baudrate,
            )
        except RuntimeError as e:
            print(f"❌ Configuration check failed for motor '{motor_name}': {e}")
            return False

        light_led_up(
            args.serialport,
            motor_config.id,
            baudrate=config.serial.baudrate,
        )

        return True


if __name__ == "__main__":
    """Entry point for the Reachy Mini motor configuration tool."""
    parser = argparse.ArgumentParser(description="Motor Configuration tool")
    parser.add_argument(
        "config_file",
        type=Path,
        help="Path to the hardware configuration file (default: hardware_config.yaml).",
    )
    parser.add_argument(
        "motor_name",
        type=str,
        help="Name of the motor to configure.",
        choices=[
            "body_yaw",
            "stewart_platform_1",
            "stewart_platform_2",
            "stewart_platform_3",
            "stewart_platform_4",
            "stewart_platform_5",
            "stewart_platform_6",
            "right_antenna",
            "left_antenna",
            "all",
        ],
    )
    parser.add_argument(
        "serialport",
        type=str,
        help="Serial port for communication with the motor.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check the configuration without applying changes.",
    )
    parser.add_argument(
        "--from-id",
        type=int,
        default=FACTORY_DEFAULT_ID,
        help=f"Current ID of the motor (default: {FACTORY_DEFAULT_ID}).",
    )
    parser.add_argument(
        "--from-baudrate",
        type=int,
        default=FACTORY_DEFAULT_BAUDRATE,
        help=f"Current baudrate of the motor (default: {FACTORY_DEFAULT_BAUDRATE}).",
    )
    parser.add_argument(
        "--update-config",
        action="store_true",
        help="Update a specific motor (assumes it already has the correct id and baudrate).",
    )
    args = parser.parse_args()
    run(args)
