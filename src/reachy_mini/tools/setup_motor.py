"""Motor setup script for the Reachy Mini robot.

This script allows to configure the motors of the Reachy Mini robot by setting their ID, baudrate, offset, angle limits, return delay time, and removing the input voltage error.

The motor needs to be configured one by one, so you will need to connect only one motor at a time to the serial port. You can specify which motor to configure by passing its name as an argument.

If not specified, it assumes the motor is in the factory settings (ID 1 and baudrate 57600). If it's not the case, you will need to use a tool like Dynamixel Wizard to first reset it or manually specify the ID and baudrate.

Please note that all values given in the configuration file are in the motor's raw units.
"""

import argparse
import time
from pathlib import Path

from rustypot import Xl330PyController

from reachy_mini.utils.hardware_config.parser import MotorConfig, parse_yaml_config

FACTORY_DEFAULT_ID = 1
FACTORY_DEFAULT_BAUDRATE = 57600
SERIAL_TIMEOUT = 0.01  # seconds
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

id2name = {
    10: "body_rotation",
    11: "stewart_1",
    12: "stewart_2",
    13: "stewart_3",
    14: "stewart_4",
    15: "stewart_5",
    16: "stewart_6",
    17: "right_antenna",
    18: "left_antenna",
}

# (TX + RX) * (1 start + 8 data + 1 stop)
COMMANDS_BITS_LENGTH = {
    "Ping": (10 + 14) * 10,
    "Read": (14 + 15) * 10,
    "Write": (16 + 11) * 10,
}


def setup_motor(
    motor_config: MotorConfig,
    serial_port: str,
    from_baudrate: int,
    target_baudrate: int,
    from_id: int,
) -> None:
    """Set up the motor with the given configuration."""
    if not lookup_for_motor(
        serial_port,
        from_id,
        from_baudrate,
    ):
        raise RuntimeError(
            f"No motor found on port {serial_port}. Make sure Reachy Mini is powered !"
        )
        # f"Make sure the motor {id2name.get(from_id, from_id)} is in factory settings (ID {from_id} and baudrate {from_baudrate}) and connected to the specified port."

    # Make sure the torque is disabled to be able to write EEPROM
    disable_torque(serial_port, from_id, from_baudrate)
    try:
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

        change_operating_mode(
            serial_port,
            id=motor_config.id,
            operating_mode=motor_config.operating_mode,
            baudrate=target_baudrate,
        )

        time.sleep(MOTOR_SETUP_DELAY)
    except Exception as e:
        print(f"Error while setting up motor ID {from_id}: {e}")
        raise e


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
    c = Xl330PyController(
        serial_port,
        baudrate=baudrate,
        timeout=SERIAL_TIMEOUT + float(COMMANDS_BITS_LENGTH["Ping"]) / baudrate,
    )
    ret = c.ping(id)
    if not silent:
        print(f"{'[OK]' if ret else '[FAIL]'}")
    return ret


def disable_torque(serial_port: str, id: int, baudrate: int) -> None:
    """Disable the torque of the motor with the given ID on the specified serial port."""
    print(f"Disabling torque for motor with ID {id}...", end="", flush=True)
    c = Xl330PyController(
        serial_port,
        baudrate=baudrate,
        timeout=SERIAL_TIMEOUT + float(COMMANDS_BITS_LENGTH["Write"]) / baudrate,
    )
    c.write_torque_enable(id, False)
    print("[OK]")


def change_baudrate(
    serial_port: str, id: int, base_baudrate: int, target_baudrate: int
) -> None:
    """Change the baudrate of the motor with the given ID on the specified serial port."""
    print(f"Changing baudrate to {target_baudrate}...", end="", flush=True)
    c = Xl330PyController(
        serial_port,
        baudrate=base_baudrate,
        timeout=SERIAL_TIMEOUT + float(COMMANDS_BITS_LENGTH["Write"]) / base_baudrate,
    )
    c.write_baud_rate(id, XL_BAUDRATE_CONV_TABLE[target_baudrate])
    print("[OK]")


def change_id(serial_port: str, current_id: int, new_id: int, baudrate: int) -> None:
    """Change the ID of the motor with the given current ID on the specified serial port."""
    print(f"Changing ID from {current_id} to {new_id}...", end="", flush=True)
    c = Xl330PyController(
        serial_port,
        baudrate=baudrate,
        timeout=SERIAL_TIMEOUT + float(COMMANDS_BITS_LENGTH["Write"]) / baudrate,
    )
    c.write_id(current_id, new_id)
    print("[OK]")


def change_offset(serial_port: str, id: int, offset: int, baudrate: int) -> None:
    """Change the offset of the motor with the given ID on the specified serial port."""
    print(f"Changing offset for motor with ID {id} to {offset}...", end="", flush=True)
    c = Xl330PyController(
        serial_port,
        baudrate=baudrate,
        timeout=SERIAL_TIMEOUT + float(COMMANDS_BITS_LENGTH["Write"]) / baudrate,
    )
    c.write_homing_offset(id, offset)
    print("[OK]")


def change_operating_mode(
    serial_port: str, id: int, operating_mode: int, baudrate: int
) -> None:
    """Change the operating mode of the motor with the given ID on the specified serial port."""
    print(
        f"Changing operating mode for motor with ID {id} to {operating_mode}...",
        end="",
        flush=True,
    )
    c = Xl330PyController(
        serial_port,
        baudrate=baudrate,
        timeout=SERIAL_TIMEOUT + float(COMMANDS_BITS_LENGTH["Write"]) / baudrate,
    )
    c.write_operating_mode(id, operating_mode)
    print("[OK]")


def change_angle_limits(
    serial_port: str,
    id: int,
    angle_limit_min: int,
    angle_limit_max: int,
    baudrate: int,
) -> None:
    """Change the angle limits of the motor with the given ID on the specified serial port."""
    print(
        f"Changing angle limits for motor with ID {id} to [{angle_limit_min}, {angle_limit_max}]...",
        end="",
        flush=True,
    )
    c = Xl330PyController(
        serial_port,
        baudrate=baudrate,
        timeout=SERIAL_TIMEOUT + float(COMMANDS_BITS_LENGTH["Write"]) / baudrate,
    )
    c.write_raw_min_position_limit(id, angle_limit_min)
    c.write_raw_max_position_limit(id, angle_limit_max)
    print("[OK]")


def change_shutdown_error(
    serial_port: str, id: int, baudrate: int, shutdown_error: int
) -> None:
    """Change the shutdown error of the motor with the given ID on the specified serial port."""
    print(
        f"Changing shutdown error for motor with ID {id} to {shutdown_error}...",
        end="",
        flush=True,
    )
    c = Xl330PyController(
        serial_port,
        baudrate=baudrate,
        timeout=SERIAL_TIMEOUT + float(COMMANDS_BITS_LENGTH["Write"]) / baudrate,
    )
    c.write_shutdown(id, shutdown_error)
    print("[OK]")


def change_return_delay_time(
    serial_port: str, id: int, return_delay_time: int, baudrate: int
) -> None:
    """Change the return delay time of the motor with the given ID on the specified serial port."""
    print(
        f"Changing return delay time for motor with ID {id} to {return_delay_time}...",
        end="",
        flush=True,
    )
    c = Xl330PyController(
        serial_port,
        baudrate=baudrate,
        timeout=SERIAL_TIMEOUT + float(COMMANDS_BITS_LENGTH["Write"]) / baudrate,
    )
    c.write_return_delay_time(id, return_delay_time)
    print("[OK]")


def light_led_up(serial_port: str, id: int, baudrate: int) -> None:
    """Light the LED of the motor with the given ID on the specified serial port."""
    c = Xl330PyController(
        serial_port,
        baudrate=baudrate,
        timeout=SERIAL_TIMEOUT + float(COMMANDS_BITS_LENGTH["Write"]) / baudrate,
    )

    trials = 0

    while trials < 3:
        try:
            c.write_led(id, 1)
            break
        except RuntimeError as e:
            print(f"Error while turning on LED for motor ID {id}: {e}")
        trials += 1


def light_led_down(serial_port: str, id: int, baudrate: int) -> None:
    """Light the LED of the motor with the given ID on the specified serial port."""
    c = Xl330PyController(
        serial_port,
        baudrate=baudrate,
        timeout=SERIAL_TIMEOUT + float(COMMANDS_BITS_LENGTH["Write"]) / baudrate,
    )
    trials = 0

    while trials < 3:
        try:
            c.write_led(id, 0)
            break
        except RuntimeError as e:
            print(f"Error while turning off LED for motor ID {id}: {e}")
        trials += 1


def check_configuration(
    motor_config: MotorConfig, serial_port: str, baudrate: int
) -> None:
    """Check the configuration of the motor with the given ID on the specified serial port."""
    c = Xl330PyController(
        serial_port,
        baudrate=baudrate,
        timeout=SERIAL_TIMEOUT + float(COMMANDS_BITS_LENGTH["Read"]) / baudrate,
    )

    print("Checking configuration...")

    # Check if there is a motor with the desired ID
    if not c.ping(motor_config.id):
        raise RuntimeError(f"No motor with ID {motor_config.id} found, cannot proceed")
    print(f"Found motor with ID {motor_config.id} [OK].")

    # Read return delay time
    return_delay = c.read_return_delay_time(motor_config.id)[0]
    if return_delay != motor_config.return_delay_time:
        raise RuntimeError(
            f"Return delay time is {return_delay}, expected {motor_config.return_delay_time}"
        )
    print(f"Return delay time is correct: {return_delay} [OK].")

    # Read operating mode
    operating_mode = c.read_operating_mode(motor_config.id)[0]
    if operating_mode != motor_config.operating_mode:
        raise RuntimeError(
            f"Operating mode is {operating_mode}, expected {motor_config.operating_mode}"
        )
    print(f"Operating mode is correct: {operating_mode} [OK].")

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
        f"Angle limits are correct: [{motor_config.angle_limit_min}, {motor_config.angle_limit_max}] [OK]."
    )

    # Read homing offset
    offset = c.read_homing_offset(motor_config.id)[0]
    if offset != motor_config.offset:
        raise RuntimeError(f"Homing offset is {offset}, expected {motor_config.offset}")
    print(f"Homing offset is correct: {offset} [OK].")

    # Read shutdown
    shutdown = c.read_shutdown(motor_config.id)[0]
    if shutdown != motor_config.shutdown_error:
        raise RuntimeError(
            f"Shutdown is {shutdown}, expected {motor_config.shutdown_error}"
        )
    print(f"Shutdown error is correct: {shutdown} [OK].")

    print("Configuration is correct [OK]!")


def run(args: argparse.Namespace) -> None:
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
            print(f"[FAIL] Configuration check failed for motor '{motor_name}': {e}")
            return

        light_led_up(
            args.serialport,
            motor_config.id,
            baudrate=config.serial.baudrate,
        )


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
            "body_rotation",
            "stewart_1",
            "stewart_2",
            "stewart_3",
            "stewart_4",
            "stewart_5",
            "stewart_6",
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
