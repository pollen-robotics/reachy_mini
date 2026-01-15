"""Scan a serial bus to find which motor IDs respond at common baudrates."""

import os
import time
from typing import List

import serial.tools.list_ports
from rustypot import Xl330PyController

SERIAL_TIMEOUT = 0.01
COMMANDS_BITS_LENGTH = {
    "Ping": (10 + 14) * 10,
    "Read": (14 + 15) * 10,
    "Write": (16 + 11) * 10,
}
XL_BAUDRATE_CONV_TABLE = {
    9600: 0,
    57600: 1,
    115200: 2,
    1000000: 3,
    2000000: 4,
    3000000: 5,
    4000000: 6,
}


def find_serial_port(
    wireless_version: bool = False,
    vid: str = "1a86",
    pid: str = "55d3",
    pi_uart: str = "/dev/ttyAMA3",
) -> list[str]:
    """Replicate from the daemon.utils.find_serial_port function."""
    # If it's a wireless version, we should use the Raspberry Pi UART
    if wireless_version:
        return [pi_uart] if os.path.exists(pi_uart) else []

    # If it's a lite version, we should find it using the VID and PID
    ports = serial.tools.list_ports.comports()

    vid = vid.upper()
    pid = pid.upper()

    return [p.device for p in ports if f"USB VID:PID={vid}:{pid}" in p.hwid]


def scan(port: str, baudrate: int) -> List[int]:
    """Scan the bus at the given baudrate and return detected IDs."""
    found_motors: list[int] = []
    try:
        controller = Xl330PyController(
            port,
            baudrate,
            float(SERIAL_TIMEOUT) + float(COMMANDS_BITS_LENGTH["Ping"]) / baudrate,
        )
        for motor_id in range(255):
            try:
                if controller.ping(motor_id):
                    found_motors.append(motor_id)
            except Exception:
                pass
    except Exception as e:
        print(f"Error while scanning port {port} at baudrate {baudrate}: {e}")
    finally:
        # CRITICAL: Close the controller to release the serial port
        if controller is not None:
            try:
                del controller
            except Exception:
                pass
        # Small delay to ensure port is fully released
        time.sleep(SERIAL_TIMEOUT)
    return found_motors


def main() -> None:
    """Iterate through baudrates and print the IDs found at each."""
    try:
        port = find_serial_port()[0]
    except IndexError:
        print("No serial port found. Please check your USB connection and permissions.")
        return
    for baudrate in XL_BAUDRATE_CONV_TABLE.keys():
        print(f"Trying baudrate: {baudrate}")
        found_motors = scan(port, baudrate)
        if found_motors:
            print(f"Found motors at baudrate {baudrate}: {found_motors}")
        else:
            print(f"No motors found at baudrate {baudrate}")


if __name__ == "__main__":
    main()
