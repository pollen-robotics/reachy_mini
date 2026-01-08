"""Scan a serial bus to find which motor IDs respond at common baudrates."""

from typing import List

from rustypot import Xl330PyController

baudrates: List[int] = [9600, 57600, 115200, 1000000]


def scan(baudrate: int) -> List[int]:
    """Scan the bus at the given baudrate and return detected IDs."""
    controller = Xl330PyController("/dev/ttyAMA3", baudrate, 0.01)
    found_motors: list[int] = []
    for motor_id in range(255):
        if controller.ping(motor_id):
            found_motors.append(motor_id)
    return found_motors


def main() -> None:
    """Iterate through baudrates and print the IDs found at each."""
    for baudrate in baudrates:
        print(f"Trying baudrate: {baudrate}")
        found_motors = scan(baudrate)
        if found_motors:
            print(f"Found motors at baudrate {baudrate}: {found_motors}")
        else:
            print(f"No motors found at baudrate {baudrate}")


if __name__ == "__main__":
    main()
