import argparse
from typing import List

import numpy as np
from gpiozero import DigitalOutputDevice
from rustypot import Xl330PyController

SERIAL_TIMEOUT = 0.1  # seconds

S0 = DigitalOutputDevice(25)
S1 = DigitalOutputDevice(8)
S2 = DigitalOutputDevice(7)
S3 = DigitalOutputDevice(1)


def get_channel_binary(channel) -> List[int]:
    """Convert channel number (0-8) to 4-bit binary representation."""
    assert channel in np.arange(9), "Channel must be between 0 and 8"
    bits = [int(b) for b in f"{channel:04b}"]  # 4-bit binary
    return bits[::-1]  # flip the order


def select_channel(channel: int):
    """Select a channel on the multiplexer."""
    bits = get_channel_binary(channel)
    S0.value = bits[0]
    S1.value = bits[1]
    S2.value = bits[2]
    S3.value = bits[3]


def lookup_for_motor(serial_port: str, id: int, baudrate: int) -> bool:
    """Check if a motor with the given ID is reachable on the specified serial port."""
    print(
        f"Looking for motor with ID {id} on port {serial_port}...",
        end="",
        flush=True,
    )
    c = Xl330PyController(serial_port, baudrate=baudrate, timeout=SERIAL_TIMEOUT)
    ret = c.ping(id)
    print(f"{'✅' if ret else '❌'}")
    return ret


id_to_channel = {
    10: 0,
    11: 1,
    12: 2,
    13: 3,
    14: 4,
    15: 5,
    16: 6,
    17: 7,
    18: 8,
}
channel_to_id = {v: k for k, v in id_to_channel.items()}

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--id", type=int, default=10, help="Motor ID (10-18)")
    args.add_argument("--serial", type=str, default="/dev/ttyAMA3", help="Serial port")
    args = args.parse_args()

    channel = id_to_channel.get(args.id, None)

    select_channel(channel)
    print(f"Selected channel {channel}")
    baudrates = [57600, 1_000_000]
    ok = False
    for baudrate in baudrates:
        if ok:
            break
        print("Trying baudrate:", baudrate)
        for i in range(255):
            if ok:
                break
            ret = lookup_for_motor(args.serial, i, baudrate=baudrate)
            if ret:
                ok = True
