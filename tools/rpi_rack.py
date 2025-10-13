import argparse
from typing import List

import numpy as np
from gpiozero import DigitalOutputDevice

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


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--channel", type=int, default=0, help="Channel number (0-8)")
    args = args.parse_args()

    select_channel(args.channel)
    print(f"Selected channel {args.channel}")
