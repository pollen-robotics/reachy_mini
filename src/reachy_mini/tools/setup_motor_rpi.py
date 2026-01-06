"""You need to `pip install gpiozero lgpio`."""

import argparse
import os
import time
from typing import List

import numpy as np
from gpiozero import DigitalOutputDevice
from setup_motor import (
    FACTORY_DEFAULT_BAUDRATE,
    FACTORY_DEFAULT_ID,
    light_led_down,
    lookup_for_motor,
    parse_yaml_config,
    run,
)

assets_root_path = "../src/reachy_mini/assets/"


UART_PORT = "/dev/ttyAMA3"
CONFIG_FILE_PATH = os.path.join(assets_root_path, "config", "hardware_config.yaml")

ID_TO_CHANNEL = {
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
CHANNEL_TO_ID = {v: k for k, v in ID_TO_CHANNEL.items()}

S0 = DigitalOutputDevice(25)
S1 = DigitalOutputDevice(8)
S2 = DigitalOutputDevice(7)
S3 = DigitalOutputDevice(1)


def get_channel_binary(channel: int) -> List[int]:
    """Convert channel number (0-8) to 4-bit binary representation."""
    assert channel in np.arange(9), "Channel must be between 0 and 8"
    bits = [int(b) for b in f"{channel:04b}"]  # 4-bit binary
    return bits[::-1]  # flip the order


def select_channel(channel: int) -> None:
    """Select a channel on the multiplexer."""
    bits = get_channel_binary(channel)
    S0.on() if bits[0] else S0.off()
    S1.on() if bits[1] else S1.off()
    S2.on() if bits[2] else S2.off()
    S3.on() if bits[3] else S3.off()


def main() -> None:
    """Scan all channels of the multiplexer to find motors in factory default state, and set them up one by one."""
    config = parse_yaml_config(CONFIG_FILE_PATH)
    motor_name_to_id = {m: config.motors[m].id for m in config.motors}
    id_to_motor_name = {v: k for k, v in motor_name_to_id.items()}

    print("Starting motor setup...")
    current_channel = 0
    while True:
        current_channel = (current_channel + 1) % 9
        select_channel(current_channel)
        target_id = CHANNEL_TO_ID[current_channel]
        target_name = id_to_motor_name[target_id]
        if lookup_for_motor(
            UART_PORT, FACTORY_DEFAULT_ID, FACTORY_DEFAULT_BAUDRATE, silent=True
        ):
            print(f"Found motor on channel {current_channel}!")
            args = argparse.Namespace(
                config_file=CONFIG_FILE_PATH,
                motor_name=target_name,
                serialport=UART_PORT,
                check_only=False,
                from_id=FACTORY_DEFAULT_ID,
                from_baudrate=FACTORY_DEFAULT_BAUDRATE,
                update_config=False,
            )
            run(args)

        elif lookup_for_motor(UART_PORT, current_channel + 10, 1000000, silent=True):
            print(f"Motor on channel {current_channel} already set up.")
            # light_led_up(UART_PORT, current_channel+10, 1000000)
            light_led_down(UART_PORT, current_channel + 10, 1000000)
            args = argparse.Namespace(
                config_file=CONFIG_FILE_PATH,
                motor_name=target_name,
                serialport=UART_PORT,
                check_only=False,
                from_id=current_channel + 10,
                from_baudrate=1000000,
                update_config=False,
            )
            run(args)
            time.sleep(2)

        time.sleep(0.01)


if __name__ == "__main__":
    main()
