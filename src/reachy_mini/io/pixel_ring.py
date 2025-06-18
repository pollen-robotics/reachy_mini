import platform
import threading
import time
from typing import Dict, List, Optional, Tuple, Union

import serial
import serial.tools.list_ports


class NeoPixelRing:

    def __init__(self, port: str = None, baudrate: int = 9600, num_pixels: int = 12):
        """
        Initialize NeoPixel ring controller

        Args:
            port: Serial port (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux)
            baudrate: Serial communication speed (default: 9600)
            num_pixels: Number of LEDs in ring (default: 12)
        """

        def __init_routine__(self):
            self.num_pixels = self.num_pixels
            self.serial = serial.Serial(self.port, self.baudrate, timeout=2)
            time.sleep(2)  # Wait for Arduino to initialize

            # Wait for Arduino ready signal
            while True:
                if self.serial.in_waiting:
                    response = self.serial.readline().decode().strip()
                    if response == "READY":
                        break

            print(f"NeoPixel ring connected on {self.port}")

        self.num_pixels = num_pixels
        self.serial = None
        self.baudrate = baudrate
        self.port = port
        self.thread = threading.Thread(target=__init_routine__)
        self.thread.start()

    def set_colors(
        self,
        colors: Union[
            List[Optional[Tuple[int, int, int]]], Dict[int, Tuple[int, int, int]]
        ],
    ):
        """
        Set LED colors. LEDs with None values keep their previous state.

        Args:
            colors: Either a list of (r,g,b) tuples with None for unchanged LEDs,
                   or a dict with LED index as key and (r,g,b) tuple as value

        Examples:
            # List format - set first 3 LEDs, leave others unchanged
            ring.set_colors([(255,0,0), (0,255,0), (0,0,255), None, None, ...])

            # Dict format - set specific LEDs
            ring.set_colors({0: (255,0,0), 5: (0,255,0), 11: (0,0,255)})
        """
        command_parts = []

        if isinstance(colors, dict):
            # Dict format: {led_id: (r,g,b)}
            for led_id, color in colors.items():
                if 0 <= led_id < self.num_pixels and color is not None:
                    r, g, b = color
                    command_parts.append(f"{led_id},{r},{g},{b}")

        elif isinstance(colors, list):
            # List format: [(r,g,b), None, (r,g,b), ...]
            for led_id, color in enumerate(colors):
                if led_id >= self.num_pixels:
                    break
                if color is not None:
                    r, g, b = color
                    command_parts.append(f"{led_id},{r},{g},{b}")

        if command_parts:
            command = "SET:" + ";".join(command_parts) + "\n"
            self.serial.write(command.encode())

            # Wait for confirmation
            response = self.serial.readline().decode().strip()
            return response == "OK"

        return True

    def clear(self):
        """Turn off all LEDs"""
        self.serial.write(b"CLEAR\n")
        response = self.serial.readline().decode().strip()
        return response == "OK"

    def get_status(self) -> Dict[int, Tuple[int, int, int]]:
        """Get current state of all LEDs"""
        self.serial.write(b"STATUS\n")
        response = self.serial.readline().decode().strip()

        if response.startswith("STATUS:"):
            status_data = response[7:]  # Remove "STATUS:"
            led_states = {}

            for led_info in status_data.split(";"):
                parts = led_info.split(",")
                if len(parts) == 4:
                    led_id = int(parts[0])
                    r, g, b = int(parts[1]), int(parts[2]), int(parts[3])
                    led_states[led_id] = (r, g, b)

            return led_states

        return {}

    def close(self):
        """Close serial connection"""
        self.serial.close()


# Example usage and test functions
def example_usage():
    # Initialize (change COM port as needed)
    # ring = NeoPixelRing("COM3")  # Windows
    ring = NeoPixelRing("/dev/ttyUSB0")  # Linux
    # ring = NeoPixelRing()  # Linux
    # ring = NeoPixelRing('/dev/cu.usbserial-*')  # macOS

    try:
        # Example 1: Set specific LEDs using dict
        print("Setting LEDs 0, 4, 8 to different colors...")
        ring.set_colors(
            {0: (255, 0, 0), 4: (0, 255, 0), 8: (0, 0, 255)}  # Red  # Green  # Blue
        )
        time.sleep(2)

        # Example 2: Set using list (None = don't change)
        print("Adding more colors, keeping previous ones...")
        colors = [None] * 12  # Start with all None
        colors[2] = (255, 255, 0)  # Yellow
        colors[6] = (255, 0, 255)  # Magenta
        colors[10] = (0, 255, 255)  # Cyan

        ring.set_colors(colors)
        time.sleep(2)

        # Example 3: Get current status
        print("Current LED states:")
        status = ring.get_status()
        for led_id, (r, g, b) in status.items():
            if r > 0 or g > 0 or b > 0:  # Only show active LEDs
                print(f"  LED {led_id}: RGB({r}, {g}, {b})")

        # Example 4: Clear all
        print("Clearing all LEDs...")
        ring.clear()

    finally:
        ring.close()


if __name__ == "__main__":
    example_usage()
