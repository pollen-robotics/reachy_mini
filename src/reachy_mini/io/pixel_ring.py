import platform
import threading
import time
from collections import deque
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

        self.num_pixels = num_pixels
        self.port = port
        self.serial = serial.Serial(port, baudrate, timeout=2)
        self.is_connected = threading.Event()
        self.com_thread = threading.Thread(target=self.com_routine, daemon=True)
        self.com_thread.start()
        self.commands = deque(maxlen=1)  # Buffer for commands
        self.commands_lock = threading.Lock()

    def wait_for_connection(self):
        while not self.is_connected.wait(timeout=5.0):
            print("Waiting for Arduino ring leds to be ready. ")

    def com_routine(self):
        while (
            True
        ):  # TODO useless for now, but will handle reconnection with this later

            # Wait for Arduino ready signal
            while True:
                if self.serial.in_waiting:
                    response = self.serial.readline().decode().strip()
                    if response == "READY":
                        break

            print(f"NeoPixel ring connected on {self.port}")
            self.is_connected.set()
            self._running = True

            while self._running:
                # Process commands from queue
                command_data = None
                with self.commands_lock:
                    if self.commands:
                        command_data = self.commands.popleft()

                if command_data:
                    cmd_type, args = command_data

                    # Execute the appropriate command
                    if cmd_type == "set_led_colors":
                        self._set_led_colors(args)
                    elif cmd_type == "clear":
                        self._clear()
                    elif cmd_type == "get_status":
                        # For get_status, we might want to store result somewhere
                        # but for now just execute it
                        self._get_status()
                else:
                    # Shouldnt end up here
                    time.sleep(0.01)

    def _set_led_colors(
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

    def _clear(self):
        """Turn off all LEDs"""
        self.serial.write(b"CLEAR\n")
        response = self.serial.readline().decode().strip()
        return response == "OK"

    def _get_status(self) -> Dict[int, Tuple[int, int, int]]:
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

    def _close(self):
        """Close serial connection"""
        self.serial.close()

    # Exposed commands
    def set_led_colors(
        self,
        colors: Union[
            List[Optional[Tuple[int, int, int]]], Dict[int, Tuple[int, int, int]]
        ],
    ):
        """Queue a set_led_colors command (non-blocking)"""
        with self.commands_lock:
            self.commands.append(("set_led_colors", colors))

    def clear(self):
        """Queue a clear command (non-blocking)"""
        with self.commands_lock:
            self.commands.append(("clear", None))

    def get_status(self):
        """Queue a get_status command (non-blocking)"""
        # with self.commands_lock:
        #     self.commands.append(("get_status", None))
        # TODO async status or matintain buffer of current status ?
        return self._get_status()  # Directly return status for now

    def close(self):
        """Close serial connection and stop thread"""
        self._running = False
        if self.com_thread.is_alive():
            self.com_thread.join(timeout=2.0)
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
