from threading import Thread

import argparse
from typing import Optional
import serial.tools.list_ports
import time

from reachy_mini import MujocoBackend, RobotBackend, ReachyMini
from reachy_mini.io import Server


class Daemon:
    def __init__(
        self,
        sim: bool = False,
        serialport: str = "auto",
        scene: str = "empty",
        localhost_only: bool = True,
        wake_up_on_start: bool = True,
        goto_sleep_on_stop: bool = True,
    ):
        if sim:
            self.backend = MujocoBackend(scene=scene)
        else:
            if serialport == "auto":
                print("Searching for Reachy Mini serial port...")
                ports = find_serial_port()
                print(f"Found Reachy Mini serial ports: {ports}")

                if len(ports) == 0:
                    raise RuntimeError(
                        "No Reachy Mini serial port found. "
                        "Check USB connection and permissions."
                        "Or directly specify the serial port using --serialport."
                    )
                elif len(ports) > 1:
                    raise RuntimeError(
                        "Multiple Reachy Mini serial ports found. "
                        "Please specify the serial port using --serialport."
                    )

                serialport = ports[0]

            self.backend = RobotBackend(serialport=serialport)

        self.wake_up_on_start = wake_up_on_start
        self.goto_sleep_on_stop = goto_sleep_on_stop

        self.server = Server(self.backend, localhost_only=localhost_only)
        self.server.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.server.stop()

    def run(self):
        """Run the daemon."""
        print("Starting Reachy Mini daemon...")
        backend_run_thread = Thread(target=self.backend.run)
        backend_run_thread.start()

        if self.wake_up_on_start:
            print("Waking up Reachy Mini...")
            with ReachyMini() as mini:
                mini.set_torque(on=True)
                mini.wake_up()

        try:
            while True:
                time.sleep(1)  # Keep the daemon running
        except KeyboardInterrupt:
            print("Daemon interrupted by user.")

        if self.goto_sleep_on_stop:
            print("Putting Reachy Mini to sleep...")
            with ReachyMini() as mini:
                mini.goto_sleep()

        self.backend.should_stop.set()
        backend_run_thread.join()

        self.server.stop()
        print("Daemon stopped.")


def find_serial_port(vid: str = "1a86", pid: str = "55d3") -> list[str]:
    ports = serial.tools.list_ports.comports()

    vid = vid.upper()
    pid = pid.upper()

    return [p.device for p in ports if f"USB VID:PID={vid}:{pid}" in p.hwid]


def main():
    """Monkey patch to run the main function using the mjpython executable on macOS."""
    import platform

    if platform.system() != "Darwin":
        return _main()

    import multiprocessing as mp
    import sys

    python_exec = sys.executable
    python_exec = python_exec.removesuffix("python") + "mjpython"
    mp.set_executable(python_exec)

    p = mp.Process(target=_main)
    p.start()
    p.join()
    return p.exitcode


def _main():
    parser = argparse.ArgumentParser(description="Run the Reachy Mini daemon.")
    parser.add_argument(
        "--sim",
        action="store_true",
        help="Run in simulation mode using Mujoco.",
    )
    parser.add_argument(
        "-p",
        "--serialport",
        type=str,
        default="auto",
        help="Serial port for real motors (default: will try to automatically find the port).",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="empty",
        help="Name of the scene to load (default: empty)",
    )
    parser.add_argument(
        "--localhost-only",
        type=bool,
        default=True,
        help="Restrict the server to localhost only (default: True).",
    )
    args = parser.parse_args()

    d = Daemon(
        sim=args.sim,
        serialport=args.serialport,
        scene=args.scene,
        localhost_only=args.localhost_only,
    )
    d.run()


if __name__ == "__main__":
    main()
