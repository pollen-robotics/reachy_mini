import argparse
import time
from threading import Thread

from reachy_mini import MujocoBackend, ReachyMini, RobotBackend
from reachy_mini.io import Server
from reachy_mini.utils import find_arduino_nano_ch340g


class Daemon:
    def __init__(
        self,
        sim: bool = False,
        serialport: str = "/dev/ttyACM0",
        scene: str = "empty",
        localhost_only: bool = True,
        wake_up_on_start: bool = True,
        goto_sleep_on_stop: bool = True,
        led_ring_port: str = None,
    ):
        # Try ardunio discovery
        if led_ring_port is None:
            led_ring_port = find_arduino_nano_ch340g()

        # if led_ring_port is None:
        #     print(
        #         "No NeoPixel ring found. "
        #         "You can specify the port with --led-ring-port."
        #     )
        # else:
        #     print(f"NeoPixel ring found on port: {led_ring_port}")

        if sim:
            self.backend = MujocoBackend(scene=scene)
        else:
            self.backend = RobotBackend(
                serialport=serialport, led_ring_port=led_ring_port
            )

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
                # time.sleep(3)  # Allow some time for the motors to wake up
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
        default="/dev/ttyACM0",
        help="Serial port for real motors (default: /dev/ttyACM0).",
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
    parser.add_argument(
        "--led-ring-port",
        type=str,
        default=None,
        help="Serial port for the NeoPixel ring (default: None, auto-detect).",
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
