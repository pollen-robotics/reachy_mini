import argparse
import signal
import time
from threading import Thread

import serial.tools.list_ports

from reachy_mini import MujocoBackend, ReachyMini, RobotBackend
from reachy_mini.io import Server


def signal_handler(signum, frame):
    """Handle termination signals"""
    print("Daemon already  stopping...")


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

    def run(self):
        """Run the daemon."""

        print("Starting Reachy Mini daemon...")
        backend_run_thread = Thread(target=self.backend.run)
        backend_run_thread.start()

        ok = True

        if self.wake_up_on_start:
            try:
                print("Waking up Reachy Mini...")
                with ReachyMini() as mini:
                    mini.set_torque(on=True)
                    mini.wake_up()
            except Exception as e:
                print(f"Error while waking up Reachy Mini: {e}")
                ok = False
            except KeyboardInterrupt:
                print("Daemon interrupted by user.")
                ok = False

        if ok:
            try:
                print("Daemon is running. Press Ctrl+C to stop.")
                while backend_run_thread.is_alive():
                    time.sleep(0.5)
                else:
                    print("Backend thread has stopped unexpectedly.")
                    ok = False

                    time.sleep(0.5)  # Wait for the backend to be ready
            except KeyboardInterrupt:
                print("Daemon interrupted by user.")
                signal.signal(
                    signal.SIGINT, signal_handler
                )  # catch Ctrl+C to avoid interrupting the proper shutdown

        if self.goto_sleep_on_stop and ok:
            try:
                print("Putting Reachy Mini to sleep...")
                with ReachyMini() as mini:
                    mini.goto_sleep()
                    mini.set_torque(on=False)
            except Exception as e:
                print(f"Error while putting Reachy Mini to sleep: {e}")
            except KeyboardInterrupt:
                pass

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
        action="store_true",
        default=True,
        help="Restrict the server to localhost only (default: True).",
    )
    parser.add_argument(
        "--no-localhost-only",
        action="store_false",
        dest="localhost_only",
        help="Allow the server to listen on all interfaces (default: False).",
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
