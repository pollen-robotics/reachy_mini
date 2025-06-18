from threading import Thread

import argparse
import time

from reachy_mini import MujocoBackend, RobotBackend, ReachyMini
from reachy_mini.io import Server


class Daemon:
    def __init__(
        self,
        sim: bool = False,
        serialport: str = "/dev/ttyACM0",
        scene: str = "empty",
        localhost_only: bool = True,
        wake_up_on_start: bool = True,
        goto_sleep_on_stop: bool = True,
    ):
        if sim:
            self.backend = MujocoBackend(scene=scene)
        else:
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
                while True:
                    time.sleep(0.5)  # Wait for the backend to be ready
            except KeyboardInterrupt:
                print("Daemon interrupted by user.")

            if self.goto_sleep_on_stop:
                try:
                    print("Putting Reachy Mini to sleep...")
                    with ReachyMini() as mini:
                        mini.goto_sleep()
                except Exception as e:
                    print(f"Error while putting Reachy Mini to sleep: {e}")
                except KeyboardInterrupt:
                    pass

        self.backend.should_stop.set()
        backend_run_thread.join()

        self.server.stop()
        print("Daemon stopped.")


def run_daemon(args: argparse.Namespace):
    d = Daemon(
        sim=args.sim,
        serialport=args.serialport,
        scene=args.scene,
        localhost_only=args.localhost_only,
    )
    d.run()


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
    args = parser.parse_args()

    # Monkey patch to run the main function using the mjpython executable on macOS
    import multiprocessing as mp
    import platform
    import sys

    python_exec = sys.executable

    if platform.system() == "Darwin" and args.sim and "mjpython" not in python_exec:
        mjpython_exec = python_exec.removesuffix("python") + "mjpython"
        mp.set_executable(mjpython_exec)

        print(f"Running Reachy Mini daemon with mjpython... {mjpython_exec}")
        p = mp.Process(target=run_daemon, args=(args,))
        p.start()
        try:
            p.join()
        except KeyboardInterrupt:
            p.terminate()
        return p.exitcode

    else:
        print(f"Running Reachy Mini daemon with {python_exec}...")
        run_daemon(args)


if __name__ == "__main__":
    main()
