from reachy_mini import MujocoBackend, RobotBackend
from reachy_mini.io import Server
import argparse


class Daemon:
    def __init__(
        self, sim: bool = False, serialport: str = "/dev/ttyACM0", scene: str = "empty"
    ):
        if sim:
            self.backend = MujocoBackend(scene=scene)
        else:
            self.backend = RobotBackend(serialport=serialport)

        self.server = Server(self.backend, localhost_only=False)
        self.server.start()

    def run(self):
        try:
            self.backend.run()
        except:
            self.server.stop()
            raise


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
    args = parser.parse_args()

    d = Daemon(sim=args.sim, serialport=args.serialport, scene=args.scene)
    d.run()


if __name__ == "__main__":
    main()
