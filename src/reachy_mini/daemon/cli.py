"""Daemon entry point for the Reachy Mini robot.

This script serves as the command-line interface (CLI) entry point for the Reachy Mini daemon.
It initializes the daemon with specified parameters such as simulation mode, serial port,
scene to load, and logging level. The daemon runs indefinitely, handling requests and
managing the robot's state.

Arguments:
    -p, --serialport: Serial port for real motors (default: will try to automatically find the port).
    --sim: Run in simulation mode using Mujoco.
    --scene: Name of the scene to load in Mujoco (default: empty).
    --localhost-only: Restrict the server to localhost only (default: True).
    --no-localhost-only: Allow the server to listen on all interfaces (default: False).
    --log-level: Set the logging level (default: INFO).

"""

import logging

from reachy_mini.daemon.daemon import Daemon


def main():
    """Cli entry point for the Reachy Mini daemon."""
    import argparse

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
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO).",
    )
    parser.add_argument(
        "--check-collision",
        action="store_true",
        default=False,
        help="Enable collision checking (default: False).",
    )

    parser.add_argument(
        "--kinematics-engine",
        type=str,
        default="Placo",
        choices=["Placo", "NN", "Analytical", "CPPAnalytical"],
        help="Set the kinematics engine (default: Placo).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run Mujoco in headless mode (no GUI) (default: False).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    Daemon(log_level=args.log_level).run4ever(
        sim=args.sim,
        serialport=args.serialport,
        scene=args.scene,
        localhost_only=args.localhost_only,
        check_collision=args.check_collision,
        kinematics_engine=args.kinematics_engine,
        headless=args.headless,
    )


if __name__ == "__main__":
    main()
