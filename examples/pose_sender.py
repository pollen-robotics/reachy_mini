"""Send a recorded pose to a robot, then exit.

Reads a pose (by ID) from the JSON file written by pose_recorder.py, enables
torque, moves there, and quits. Everything else the robot does (handshakes,
etc.) keeps working after.

Note:
    The daemon must be running. By default this connects to the Lite robot on
    localhost; pass --robot wireless to reach reachy-mini.local (or run this
    script directly on the wireless robot, where localhost is the wireless one).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from reachy_mini import ReachyMini


def connect(robot: str) -> ReachyMini:
    """Connect to the Lite (localhost) or wireless (reachy-mini.local) robot."""
    if robot == "wireless":
        return ReachyMini(
            media_backend="no_media",
            connection_mode="network",
            host="reachy-mini.local",
        )
    return ReachyMini(media_backend="no_media", connection_mode="localhost_only")


def main() -> None:
    """Send one recorded pose to the robot and exit."""
    parser = argparse.ArgumentParser(description="Send a recorded pose to a robot.")
    parser.add_argument("id", type=int, help="Pose ID to send (see pose_recorder.py).")
    parser.add_argument(
        "--robot",
        choices=["lite", "wireless"],
        default="lite",
        help="Which robot to send to (default: lite / localhost).",
    )
    parser.add_argument(
        "--file",
        default="recorded_poses.json",
        help="JSON file of recorded poses (default: recorded_poses.json).",
    )
    parser.add_argument(
        "--duration", type=float, default=1.0, help="Move duration in seconds."
    )
    parser.add_argument(
        "--no-antennas", action="store_true", help="Send only the head pose."
    )
    args = parser.parse_args()

    path = Path(args.file).resolve()
    if not path.exists():
        sys.exit(f"No pose file at {path}")
    poses = {p["id"]: p for p in json.loads(path.read_text())}
    if args.id not in poses:
        sys.exit(f"No pose with ID {args.id} in {path}. Available: {sorted(poses)}")

    pose = poses[args.id]
    head = np.array(pose["head"], dtype=float)
    antennas = None if args.no_antennas else pose.get("antennas")

    with connect(args.robot) as mini:
        mini.enable_motors()
        mini.goto_target(head=head, antennas=antennas, duration=args.duration)
    print(f"Sent pose {args.id} to {args.robot}.")


if __name__ == "__main__":
    main()
