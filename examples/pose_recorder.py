"""Record head poses by hand.

Workflow:
    - SPACE toggles torque. Torque OFF -> move the head by hand; torque ON ->
      it holds where you left it.
    - S saves the current pose to a JSON file, printing its numeric ID.
    - Q (or Ctrl-C) quits.

Saved poses accumulate in the file (IDs 1, 2, 3, ...) so you can record a few,
note which ID is which, then replay any of them with pose_sender.py.

Note:
    The daemon must be running. By default this connects to the Lite robot on
    localhost; pass --robot wireless to reach reachy-mini.local.
"""

import argparse
import json
import sys
import termios
import tty
from pathlib import Path

import numpy as np

from reachy_mini import ReachyMini


def read_key() -> str:
    """Read a single keypress from the terminal (raw mode)."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def connect(robot: str) -> ReachyMini:
    """Connect to the Lite (localhost) or wireless (reachy-mini.local) robot."""
    if robot == "wireless":
        return ReachyMini(
            media_backend="no_media",
            connection_mode="network",
            host="reachy-mini.local",
        )
    return ReachyMini(media_backend="no_media", connection_mode="localhost_only")


def next_id(poses: list[dict]) -> int:
    """Next sequential ID (1-based)."""
    return max((p["id"] for p in poses), default=0) + 1


def main() -> None:
    """Run the interactive pose recorder."""
    parser = argparse.ArgumentParser(description="Record head poses by hand.")
    parser.add_argument(
        "--robot",
        choices=["lite", "wireless"],
        default="lite",
        help="Which robot to connect to (default: lite / localhost).",
    )
    parser.add_argument(
        "--file",
        default="recorded_poses.json",
        help="JSON file to append poses to (default: recorded_poses.json).",
    )
    args = parser.parse_args()

    path = Path(args.file).resolve()
    poses = json.loads(path.read_text()) if path.exists() else []

    print(f"Saving to {path}")
    print("SPACE = toggle torque | S = save pose | Q = quit")

    with connect(args.robot) as mini:
        torque_on = True
        mini.enable_motors()
        print("Torque ON (holding). Press SPACE to release and move the head.")
        try:
            while True:
                key = read_key()
                if key in ("q", "\x03"):  # q or Ctrl-C
                    break
                if key == " ":
                    torque_on = not torque_on
                    if torque_on:
                        mini.enable_motors()
                        print("Torque ON  (holding current pose)")
                    else:
                        mini.disable_motors()
                        print("Torque OFF (move the head by hand)")
                elif key in ("s", "S"):
                    head = np.array(mini.get_current_head_pose(), dtype=float)
                    antennas = list(mini.get_present_antenna_joint_positions())
                    pid = next_id(poses)
                    poses.append(
                        {
                            "id": pid,
                            "head": head.tolist(),
                            "antennas": antennas,
                        }
                    )
                    path.write_text(json.dumps(poses, indent=2))
                    print(
                        f"Saved pose {pid}: "
                        f"pos={np.round(head[:3, 3], 4).tolist()} "
                        f"antennas={np.round(antennas, 3).tolist()}"
                    )
        finally:
            mini.enable_motors()
            print("\nTorque ON. Bye.")


if __name__ == "__main__":
    main()
