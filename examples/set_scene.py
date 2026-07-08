"""Set the two-robot opening pose for the video, then exit.

Sends the dominant pose to the Lite (localhost) and the submissive pose to the
wireless robot (reachy-mini.local) at the same time, with a smooth goto, and
quits so the robots stay put and the handshakes keep working. Built to be run
over and over between takes.

Default mapping (override with --lite-id / --wireless-id):
    Lite     <- pose 3 (dominant)
    wireless <- pose 2 (submissive)

Note:
    Both daemons must be running (Lite on localhost, wireless on
    reachy-mini.local). Poses come from recorded_poses.json next to this script.
"""

import argparse
import json
import sys
from pathlib import Path
from threading import Thread

import numpy as np

from reachy_mini import ReachyMini

POSES_FILE = Path(__file__).with_name("recorded_poses.json")


def send(robot: str, pose: dict, duration: float) -> None:
    """Connect to one robot, smoothly go to the pose, and disconnect."""
    if robot == "wireless":
        mini = ReachyMini(
            media_backend="no_media",
            connection_mode="network",
            host="reachy-mini.local",
        )
    else:
        mini = ReachyMini(media_backend="no_media", connection_mode="localhost_only")
    with mini:
        mini.enable_motors()
        mini.goto_target(
            head=np.array(pose["head"], dtype=float),
            antennas=pose.get("antennas"),
            duration=duration,
        )


def main() -> None:
    """Send both poses concurrently, then exit."""
    parser = argparse.ArgumentParser(description="Set the two-robot video pose.")
    parser.add_argument("--lite-id", type=int, default=3, help="Pose ID for the Lite.")
    parser.add_argument(
        "--wireless-id", type=int, default=2, help="Pose ID for the wireless robot."
    )
    parser.add_argument(
        "--duration", type=float, default=1.0, help="Move duration in seconds."
    )
    args = parser.parse_args()

    if not POSES_FILE.exists():
        sys.exit(f"No pose file at {POSES_FILE}")
    poses = {p["id"]: p for p in json.loads(POSES_FILE.read_text())}
    for pid in (args.lite_id, args.wireless_id):
        if pid not in poses:
            sys.exit(f"No pose {pid} in {POSES_FILE}. Available: {sorted(poses)}")

    threads = [
        Thread(target=send, args=("lite", poses[args.lite_id], args.duration)),
        Thread(target=send, args=("wireless", poses[args.wireless_id], args.duration)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"Lite <- pose {args.lite_id}, wireless <- pose {args.wireless_id}. Done.")


if __name__ == "__main__":
    main()
