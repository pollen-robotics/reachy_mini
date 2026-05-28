"""Cloud-backend consumer example: read a Reachy Mini camera via the HF
central signaling relay, with no daemon and no hardware on this side.

This mirrors what you'd run inside a Hugging Face Space, Cloud Run
container, or any other hardware-free Python backend that needs to read
a visitor's robot camera (e.g. to run inference) and optionally drive
the robot via the visitor's HF Bearer token.

Two flows are demonstrated:

1. Pulling decoded frames into a numpy array (RGB24) at ~30 fps and
   running a stand-in "inference" on them (here: just print the mean
   brightness).
2. Sending a one-shot ``goto_target`` head command on the data channel
   the daemon offers — same wire format the SDK speaks.

Run::

    export HF_TOKEN=hf_...    # token of the account that owns the robot
    python central_consumer_cloud_backend.py \\
        --robot-peer-id <peer-id-of-your-reachy>

If you omit ``--robot-peer-id``, the consumer will auto-pick the first
robot in central's listing whose ``meta.name`` matches ``--robot-name``
(default ``reachymini``).

Note:
    No GStreamer / no daemon required — only the ``aiortc`` Python
    package (plus ``aiohttp``, ``numpy``).

"""

# START doc_example

from __future__ import annotations

import argparse
import asyncio
import math
import os
import sys
from typing import Optional

import numpy as np

from reachy_mini.media.central_consumer import ReachyCentralConsumer


def _rpy_to_pose(roll_deg: float, pitch_deg: float, yaw_deg: float) -> list[float]:
    """Tiny RPY→4x4 helper (ZYX, row-major, returns a 16-float list).

    The same convention ``reachy_mini.ReachyMini.set_target`` and
    ``goto_target`` use on the wire.
    """
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    cy, sy = math.cos(y), math.sin(y)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)
    m = [
        cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, 0.0,
        sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, 0.0,
        -sp,     cp * sr,                cp * cr,                0.0,
        0.0,     0.0,                    0.0,                    1.0,
    ]
    return m


async def run(
    hf_token: str,
    robot_peer_id: Optional[str],
    robot_name: str,
    duration_s: float,
) -> None:
    consumer = ReachyCentralConsumer(
        hf_token=hf_token,
        robot_peer_id=robot_peer_id,
        robot_name=robot_name,
        consumer_label="examples/central-consumer-cloud-backend",
    )
    await consumer.start()
    try:
        start_t = asyncio.get_running_loop().time()
        last_id = -1
        sent_goto = False
        while asyncio.get_running_loop().time() - start_t < duration_s:
            snap = consumer.latest_frame()
            if snap is not None and snap[0] != last_id:
                last_id, rgb = snap
                # Stand-in for whatever inference you'd run on the
                # frame: just print the mean brightness every 30 frames.
                if last_id % 30 == 0:
                    print(
                        f"[frame {last_id}] shape={rgb.shape} "
                        f"mean={float(np.mean(rgb)):.1f}"
                    )

            # Once we're connected, send one ``goto_target`` ("nod down a
            # bit, then back to neutral") to demonstrate the command path.
            if not sent_goto and consumer.is_command_ready():
                print("[example] sending goto_target: pitch=15° for 0.6s")
                consumer.send_command({
                    "type": "goto_target",
                    "head": _rpy_to_pose(0.0, 15.0, 0.0),
                    "duration": 0.6,
                    "body_yaw": None,
                    "antennas": None,
                })
                sent_goto = True

            await asyncio.sleep(1.0 / 30.0)

        # Polite return to neutral on exit.
        if consumer.is_command_ready():
            print("[example] returning head to neutral")
            consumer.send_command({
                "type": "goto_target",
                "head": _rpy_to_pose(0.0, 0.0, 0.0),
                "duration": 0.6,
                "body_yaw": None,
                "antennas": None,
            })
            await asyncio.sleep(0.7)
    finally:
        await consumer.stop()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Hardware-free cloud-backend consumer: read a Reachy Mini "
            "camera via the HF central relay and send a head command."
        )
    )
    parser.add_argument(
        "--robot-peer-id",
        default=os.getenv("REACHY_ROBOT_PEER_ID"),
        help=(
            "Pin to a specific robot peerId on central. If omitted, "
            "auto-pick the first robot whose meta.name matches "
            "--robot-name (default: 'reachymini')."
        ),
    )
    parser.add_argument(
        "--robot-name",
        default=os.getenv("REACHY_ROBOT_NAME", "reachymini"),
        help="Robot meta.name to match in central's listing.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=10.0,
        help="How long to keep the consumer alive before exiting.",
    )
    args = parser.parse_args()

    hf_token = os.getenv("HF_TOKEN", "").strip()
    if not hf_token:
        print(
            "Error: set HF_TOKEN to the HF Bearer token of the account "
            "that owns the robot you want to consume.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        asyncio.run(run(hf_token, args.robot_peer_id, args.robot_name, args.seconds))
    except KeyboardInterrupt:
        print("\nInterrupted, exiting.")


if __name__ == "__main__":
    main()

# END doc_example
