"""Live low-level collision probe. Prints the geometric signals raw.

Lower-level than live_handshake_probe.py: no state machine, no beeps, just
the exact per-tick numbers the detector sees. Use it to stare at the signal
while playing with the antennas and to tune CollisionConfig:

    l    left antenna (index 0), degrees
    r    right antenna (index 1), degrees
    sum  l + r, degrees: in collision it sits in the narrow band
         (default [-9, 0], see CollisionConfig)
    the collision needs BOTH sum in the band AND l in [20, 150] deg

It disables torque at startup (hold the head if it is up: it will slump) so
the antennas are floppy and ready to play with; pass --keep-torque to leave
the motors as they are. It never moves the robot.

Run on the robot (or against a robot on the LAN):
    python examples/secret_handshake_lab/live_contact_probe.py
"""

from __future__ import annotations

import argparse
import math
import time

from collision import CollisionConfig, CollisionDetector

from reachy_mini import ReachyMini


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep-torque",
        action="store_true",
        help="do not touch the motors at startup (default: disable them)",
    )
    args = parser.parse_args()

    cfg = CollisionConfig()
    det = CollisionDetector(cfg)
    print(
        f"band: sum in [{det.sum_lo_deg:.0f}, {det.sum_hi_deg:.0f}] deg "
        f"(measured [{cfg.sum_min_deg:.0f}, {cfg.sum_max_deg:.0f}] "
        f"+ {cfg.margin_deg:.0f} deg margin), l in "
        f"[{cfg.l_min_deg:.0f}, {cfg.l_max_deg:.0f}] deg"
    )
    print("Reading present antenna positions at 50 Hz. Ctrl-C to stop.")

    count = 0
    with ReachyMini(media_backend="no_media") as mini:
        if not args.keep_torque:
            print("disabling motors (torque off), hold the head if it is up...\n")
            mini.disable_motors()
            time.sleep(1.0)
        while True:
            t = time.monotonic()
            ant0, ant1 = mini.get_present_antenna_joint_positions()
            onset = det.update(t, ant0, ant1)
            if onset:
                count += 1
            state = "IN-BAND" if det.in_collision else "  ...  "
            flag = f"  <-- collision #{count}" if onset else ""
            print(
                f"l={math.degrees(ant0):+7.1f}  r={math.degrees(ant1):+7.1f}  "
                f"sum={det.sum_deg:+6.1f}  {state}{flag}",
                end="\r" if not onset else "\n",
                flush=True,
            )
            time.sleep(0.02)


if __name__ == "__main__":
    main()
