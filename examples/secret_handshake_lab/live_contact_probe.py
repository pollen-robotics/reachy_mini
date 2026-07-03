"""Live low-level collision probe. Prints the geometric signals raw.

Lower-level than live_handshake_probe.py: no state machine, no beeps, just
the exact per-tick numbers the detector sees. Use it to stare at the signal
while playing with the antennas and to tune CollisionConfig:

    l    left antenna (index 0), degrees
    r    right antenna (index 1), degrees
    sum  l + r, degrees: in collision it sits in the narrow band
         (default [-9, -1] = measured [-7, -3] plus a 4 deg margin)
    the collision needs BOTH sum in the band AND l in [20, 150] deg

It does NOT enable torque and does NOT move the robot. Antennas stay floppy
so you can play with them.

Run on the robot (or against a robot on the LAN):
    python examples/secret_handshake_lab/live_contact_probe.py
"""

from __future__ import annotations

import math
import time

from collision import CollisionConfig, CollisionDetector

from reachy_mini import ReachyMini


def main() -> None:
    cfg = CollisionConfig()
    det = CollisionDetector(cfg)
    print(
        f"band: sum in [{det.sum_lo_deg:.0f}, {det.sum_hi_deg:.0f}] deg "
        f"(measured [{cfg.sum_min_deg:.0f}, {cfg.sum_max_deg:.0f}] "
        f"+ {cfg.margin_deg:.0f} deg margin), l in "
        f"[{cfg.l_min_deg:.0f}, {cfg.l_max_deg:.0f}] deg"
    )
    print("Reading present antenna positions at 50 Hz. Ctrl-C to stop.")
    print("Torque is left untouched; disable it so the antennas are floppy.\n")

    count = 0
    with ReachyMini(media_backend="no_media") as mini:
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
