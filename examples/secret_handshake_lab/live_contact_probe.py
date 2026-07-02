"""Live antenna-collision probe. Prints contact state in real time.

This is the "first step before touching the daemon": run it, disable torque,
wiggle the antennas around, and watch for false positives / missed collisions.
It uses the exact same pure CollisionDetector that will later go in the daemon,
so whatever you tune here transfers directly.

It does NOT enable torque and does NOT move the robot. It only reads present
antenna positions and prints. Antennas stay floppy so you can play with them.

Run on the robot (or against a robot on the LAN):
    python examples/secret_handshake_lab/live_contact_probe.py
"""

from __future__ import annotations

import time

from collision import CollisionConfig, CollisionDetector

from reachy_mini import ReachyMini


def main() -> None:
    cfg = CollisionConfig()
    det = CollisionDetector(cfg)
    print(f"CollisionConfig(t_on={cfg.t_on}, t_off={cfg.t_off})  diff = ant0 - ant1")
    print("Reading present antenna positions at 50 Hz. Ctrl-C to stop.")
    print("Torque is left untouched; disable it so the antennas are floppy.\n")

    onset_count = 0
    with ReachyMini(media_backend="no_media") as mini:
        while True:
            ant0, ant1 = mini.get_present_antenna_joint_positions()
            new = det.update(ant0, ant1)
            if new:
                onset_count += 1
            diff = ant0 - ant1
            state = "CONTACT" if det.in_contact else "  ...  "
            flag = f"  <-- collision #{onset_count}" if new else ""
            print(
                f"ant0={ant0:+.3f} ant1={ant1:+.3f} diff={diff:+.3f}  {state}{flag}",
                end="\r" if not new else "\n",
                flush=True,
            )
            time.sleep(0.02)


if __name__ == "__main__":
    main()
