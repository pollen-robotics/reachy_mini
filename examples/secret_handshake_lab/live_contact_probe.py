"""Live low-level collision probe. Prints the coupled-motion signals raw.

Lower-level than live_handshake_probe.py: no state machine, no beeps, just
the exact per-tick numbers the detector sees. Use it to stare at the signal
while playing with the antennas and to tune CollisionConfig:

    ant0/ant1  present antenna positions (rad); absolute values are
               meaningless (floppy friction-fit parts), only motion matters
    m          coupled speed = min(|v0|, |v1|): ~0 at rest and for
               single-antenna motion, 0.3-1.2 while rubbing, spikes 4-9
               on a knock

It does NOT enable torque and does NOT move the robot. Antennas stay floppy
so you can play with them.

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
    print(
        f"CollisionConfig(knock_on={cfg.knock_on}, couple_on={cfg.couple_on}, "
        f"refractory={cfg.knock_refractory_s}s)"
    )
    print("Reading present antenna positions at 50 Hz. Ctrl-C to stop.")
    print("Torque is left untouched; disable it so the antennas are floppy.\n")

    knock_count = 0
    with ReachyMini(media_backend="no_media") as mini:
        while True:
            t = time.monotonic()
            ant0, ant1 = mini.get_present_antenna_joint_positions()
            knock = det.update(t, ant0, ant1)
            if knock:
                knock_count += 1
            state = "COUPLED" if det.coupled else "  ...  "
            flag = f"  <-- collision #{knock_count}" if knock else ""
            print(
                f"ant0={ant0:+.3f} ant1={ant1:+.3f} m={det.coupled_speed:5.2f}  {state}{flag}",
                end="\r" if not knock else "\n",
                flush=True,
            )
            time.sleep(0.02)


if __name__ == "__main__":
    main()
