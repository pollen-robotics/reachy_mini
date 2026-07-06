"""Repro: one hand-spun turn + one plain goto = violent antenna sweep.

THE MECHANISM (all confirmed against rustypot source, a live boundary
probe, and the 50 Hz trace of the 2026-07-06 incident):

  - The STS3215 firmware tracks MULTI-TURN present positions: spinning a
    floppy (torque off) antenna past +-180 deg makes the reading carry
    whole turns (e.g. physical +82 deg read as -278 deg).
  - goal_position is a single-turn register (0..4095 = -180..+180 deg).
    rustypot's AnglePosition::to_raw is linear, no wrap, no clamp
    (src/servo/dynamixel/mx.rs): commands beyond +-180 deg produce
    out-of-range registers that the firmware SILENTLY REJECTS (probe:
    goto +170 -> +190 deg follows smoothly to +180 then holds, 35 deg/s).
  - On torque enable the firmware sets goal := present, so enabling never
    jumps by itself.
  - A goto whose start came from a multi-turn reading interpolates its
    commands from that offset value: every command is rejected until the
    interpolation crosses +-180 deg, and the FIRST ACCEPTED command sits
    a large step away from the physical position. The servo chases it at
    full speed (incident trace: +82 -> -177 deg, ~1800 deg/s peak, at 42%
    of a 5 s min-jerk goto, then perfect tracking).

A goto alone cannot build the multi-turn gap (torque-on antennas stop at
the boundary), hence the one manual step below. GotoMove now wraps its
start into one turn (commit "goto: wrap multi-turn antenna start"), which
kills the bug: run this against a daemon WITHOUT that commit to see the
whip, and with it to see a gentle glide - same script.

Run (robot on the LAN; keep hands clear after the spin):
    python examples/secret_handshake_lab/repro_goto_antenna_jump.py
    -> antennas go torque-off for 12 s: spin the LEFT antenna one full
       turn (either direction), let it rest, hands off
    -> a single goto to (+-10 deg) runs; peak speed is printed
"""

import math
import threading
import time

import numpy as np

from reachy_mini import ReachyMini

RAD2DEG = 180.0 / math.pi
SPIN_WINDOW_S = 12.0


def record(mini: ReachyMini, out: list, stop: threading.Event) -> None:
    while not stop.is_set():
        t = time.monotonic()
        ant = mini.get_present_antenna_joint_positions()
        out.append((t, ant[0] * RAD2DEG, ant[1] * RAD2DEG))
        time.sleep(0.02)


def main() -> None:
    with ReachyMini(media_backend="no_media") as mini:
        mini.disable_motors()
        print(
            f"Torque OFF: spin the LEFT antenna ONE FULL TURN now "
            f"({SPIN_WINDOW_S:.0f} s)..."
        )
        time.sleep(SPIN_WINDOW_S)

        ant = mini.get_present_antenna_joint_positions()
        l_deg = ant[0] * RAD2DEG
        print(f"left antenna reads {l_deg:+.1f} deg "
              f"({'multi-turn offset present' if abs(l_deg) > 180 else 'NO turn offset: spin it further and rerun'})")

        mini.enable_motors()
        time.sleep(0.5)

        trace: list = []
        stop = threading.Event()
        rec = threading.Thread(target=record, args=(mini, trace, stop))
        rec.start()
        mini.goto_target(antennas=np.radians([-10.0, 10.0]), duration=5.0)
        stop.set()
        rec.join()

        worst = 0.0
        for (t0, l0, _), (t1, l1, _) in zip(trace, trace[1:]):
            worst = max(worst, abs(l1 - l0) / (t1 - t0))
        print(f"goto to -10 deg: left antenna peak speed {worst:7.0f} deg/s")
        print("(> 500 deg/s = the bug; < 100 deg/s = wrapped start, fixed)")

        mini.disable_motors()


if __name__ == "__main__":
    main()
