"""Minimal repro: a goto whose antenna path crosses +-180 deg jumps violently.

No torque tricks, no multi-turn setup: TWO plain gotos.

    1. goto left antenna to +170 deg   (in range: smooth, as always)
    2. goto left antenna to +190 deg   (a 20 deg nudge... in theory)

Root cause (rustypot src/servo/dynamixel/mx.rs, used by the STS3215
goal_position register): `to_raw(value) = (4096*(pi+value)/(2*pi)) as i16`
is a pure linear conversion, no wrap, no clamp. The goal register is one
turn (0..4095); any command beyond +-180 deg leaves that range and the
firmware effectively masks it, so while the interpolated command crosses
+-180 the register teleports from ~4095 to ~0: one full turn. The servo
chases it at full speed.

This is exactly what the WiFi-provisioning wake goto triggered on
2026-07-06: the STS3215 firmware tracks multi-turn PRESENT positions
(reads carried a stale -278 deg for a physical +82 after torque-off
antenna play), the goto interpolated from that multi-turn start toward
-10 deg, and the command crossed -180 mid-move (violent sweep at 2.1 s of
a 5 s min-jerk goto, ~1800 deg/s, matching the recorded trace).

Run (robot on the LAN, torque will be enabled, HOLD NOTHING):
    python examples/secret_handshake_lab/repro_goto_antenna_jump.py
"""

import math
import threading
import time

import numpy as np

from reachy_mini import ReachyMini

RAD2DEG = 180.0 / math.pi


def record(mini: ReachyMini, out: list, stop: threading.Event) -> None:
    while not stop.is_set():
        t = time.monotonic()
        ant = mini.get_present_antenna_joint_positions()
        out.append((t, ant[0] * RAD2DEG, ant[1] * RAD2DEG))
        time.sleep(0.02)


def report(trace: list, label: str) -> None:
    worst = 0.0
    for (t0, l0, _), (t1, l1, _) in zip(trace, trace[1:]):
        speed = abs(l1 - l0) / (t1 - t0)
        worst = max(worst, speed)
    print(f"{label}: left antenna peak speed {worst:7.0f} deg/s")


def main() -> None:
    with ReachyMini(media_backend="no_media") as mini:
        mini.enable_motors()
        time.sleep(0.5)

        # step 0: park the left antenna near +170 deg, gently (in range)
        mini.goto_target(antennas=np.radians([170.0, -10.0]), duration=3.0)
        time.sleep(0.3)

        # step 1: the "20 deg nudge" whose command path crosses +180 deg
        trace: list = []
        stop = threading.Event()
        rec = threading.Thread(target=record, args=(mini, trace, stop))
        rec.start()
        mini.goto_target(antennas=np.radians([190.0, -10.0]), duration=2.0)
        stop.set()
        rec.join()
        report(trace, "goto +170 -> +190 deg (crosses +180)")

        # cleanup: back to a sane position, torque off
        mini.goto_target(antennas=np.radians([10.0, -10.0]), duration=3.0)
        mini.disable_motors()


if __name__ == "__main__":
    main()
