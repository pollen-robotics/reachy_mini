"""Benchmark SecretHandshake.update(): the cost added to the daemon loop.

The daemon control loop (`RobotBackend._update()`) runs at 50 Hz, i.e. a
20 ms budget per tick. This measures the ONE call the secret handshake adds,
with inputs shaped like the daemon's (numpy 4x4 head pose, float antenna
positions), across every machine state, including the paths that run 99% of
the time (torque ON during normal use, head not in sleep pose).

Method: for each scenario, a fresh machine is warmed into the target state,
then update() is called in a tight loop with time advancing 20 ms per call.
Each scenario is repeated; the MEDIAN per-call time is reported, plus the
share of the 20 ms tick budget.

Run:
    python examples/secret_handshake_lab/bench.py
"""

from __future__ import annotations

import math
import time

import numpy as np

from handshake import HandshakeConfig, SecretHandshake
from pose_gate import SLEEP_HEAD_POSE

DT = 0.02
N_CALLS = 50_000
N_REPEATS = 5

SLEEP_POSE = np.array(SLEEP_HEAD_POSE)
WAKE_POSE = np.eye(4)  # head up: pose gate fails


def d2r(deg: float) -> float:
    return math.radians(deg)


REST = (d2r(-12.0), d2r(10.0))
CONTACT = (d2r(60.0), d2r(-65.0))
PRESS = (d2r(65.0), d2r(-30.0))

# One knock every ~0.5 s: 5 ticks through the band, 20 ticks rest.
TAP_TRAFFIC = [CONTACT] * 2 + [PRESS] * 3 + [CONTACT] * 2 + [REST] * 18


def bench(name: str, pose, torque_off: bool, antennas, warmup) -> float:
    per_call_ns = []
    for _ in range(N_REPEATS):
        hs = SecretHandshake(HandshakeConfig())
        t = 0.0
        for a0, a1 in warmup:
            hs.update(t, a0, a1, pose, torque_off)
            t += DT
        n = len(antennas)
        start = time.perf_counter()
        for i in range(N_CALLS):
            a0, a1 = antennas[i % n]
            hs.update(t, a0, a1, pose, torque_off)
            t += DT
        elapsed = time.perf_counter() - start
        per_call_ns.append(elapsed / N_CALLS * 1e9)
    ns = sorted(per_call_ns)[N_REPEATS // 2]
    budget = ns / (0.02 * 1e9)
    print(f"  {name:44s} {ns:7.0f} ns/call   {budget:.6%} of the 20 ms tick")
    return ns


def main() -> None:
    print(
        f"SecretHandshake.update() cost per 50 Hz control tick\n"
        f"(median of {N_REPEATS} runs of {N_CALLS} calls, numpy 4x4 head pose)\n"
    )
    still = [REST]
    settle = [REST] * 30  # 0.6 s: enough to pass the 0.5 s arming settle

    results = [
        bench("torque ON (normal use, earliest exit)", SLEEP_POSE, False, still, []),
        bench("idle, head up (pose gate fails)", WAKE_POSE, True, still, []),
        bench("sleep pose: settles idle->armed, then rests", SLEEP_POSE, True, still, []),
        bench("armed, antennas at rest", SLEEP_POSE, True, still, settle),
        bench("armed, continuous tap traffic", SLEEP_POSE, True, TAP_TRAFFIC, settle),
        bench("primed, holding in the band", SLEEP_POSE, True, [CONTACT], settle + TAP_TRAFFIC * 3),
    ]
    worst = max(results)
    print(
        f"\nworst case: {worst:.0f} ns/call = {worst / (0.02 * 1e9):.6%} of the tick "
        f"budget\n({20_000_000 / worst:,.0f} times smaller than the 20 ms budget)"
    )


if __name__ == "__main__":
    main()
