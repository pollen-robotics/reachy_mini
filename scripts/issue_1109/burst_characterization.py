#!/usr/bin/env python3
"""GPIO23 motor-EMI burst characterization for issue #1109.

Runs ON the robot (Raspberry Pi inside Reachy Mini). Listens for raw
GPIO23 edges with high-resolution monotonic timestamps while optionally
driving the motors at a configurable cadence. Produces a JSONL edge log
plus a stdout summary so we can answer "how long are the EMI bursts,
actually?" with measured numbers instead of inference.

Pre-conditions
--------------
* `sudo systemctl stop gpio-shutdown-daemon` (we need exclusive GPIO23
  access; the daemon's own monitor would otherwise also fire callbacks
  and may actually shut the robot down).
* `reachy-mini-daemon` running (so `set_target` works). Use `--no-motor`
  if you want a passive EMI-floor run without motor activity.
* Run as the daemon user (typically `pollen`) — gpiozero needs hardware
  access. `sudo` is fine too.

What it measures
----------------
* Every `when_pressed` (rising edge) and `when_released` (falling edge)
  with `time.monotonic_ns()` relative to start.
* Burst clustering: a run of >=2 edges with inter-edge gap < 100 ms is
  treated as one "burst" and reported as a duration.
* Inter-edge gap distribution (count <1ms, count <10ms).

The shutdown-monitor's edge-driven Timer cancels on ANY `when_pressed`
edge during the hold window, so the **burst duration** is what matters
for HOLD_TIME sizing: HOLD_TIME just needs to outlast a single burst
without an intervening cancel-edge. Inter-edge gap distribution is what
matters for "could a burst be wide enough that no cancel arrives in
HOLD_TIME?"

Output
------
* `logs/issue_1109/burst-<UTC>.jsonl` — one JSON object per edge plus a
  session_meta header.
* Stdout summary table (also written to the meta line).

Usage
-----
    sudo systemctl stop gpio-shutdown-daemon
    python3 scripts/issue_1109/burst_characterization.py \\
        --duration 300 --freq 10

    # baseline EMI floor (no motor stress):
    python3 scripts/issue_1109/burst_characterization.py \\
        --duration 60 --no-motor
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any

from gpiozero import Button  # type: ignore[import-untyped]


def _motor_stress_loop(stop_event: threading.Event, freq_hz: float) -> None:
    """Drive the head + antennas + body_yaw at `freq_hz` until stop_event."""
    import math

    import numpy as np
    from reachy_mini import ReachyMini  # type: ignore[import-untyped]

    period = 1.0 / freq_hz
    n = 0
    try:
        with ReachyMini() as robot:
            # Robot may be in sleep state — `set_target` is silently
            # ignored until wake_up() transitions the daemon out of
            # sleep. `enable_motors()` alone enables torque but the
            # daemon ignores streamed targets while asleep.
            # wake_up() plays a "toudoum" sound and wiggles the head;
            # acceptable for a test run.
            print("[motor] wake_up...", file=sys.stderr)
            robot.wake_up()
            print("[motor] awake; starting stress loop", file=sys.stderr)
            try:
                t0 = time.monotonic()
                while not stop_event.is_set():
                    t = time.monotonic() - t0
                    # Small sinusoidal head yaw (~5 deg) + antenna sweep + body yaw
                    # — enough motor activity to reproduce #1109 without large
                    # excursions.
                    yaw_rad = math.radians(5.0) * math.sin(2 * math.pi * 0.5 * t)
                    head = np.eye(4, dtype=np.float64)
                    head[0, 0] = math.cos(yaw_rad)
                    head[0, 1] = -math.sin(yaw_rad)
                    head[1, 0] = math.sin(yaw_rad)
                    head[1, 1] = math.cos(yaw_rad)
                    antenna_rad = math.radians(15.0) * math.sin(2 * math.pi * 0.7 * t)
                    try:
                        robot.set_target(
                            head=head,
                            antennas=[antenna_rad, -antenna_rad],
                            body_yaw=math.radians(3.0) * math.sin(2 * math.pi * 0.3 * t),
                        )
                    except Exception as exc:  # noqa: BLE001
                        print(f"[motor] set_target error at n={n}: {exc}", file=sys.stderr)
                    n += 1
                    # Use Event.wait so we exit promptly on stop.
                    if stop_event.wait(timeout=period):
                        break
            finally:
                try:
                    robot.disable_motors()
                    print("[motor] motors disabled", file=sys.stderr)
                except Exception as exc:  # noqa: BLE001
                    print(f"[motor] disable_motors error: {exc}", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        print(f"[motor] fatal: {exc}", file=sys.stderr)
    finally:
        print(f"[motor] stopped after {n} commands", file=sys.stderr)


def _summarize(
    edges: list[dict[str, Any]],
    duration_s: float,
    burst_gap_threshold_us: float = 100_000.0,
) -> dict[str, Any]:
    """Compute burst-cluster + inter-edge-gap statistics from the edge log."""
    summary: dict[str, Any] = {
        "n_edges": len(edges),
        "duration_s": duration_s,
        "burst_gap_threshold_us": burst_gap_threshold_us,
    }
    if len(edges) < 2:
        return summary

    pressed = [e for e in edges if e["edge"] == "pressed"]
    released = [e for e in edges if e["edge"] == "released"]
    summary["n_pressed"] = len(pressed)
    summary["n_released"] = len(released)

    gaps_us = [
        (edges[i + 1]["monotonic_ns"] - edges[i]["monotonic_ns"]) / 1000.0
        for i in range(len(edges) - 1)
    ]
    summary["inter_edge_gap_us"] = {
        "min": min(gaps_us),
        "p50": statistics.median(gaps_us),
        "max": max(gaps_us),
        "count_lt_1ms": sum(1 for g in gaps_us if g < 1_000),
        "count_lt_10ms": sum(1 for g in gaps_us if g < 10_000),
        "count_lt_100ms": sum(1 for g in gaps_us if g < 100_000),
    }

    # Cluster edges into bursts. A burst = >=2 edges separated by gaps <
    # burst_gap_threshold_us. Burst duration = first→last edge in the run.
    bursts: list[list[dict[str, Any]]] = []
    current = [edges[0]]
    for prev, nxt in zip(edges, edges[1:], strict=False):
        gap_us = (nxt["monotonic_ns"] - prev["monotonic_ns"]) / 1000.0
        if gap_us < burst_gap_threshold_us:
            current.append(nxt)
        else:
            if len(current) > 1:
                bursts.append(current)
            current = [nxt]
    if len(current) > 1:
        bursts.append(current)

    summary["n_bursts"] = len(bursts)
    if bursts:
        durs_us = [
            (b[-1]["monotonic_ns"] - b[0]["monotonic_ns"]) / 1000.0 for b in bursts
        ]
        counts = [len(b) for b in bursts]
        burst_stats: dict[str, Any] = {
            "duration_us": {
                "min": min(durs_us),
                "p50": statistics.median(durs_us),
                "max": max(durs_us),
            },
            "edge_count": {
                "min": min(counts),
                "p50": statistics.median(counts),
                "max": max(counts),
            },
        }
        if len(durs_us) >= 20:
            burst_stats["duration_us"]["p95"] = statistics.quantiles(durs_us, n=20)[18]
            burst_stats["duration_us"]["p99"] = statistics.quantiles(durs_us, n=100)[98]
        summary["bursts"] = burst_stats

    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--duration",
        type=float,
        default=300.0,
        help="seconds to characterize (default: 300)",
    )
    ap.add_argument(
        "--freq",
        type=float,
        default=10.0,
        help="set_target frequency in Hz (default: 10 — matches #1109 reproducer)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("logs/issue_1109"),
        help="output directory (default: logs/issue_1109/)",
    )
    ap.add_argument(
        "--gpio",
        type=int,
        default=23,
        help="GPIO pin to monitor (default: 23)",
    )
    ap.add_argument(
        "--no-motor",
        action="store_true",
        help="skip motor stress (passive EMI floor check)",
    )
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_path = args.out / f"burst-{ts}.jsonl"
    print(f"[main] logging to {out_path}", file=sys.stderr)

    edges: list[dict[str, Any]] = []
    edges_lock = threading.Lock()
    t0_ns = time.monotonic_ns()

    # pull_up=False matches production shutdown_monitor.py — defers to the
    # kernel device-tree pull configuration.
    button = Button(args.gpio, pull_up=False)

    def _record(edge: str) -> None:
        with edges_lock:
            edges.append(
                {
                    "edge": edge,
                    "monotonic_ns": time.monotonic_ns() - t0_ns,
                    "wall_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            )

    button.when_pressed = lambda: _record("pressed")
    button.when_released = lambda: _record("released")

    initial = "pressed (HIGH/latched-in)" if button.is_pressed else "released (LOW/latched-out)"
    print(f"[main] initial GPIO{args.gpio} state: {initial}", file=sys.stderr)
    print(
        f"[main] starting {args.duration:.0f}s run | freq={args.freq} Hz | motor={not args.no_motor}",
        file=sys.stderr,
    )

    stop_event = threading.Event()
    motor_thread: threading.Thread | None = None
    if not args.no_motor:
        motor_thread = threading.Thread(
            target=_motor_stress_loop, args=(stop_event, args.freq), daemon=True
        )
        motor_thread.start()

    try:
        stop_event.wait(timeout=args.duration)
    except KeyboardInterrupt:
        print("[main] interrupted - flushing", file=sys.stderr)
    finally:
        stop_event.set()
        if motor_thread is not None:
            motor_thread.join(timeout=2.0)

    with edges_lock:
        snapshot = list(edges)

    summary = _summarize(snapshot, args.duration)
    summary["pin_initial_state"] = initial
    summary["motor_stress"] = not args.no_motor
    summary["freq_hz"] = args.freq

    with out_path.open("w") as f:
        f.write(json.dumps({"event": "session_meta", **summary}) + "\n")
        for e in snapshot:
            f.write(json.dumps(e) + "\n")

    # Pretty-print summary to stderr.
    print("", file=sys.stderr)
    print("=== summary ===", file=sys.stderr)
    print(json.dumps(summary, indent=2, default=str), file=sys.stderr)
    print(f"\n[main] JSONL: {out_path}", file=sys.stderr)

    button.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
