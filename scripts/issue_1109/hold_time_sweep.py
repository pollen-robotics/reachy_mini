#!/usr/bin/env python3
"""HOLD_TIME floor sweep for issue #1109 (PR #1110 follow-up).

Reproduces the production edge-driven Timer logic from
`reachy_mini.daemon.app.services.gpio_shutdown.shutdown_monitor` in-process
with HOLD_TIME parameterized, replaces `shutdown_now` with a counter,
and runs the #1109 reproducer (10 Hz `set_target` motor stress) for a
configurable duration per HOLD_TIME value. Reports per-HOLD_TIME spurious
firings so we can establish the minimum value that survives motor EMI
with zero false positives — the floor of the safe range. Pierre's
~2 s latch→power-rail-cut bound (review on PR #1110) is the ceiling.

Pre-conditions
--------------
* `sudo systemctl stop gpio-shutdown-daemon` — exclusive GPIO23 access.
* `reachy-mini-daemon` running.
* Do NOT touch the latch during a sweep. We're measuring spurious
  firings from EMI alone; a deliberate latch pull would be a true
  positive and contaminate the floor estimate.

What it tests
-------------
For each HOLD_TIME in the sweep:
  1. Reset Timer state.
  2. Start motor stress at `--freq` Hz.
  3. Listen for GPIO23 edges; falling edge schedules
     `Timer(HOLD_TIME, _fake_shutdown)`, rising edge cancels it.
  4. Run for `--duration-per-step` seconds.
  5. Tally: scheduled timers, cancelled timers, fires (= spurious
     shutdowns), edge counts.

A PASS for a HOLD_TIME value is **0 fires**. The floor is the smallest
HOLD_TIME that passes. The PR's claim that EMI bursts are
"sub-millisecond" predicts that even HOLD_TIME = 50 ms should pass;
this harness verifies or falsifies that prediction with measurements.

Output
------
* `logs/issue_1109/sweep-<UTC>.jsonl` — per-step result row + a final
  summary row. One JSON object per line.
* Stdout table.

Usage
-----
    sudo systemctl stop gpio-shutdown-daemon
    python3 scripts/issue_1109/hold_time_sweep.py \\
        --hold-times 50 100 200 500 1000 2000 \\
        --duration-per-step 120 --freq 10

A 6-value sweep at 120 s/step = ~12 min of robot time. Start with a
narrower sweep if you want a quick smoke (e.g. `--hold-times 200 500`).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from gpiozero import Button  # type: ignore[import-untyped]


class TimerDebounce:
    """In-process clone of production shutdown_monitor.py's Timer logic.

    Keeps the same semantics: `when_released` schedules
    `Timer(hold_time, shutdown_cb)`; `when_pressed` cancels any pending
    Timer. `shutdown_cb` is replaced with a counter for sweep purposes
    so the robot does not actually shut down.
    """

    def __init__(self, hold_time: float, shutdown_cb: Callable[[], None]) -> None:
        self.hold_time = hold_time
        self._shutdown_cb = shutdown_cb
        self._lock = threading.Lock()
        self._pending: threading.Timer | None = None
        self.n_scheduled = 0
        self.n_cancelled = 0
        self.n_fired = 0

    def on_released(self) -> None:
        with self._lock:
            if self._pending is not None:
                self._pending.cancel()
                self.n_cancelled += 1
            timer = threading.Timer(self.hold_time, self._fire)
            timer.daemon = True
            self._pending = timer
            self.n_scheduled += 1
            timer.start()

    def on_pressed(self) -> None:
        with self._lock:
            if self._pending is not None:
                self._pending.cancel()
                self._pending = None
                self.n_cancelled += 1

    def _fire(self) -> None:
        with self._lock:
            self.n_fired += 1
            self._pending = None
        self._shutdown_cb()

    def shutdown(self) -> None:
        with self._lock:
            if self._pending is not None:
                self._pending.cancel()
                self._pending = None


def _motor_stress_loop(robot: Any, stop_event: threading.Event, freq_hz: float) -> None:
    """Drive head + antennas + body_yaw at `freq_hz` (reproduces #1109 scenario c).

    Caller owns the ReachyMini context. We run continuously for the whole
    sweep; per-step wake_up cycling crashes the GStreamer pipeline used
    by the sound playback in wake_up().
    """
    period = 1.0 / freq_hz
    n = 0
    t0 = time.monotonic()
    while not stop_event.is_set():
        t = time.monotonic() - t0
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
        if stop_event.wait(timeout=period):
            break
    print(f"[motor] stopped after {n} commands", file=sys.stderr)


def _run_step(
    hold_time: float,
    duration_s: float,
    freq_hz: float,
    gpio_pin: int,
) -> dict[str, Any]:
    """Run one HOLD_TIME value; return its result dict.

    Caller must already be running the motor stress thread continuously.
    """
    n_pressed = 0
    n_released = 0
    n_pressed_lock = threading.Lock()

    debounce = TimerDebounce(hold_time, shutdown_cb=lambda: None)
    # pull_up=False matches production.
    button = Button(gpio_pin, pull_up=False)

    def _pressed() -> None:
        nonlocal n_pressed
        with n_pressed_lock:
            n_pressed += 1
        debounce.on_pressed()

    def _released() -> None:
        nonlocal n_released
        with n_pressed_lock:
            n_released += 1
        debounce.on_released()

    button.when_pressed = _pressed
    button.when_released = _released

    initial_pressed = button.is_pressed
    print(
        f"\n[step] HOLD_TIME={hold_time:.3f}s | duration={duration_s:.0f}s | "
        f"initial_state={'pressed' if initial_pressed else 'released'}",
        file=sys.stderr,
    )

    t_start = time.monotonic()
    try:
        # Block for duration_s while motors stress in the background.
        time.sleep(duration_s)
    except KeyboardInterrupt:
        print("[step] interrupted", file=sys.stderr)
    finally:
        # Give any pending Timer hold_time + 100ms to fire before tearing
        # down; otherwise we'd under-count fires in the last hold_time
        # window.
        time.sleep(hold_time + 0.1)
        debounce.shutdown()
        button.close()

    elapsed = time.monotonic() - t_start
    result = {
        "hold_time_s": hold_time,
        "duration_s": duration_s,
        "elapsed_s": elapsed,
        "freq_hz": freq_hz,
        "initial_pressed": initial_pressed,
        "n_pressed_edges": n_pressed,
        "n_released_edges": n_released,
        "n_timers_scheduled": debounce.n_scheduled,
        "n_timers_cancelled": debounce.n_cancelled,
        "n_timers_fired": debounce.n_fired,
        "pass": debounce.n_fired == 0,
    }
    print(
        f"[step] pressed={n_pressed} released={n_released} "
        f"scheduled={debounce.n_scheduled} cancelled={debounce.n_cancelled} "
        f"fired={debounce.n_fired} -> {'PASS' if result['pass'] else 'FAIL'}",
        file=sys.stderr,
    )
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--hold-times",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
        help="HOLD_TIME values to sweep, in seconds (default: 0.05 0.1 0.2 0.5 1.0 2.0)",
    )
    ap.add_argument(
        "--duration-per-step",
        type=float,
        default=120.0,
        help="seconds per HOLD_TIME value (default: 120)",
    )
    ap.add_argument(
        "--freq",
        type=float,
        default=10.0,
        help="set_target frequency in Hz (default: 10 — matches #1109 reproducer)",
    )
    ap.add_argument(
        "--gpio",
        type=int,
        default=23,
        help="GPIO pin (default: 23)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("logs/issue_1109"),
        help="output directory (default: logs/issue_1109/)",
    )
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_path = args.out / f"sweep-{ts}.jsonl"
    print(f"[main] logging to {out_path}", file=sys.stderr)

    print(
        f"[main] sweep: {len(args.hold_times)} values x {args.duration_per_step:.0f}s "
        f"= ~{len(args.hold_times) * args.duration_per_step / 60:.1f} min total",
        file=sys.stderr,
    )

    results: list[dict[str, Any]] = []
    meta = {
        "event": "session_meta",
        "hold_times_s": args.hold_times,
        "duration_per_step_s": args.duration_per_step,
        "freq_hz": args.freq,
        "gpio": args.gpio,
        "started_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Single ReachyMini context + single wake_up + continuous motor stress
    # thread for the whole sweep. Per-step wake_up was crashing GStreamer.
    from reachy_mini import ReachyMini  # type: ignore[import-untyped]

    motor_stop = threading.Event()
    with out_path.open("w") as f:
        f.write(json.dumps(meta) + "\n")
        with ReachyMini() as robot:
            # Gentle wake — no chime, no 20deg wiggle. wake_up() is violent
            # AND its play_sound() crashes the GStreamer pipeline on
            # repeat calls; use the enable+goto path which is proven
            # reliable and quiet.
            print("[main] gentle wake (enable_motors + slow goto)...", file=sys.stderr)
            robot.enable_motors()
            time.sleep(0.2)
            robot.goto_target(
                np.eye(4), antennas=[-0.1745, 0.1745], duration=2.0
            )
            time.sleep(2.3)
            print("[main] awake; starting continuous motor stress", file=sys.stderr)
            motor_thread = threading.Thread(
                target=_motor_stress_loop,
                args=(robot, motor_stop, args.freq),
                daemon=True,
            )
            motor_thread.start()
            try:
                for hold_time in args.hold_times:
                    try:
                        result = _run_step(
                            hold_time=hold_time,
                            duration_s=args.duration_per_step,
                            freq_hz=args.freq,
                            gpio_pin=args.gpio,
                        )
                    except KeyboardInterrupt:
                        print("[main] sweep interrupted", file=sys.stderr)
                        break
                    results.append(result)
                    f.write(json.dumps({"event": "step_result", **result}) + "\n")
                    f.flush()
            finally:
                motor_stop.set()
                motor_thread.join(timeout=2.0)
                try:
                    robot.disable_motors()
                except Exception as exc:  # noqa: BLE001
                    print(f"[main] disable_motors error: {exc}", file=sys.stderr)

        floor: float | None = None
        for r in sorted(results, key=lambda x: x["hold_time_s"]):
            if r["pass"]:
                floor = r["hold_time_s"]
                break
        summary = {
            "event": "sweep_summary",
            "n_steps": len(results),
            "floor_s": floor,
            "all_results": results,
            "ended_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        f.write(json.dumps(summary) + "\n")

    # Pretty-print summary table.
    print("\n=== sweep summary ===", file=sys.stderr)
    print(
        f"{'HOLD_TIME (s)':>14} {'pressed':>8} {'released':>9} "
        f"{'scheduled':>10} {'cancelled':>10} {'fired':>6} {'verdict':>8}",
        file=sys.stderr,
    )
    for r in results:
        print(
            f"{r['hold_time_s']:>14.3f} {r['n_pressed_edges']:>8} {r['n_released_edges']:>9} "
            f"{r['n_timers_scheduled']:>10} {r['n_timers_cancelled']:>10} "
            f"{r['n_timers_fired']:>6} {'PASS' if r['pass'] else 'FAIL':>8}",
            file=sys.stderr,
        )
    if floor is not None:
        print(f"\n[main] floor: HOLD_TIME >= {floor:.3f}s passed with 0 fires", file=sys.stderr)
    else:
        print("\n[main] NO value in sweep passed (all had spurious fires)", file=sys.stderr)
    print(f"[main] JSONL: {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
