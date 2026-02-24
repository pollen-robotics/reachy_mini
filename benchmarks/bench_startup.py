#!/usr/bin/env python
"""Benchmark ReachyMini SDK startup time.

Spawns a mockup daemon, then runs N fresh subprocesses that each measure:
  - import_time:  `from reachy_mini import ReachyMini`
  - connect_time: `ReachyMini(media_backend='no_media', connection_mode='localhost_only')`
  - total_time:   sum of the above

Each subprocess is a clean Python process (no cached imports).
Reports mean ± stddev over N runs.

Usage:
    python benchmarks/bench_startup.py [--runs 10] [--label baseline]
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from statistics import mean, stdev

DAEMON_PORT = 8000
BENCH_SUBPROCESS = textwrap.dedent("""\
    import json, sys, time

    t0 = time.perf_counter()
    from reachy_mini import ReachyMini
    t_import = time.perf_counter()

    rm = ReachyMini(media_backend="no_media", connection_mode="localhost_only")
    t_connect = time.perf_counter()

    # Clean up
    try:
        rm.client.disconnect()
    except Exception:
        pass

    result = {
        "import_s": t_import - t0,
        "connect_s": t_connect - t_import,
        "total_s": t_connect - t0,
    }
    print(json.dumps(result))
""")


def is_daemon_running(port: int = DAEMON_PORT, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def start_daemon() -> subprocess.Popen:
    """Start a mockup daemon and wait for it to be ready."""
    proc = subprocess.Popen(
        [
            "reachy-mini-daemon",
            "--mockup-sim",
            "--no-wake-up-on-start",
            "--no-goto-sleep-on-stop",
            "--deactivate-audio",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    # Wait for daemon to be ready
    for _ in range(40):
        if is_daemon_running():
            return proc
        time.sleep(0.25)
    proc.kill()
    raise RuntimeError("Daemon did not start within 10 seconds")


def run_single_bench() -> dict:
    """Run one benchmark iteration in a fresh subprocess."""
    result = subprocess.run(
        [sys.executable, "-c", BENCH_SUBPROCESS],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr.strip()}", file=sys.stderr)
        return {}
    try:
        return json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        print(f"  BAD OUTPUT: {result.stdout.strip()}", file=sys.stderr)
        return {}


def fmt(values: list[float]) -> str:
    """Format mean ± stddev."""
    if len(values) < 2:
        return f"{values[0]:.3f}s" if values else "N/A"
    m = mean(values)
    s = stdev(values)
    return f"{m:.3f}s ± {s:.3f}s"


def main():
    parser = argparse.ArgumentParser(description="Benchmark ReachyMini startup")
    parser.add_argument("--runs", type=int, default=10, help="Number of iterations")
    parser.add_argument("--label", type=str, default="", help="Label for this run")
    parser.add_argument("--json-out", type=str, default="", help="Save results to JSON file")
    args = parser.parse_args()

    # Ensure daemon is running
    daemon_proc = None
    if not is_daemon_running():
        print("Starting mockup daemon...")
        daemon_proc = start_daemon()
        print(f"Daemon ready (PID {daemon_proc.pid})")
    else:
        print("Daemon already running")

    # Warmup run (not counted — lets Zenoh settle)
    print("Warmup run...")
    run_single_bench()

    # Benchmark runs
    import_times = []
    connect_times = []
    total_times = []

    print(f"Running {args.runs} iterations...")
    for i in range(args.runs):
        result = run_single_bench()
        if not result:
            print(f"  Run {i+1}: FAILED")
            continue
        import_times.append(result["import_s"])
        connect_times.append(result["connect_s"])
        total_times.append(result["total_s"])
        print(f"  Run {i+1}: import={result['import_s']:.3f}s  connect={result['connect_s']:.3f}s  total={result['total_s']:.3f}s")

    # Report
    print()
    label = f" ({args.label})" if args.label else ""
    print(f"=== Results{label} — {len(total_times)}/{args.runs} successful runs ===")
    print(f"  Import:  {fmt(import_times)}")
    print(f"  Connect: {fmt(connect_times)}")
    print(f"  Total:   {fmt(total_times)}")

    # Save JSON
    if args.json_out:
        data = {
            "label": args.label,
            "runs": args.runs,
            "successful": len(total_times),
            "import": {"mean": mean(import_times), "stdev": stdev(import_times) if len(import_times) > 1 else 0, "values": import_times},
            "connect": {"mean": mean(connect_times), "stdev": stdev(connect_times) if len(connect_times) > 1 else 0, "values": connect_times},
            "total": {"mean": mean(total_times), "stdev": stdev(total_times) if len(total_times) > 1 else 0, "values": total_times},
        }
        Path(args.json_out).write_text(json.dumps(data, indent=2))
        print(f"\nResults saved to {args.json_out}")

    # Stop daemon if we started it
    if daemon_proc:
        os.killpg(os.getpgid(daemon_proc.pid), signal.SIGTERM)
        daemon_proc.wait(timeout=5)
        print("Daemon stopped")


if __name__ == "__main__":
    main()
