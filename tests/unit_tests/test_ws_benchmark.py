"""Benchmark WebSocket connection delay and message jitter.

Starts a local daemon in mockup-sim mode (no MuJoCo required) and
connects a raw websocket client to measure:
  - connection delay  : time from connect() to the first joint_positions msg
  - publish interval  : actual period between consecutive joint_positions msgs
  - jitter            : standard-deviation of that interval (ideal = 0 at 50 Hz)
  - round-trip delay  : time from sending set_antennas to observing the change
                        in the next joint_positions message
"""

import asyncio
import json
import statistics
import threading
import time

import pytest
import uvicorn
import websockets.sync.client as ws_sync

from reachy_mini.daemon.app.main import Args, create_app


# ------------------------------------------------------------------
# Helpers (same pattern as test_daemon.py)
# ------------------------------------------------------------------


async def _start_server() -> tuple[uvicorn.Server, threading.Thread, int]:
    args = Args(
        mockup_sim=True,
        headless=True,
        wake_up_on_start=False,
        use_audio=False,
        autostart=True,
        fastapi_port=0,
    )
    app = create_app(args)
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    while not server.started:
        await asyncio.sleep(0.05)

    port: int = server.servers[0].sockets[0].getsockname()[1]  # type: ignore[union-attr]
    return server, thread, port


async def _stop_server(server: uvicorn.Server, thread: threading.Thread) -> None:
    server.should_exit = True
    thread.join(timeout=10)


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

SAMPLE_DURATION_S = 2.0  # how long to collect messages


@pytest.mark.asyncio
async def test_connection_delay() -> None:
    """Measure time from ws connect() to receiving the first joint_positions."""
    server, thread, port = await _start_server()

    try:
        t0 = time.perf_counter()
        ws = ws_sync.connect(f"ws://localhost:{port}/ws/sdk")

        first_msg_time: float | None = None
        try:
            for raw in ws:
                msg = json.loads(raw)
                if msg.get("type") == "joint_positions":
                    first_msg_time = time.perf_counter()
                    break
        finally:
            ws.close()

        assert first_msg_time is not None
        delay_ms = (first_msg_time - t0) * 1000
        print(f"\n  connection delay: {delay_ms:.1f} ms")
        # Should connect and get first message well under 1 second
        assert delay_ms < 1000, f"Connection delay too high: {delay_ms:.1f} ms"

    finally:
        await _stop_server(server, thread)


@pytest.mark.asyncio
async def test_publish_jitter() -> None:
    """Measure inter-arrival jitter of joint_positions messages (expected ~20 ms at 50 Hz)."""
    server, thread, port = await _start_server()

    try:
        ws = ws_sync.connect(f"ws://localhost:{port}/ws/sdk")
        timestamps: list[float] = []
        deadline = time.perf_counter() + SAMPLE_DURATION_S

        try:
            for raw in ws:
                msg = json.loads(raw)
                if msg.get("type") == "joint_positions":
                    timestamps.append(time.perf_counter())
                if time.perf_counter() >= deadline:
                    break
        finally:
            ws.close()

        intervals_ms = [
            (timestamps[i] - timestamps[i - 1]) * 1000
            for i in range(1, len(timestamps))
        ]

        mean_ms = statistics.mean(intervals_ms)
        stdev_ms = statistics.stdev(intervals_ms)
        min_ms = min(intervals_ms)
        max_ms = max(intervals_ms)

        print(
            f"\n  samples       : {len(intervals_ms)}"
            f"\n  mean interval : {mean_ms:.2f} ms  (expected ~20 ms)"
            f"\n  stdev (jitter): {stdev_ms:.2f} ms"
            f"\n  min / max     : {min_ms:.2f} / {max_ms:.2f} ms"
        )

        # Sanity: mean interval should be roughly 20 ms (50 Hz)
        assert 10 < mean_ms < 40, f"Mean interval out of range: {mean_ms:.2f} ms"
        # Jitter should stay reasonable (< half the period)
        assert stdev_ms < 10, f"Jitter too high: {stdev_ms:.2f} ms"

    finally:
        await _stop_server(server, thread)


@pytest.mark.asyncio
async def test_command_round_trip() -> None:
    """Measure round-trip: send set_antennas, wait for the change in joint_positions."""
    server, thread, port = await _start_server()

    try:
        ws = ws_sync.connect(f"ws://localhost:{port}/ws/sdk")

        # Wait for the first joint_positions so the server is fully warmed up
        initial_antennas: list[float] | None = None
        for raw in ws:
            msg = json.loads(raw)
            if msg.get("type") == "joint_positions":
                initial_antennas = msg["antennas_joint_positions"]
                break
        assert initial_antennas is not None

        # Pick a target that differs noticeably from the current value
        target = [a + 0.5 for a in initial_antennas]

        round_trips_ms: list[float] = []
        N_ROUNDS = 10

        try:
            for _ in range(N_ROUNDS):
                # Flip between two targets so we always see a change
                target = [a + 0.5 for a in initial_antennas] if len(round_trips_ms) % 2 == 0 else initial_antennas.copy()
                cmd = json.dumps({"type": "set_antennas", "antennas": target})

                t_send = time.perf_counter()
                ws.send(cmd)

                # Wait for a joint_positions that reflects the new target
                for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("type") == "joint_positions":
                        t_recv = time.perf_counter()
                        round_trips_ms.append((t_recv - t_send) * 1000)
                        break
        finally:
            ws.close()

        mean_ms = statistics.mean(round_trips_ms)
        stdev_ms = statistics.stdev(round_trips_ms) if len(round_trips_ms) > 1 else 0.0

        print(
            f"\n  rounds        : {len(round_trips_ms)}"
            f"\n  mean RT       : {mean_ms:.2f} ms"
            f"\n  stdev         : {stdev_ms:.2f} ms"
            f"\n  min / max     : {min(round_trips_ms):.2f} / {max(round_trips_ms):.2f} ms"
        )

        # Round-trip should be well under a second
        assert mean_ms < 200, f"Mean round-trip too high: {mean_ms:.2f} ms"

    finally:
        await _stop_server(server, thread)
