"""Tests for WSServer._broadcast scheduling behavior.

The backend's control loop calls _broadcast() 2-3 times per tick through the
publishers. Each call used to unconditionally schedule a callback on the
uvicorn event loop via call_soon_threadsafe once self._loop had been captured
(i.e. after the first client ever connected). With zero connected clients that
meant 100-150 pointless cross-thread wakeups per second (backend thread pays
the loop lock + self-pipe write, the uvicorn loop pays the wakeup + callback),
forever after the first app run.

These tests pin the contract: no clients -> no scheduling; a connected client
still receives broadcast messages.
"""

import asyncio
import time
from threading import Thread
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from fastapi import WebSocket, WebSocketDisconnect

from reachy_mini.daemon.daemon import Daemon
from reachy_mini.io.protocol import DaemonState, DaemonStatus
from reachy_mini.io.ws_server import WSServer

STATUS = DaemonStatus(
    robot_name="test",
    state=DaemonState.RUNNING,
    wireless_version=False,
    desktop_app_daemon=False,
    simulation_enabled=False,
    mockup_sim_enabled=False,
    backend_status=None,
)


class FakeWebSocket:
    """Minimal stand-in for fastapi.WebSocket driven by the test."""

    def __init__(self) -> None:
        """Create a connected-looking fake with no messages sent yet."""
        self.sent: list[str] = []
        self._disconnected = asyncio.Event()

    async def accept(self) -> None:
        """Accept the connection (no-op)."""

    async def send_text(self, msg: str) -> None:
        """Record an outbound message."""
        self.sent.append(msg)

    async def receive_text(self) -> str:
        """Block until the test disconnects, then raise like a closed socket."""
        await self._disconnected.wait()
        raise WebSocketDisconnect(1000)

    def disconnect(self) -> None:
        """Make receive_text raise WebSocketDisconnect."""
        self._disconnected.set()


@pytest.mark.asyncio
async def test_broadcast_schedules_nothing_after_last_client_disconnects() -> None:
    """Once every client is gone, _broadcast must not touch the event loop.

    This is the idle-daemon spin bug: _loop stays captured after the first
    connection, and each _broadcast call paid a call_soon_threadsafe (loop
    lock + self-pipe write) just to iterate an empty client set.
    """
    server = WSServer(backend=MagicMock())

    ws = FakeWebSocket()
    ws.disconnect()  # receive_text raises immediately: connect then disconnect
    await server.handle_client(cast(WebSocket, ws))
    await asyncio.sleep(0)  # let the cancelled send task finish

    assert not server._clients

    loop = asyncio.get_running_loop()
    with patch.object(
        loop, "call_soon_threadsafe", wraps=loop.call_soon_threadsafe
    ) as spy:
        for _ in range(50):
            server._broadcast('{"type": "joint_positions"}')

    assert spy.call_count == 0


@pytest.mark.asyncio
async def test_broadcast_delivers_to_connected_client() -> None:
    """A connected client still receives broadcast messages."""
    server = WSServer(backend=MagicMock())

    ws = FakeWebSocket()
    client_task = asyncio.create_task(server.handle_client(cast(WebSocket, ws)))

    async with asyncio.timeout(2):
        while not server._clients:
            await asyncio.sleep(0.01)

        server._broadcast('{"type": "status"}')
        while not ws.sent:
            await asyncio.sleep(0.01)

    assert ws.sent == ['{"type": "status"}']

    ws.disconnect()
    await client_task


@pytest.mark.asyncio
async def test_broadcast_before_any_client_is_noop() -> None:
    """Before the first connection there is no loop, so _broadcast returns."""
    server = WSServer(backend=MagicMock())
    server._broadcast('{"type": "status"}')  # must not raise


@pytest.mark.asyncio
async def test_new_client_receives_status_without_waiting_for_tick() -> None:
    """The status thread wakes on connect and sends a status read after it.

    It used to sleep 1 s between publishes, so ReachyMini() waited up to 1 s
    for its first status during media setup.
    """
    server = WSServer(backend=MagicMock())
    daemon = Daemon(no_media=True)
    daemon.ws_server = server
    reads: list[float] = []

    def status() -> DaemonStatus:
        reads.append(time.monotonic())
        return STATUS.model_copy(update={"robot_name": str(len(reads) - 1)})

    ws = FakeWebSocket()
    accepted_at = 0.0

    async def accept() -> None:
        nonlocal accepted_at
        accepted_at = time.monotonic()

    def read_after_accept() -> bool:
        return any(
            reads[int(DaemonStatus.model_validate_json(msg).robot_name)] >= accepted_at
            for msg in ws.sent
        )

    with patch.object(daemon, "status", side_effect=status):
        thread = Thread(target=daemon._publish_status, daemon=True)
        thread.start()
        async with asyncio.timeout(2):
            while not reads:  # first tick, no client yet
                await asyncio.sleep(0.01)

        with patch.object(ws, "accept", side_effect=accept):
            client_task = asyncio.create_task(server.handle_client(cast(WebSocket, ws)))
            # The first tick may still land after registration; what matters
            # is a status read after connect, well before the next 1 s tick.
            async with asyncio.timeout(0.5):
                while not read_after_accept():
                    await asyncio.sleep(0.01)

        daemon._thread_event_publish_status.set()
        server.client_connected_event().set()
        thread.join(timeout=2)

    ws.disconnect()
    await client_task
