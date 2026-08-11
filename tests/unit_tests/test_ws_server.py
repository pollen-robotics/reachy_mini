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
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from fastapi import WebSocket, WebSocketDisconnect

from reachy_mini.io.ws_server import WSServer


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
