"""Tests for the SDK WebSocket server."""

import asyncio
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from fastapi import WebSocket, WebSocketDisconnect

from reachy_mini.io.protocol import DaemonState, DaemonStatus
from reachy_mini.io.ws_server import WSServer


class _FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self.accepted = asyncio.Event()
        self.message_sent = asyncio.Event()
        self._disconnected = asyncio.Event()

    async def accept(self) -> None:
        self.accepted.set()

    async def send_text(self, message: str) -> None:
        self.sent.append(message)
        self.message_sent.set()

    async def receive_text(self) -> str:
        await self._disconnected.wait()
        raise WebSocketDisconnect(1000)

    def disconnect(self) -> None:
        self._disconnected.set()


@pytest.mark.asyncio
async def test_new_client_receives_current_status_immediately() -> None:
    """Check that a new client immediately receives the current status."""
    status = DaemonStatus(
        robot_name="test",
        state=DaemonState.RUNNING,
        wireless_version=False,
        desktop_app_daemon=False,
        simulation_enabled=False,
        mockup_sim_enabled=False,
        backend_status=None,
    )
    server = WSServer(
        backend=MagicMock(),
        status_provider=lambda: status,
    )
    websocket = _FakeWebSocket()
    client_task = asyncio.create_task(server.handle_client(cast(WebSocket, websocket)))

    async with asyncio.timeout(1):
        await websocket.message_sent.wait()

    websocket.disconnect()
    await client_task

    assert len(websocket.sent) == 1
    assert DaemonStatus.model_validate_json(websocket.sent[0]) == status


@pytest.mark.asyncio
async def test_status_publish_schedules_nothing_after_last_client_disconnects() -> None:
    """Check that publishing schedules nothing after the last disconnect."""
    server = WSServer(backend=MagicMock())
    websocket = _FakeWebSocket()
    websocket.disconnect()
    await server.handle_client(cast(WebSocket, websocket))
    await asyncio.sleep(0)

    loop = asyncio.get_running_loop()
    with patch.object(
        loop, "call_soon_threadsafe", wraps=loop.call_soon_threadsafe
    ) as call_soon:
        for _ in range(50):
            server.publish_status('{"type": "daemon_status"}')

    assert call_soon.call_count == 0


@pytest.mark.asyncio
async def test_status_publish_delivers_to_connected_client() -> None:
    """Check that publishing delivers status to a connected client."""
    message = '{"type": "daemon_status"}'
    server = WSServer(backend=MagicMock())
    websocket = _FakeWebSocket()
    client_task = asyncio.create_task(server.handle_client(cast(WebSocket, websocket)))

    async with asyncio.timeout(1):
        await websocket.accepted.wait()
        server.publish_status(message)
        await websocket.message_sent.wait()

    websocket.disconnect()
    await client_task

    assert websocket.sent == [message]


@pytest.mark.asyncio
async def test_status_publish_before_any_client_schedules_nothing() -> None:
    """Check that publishing before the first connection schedules nothing."""
    server = WSServer(backend=MagicMock())
    loop = asyncio.get_running_loop()

    with patch.object(
        loop, "call_soon_threadsafe", wraps=loop.call_soon_threadsafe
    ) as call_soon:
        server.publish_status('{"type": "daemon_status"}')

    assert call_soon.call_count == 0
