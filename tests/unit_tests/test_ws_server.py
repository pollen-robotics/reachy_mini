"""Tests for the SDK WebSocket server."""

import asyncio
from unittest.mock import MagicMock

import pytest
from fastapi import WebSocketDisconnect

from reachy_mini.io.protocol import DaemonState, DaemonStatus
from reachy_mini.io.ws_server import WSServer


class _FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self._status_sent = asyncio.Event()

    async def accept(self) -> None:
        pass

    async def send_text(self, message: str) -> None:
        self.sent.append(message)
        self._status_sent.set()

    async def receive_text(self) -> str:
        await self._status_sent.wait()
        raise WebSocketDisconnect()


@pytest.mark.asyncio
async def test_new_client_receives_current_status_immediately() -> None:
    """A new client receives a fresh status without waiting for the 1 Hz publisher."""
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

    await asyncio.wait_for(
        server.handle_client(websocket),
        timeout=1,
    )

    assert len(websocket.sent) == 1
    assert DaemonStatus.model_validate_json(websocket.sent[0]) == status
