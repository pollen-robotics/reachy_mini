"""Unit tests for the reusable app-side JSON-RPC server (apps/jsonrpc_server).

Exercised with a fake WebSocket so no FastAPI/uvicorn server is needed.
"""

import json
from typing import Any

import pytest

from reachy_mini.apps.jsonrpc_server import JsonRpcServer
from reachy_mini.io.jsonrpc import JsonRpcError


class FakeWs:
    """Records send_text calls; that's all the server needs from a peer."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send_text(self, text: str) -> None:
        self.sent.append(json.loads(text))


@pytest.mark.asyncio
async def test_dispatch_result() -> None:
    server = JsonRpcServer()

    @server.method("conversation.status")
    def _status(_params: dict[str, Any]) -> dict[str, Any]:
        return {"phase": "running"}

    ws = FakeWs()
    await server._dispatch(ws, '{"jsonrpc":"2.0","id":"1","method":"conversation.status"}')
    assert ws.sent == [{"jsonrpc": "2.0", "id": "1", "result": {"phase": "running"}}]


@pytest.mark.asyncio
async def test_dispatch_async_handler_and_params() -> None:
    server = JsonRpcServer()

    @server.method("conversation.say")
    async def _say(params: dict[str, Any]) -> dict[str, Any]:
        return {"echo": params["text"]}

    ws = FakeWs()
    await server._dispatch(
        ws, '{"jsonrpc":"2.0","id":"7","method":"conversation.say","params":{"text":"hi"}}'
    )
    assert ws.sent[0]["result"] == {"echo": "hi"}


@pytest.mark.asyncio
async def test_unknown_method() -> None:
    server = JsonRpcServer()
    ws = FakeWs()
    await server._dispatch(ws, '{"jsonrpc":"2.0","id":"1","method":"nope.foo"}')
    assert ws.sent[0]["error"]["data"]["reason"] == "method_not_found"


@pytest.mark.asyncio
async def test_handler_raises_jsonrpc_error() -> None:
    server = JsonRpcServer()

    @server.method("conversation.say")
    def _say(_params: dict[str, Any]) -> dict[str, Any]:
        raise JsonRpcError("no session", reason="not_running")

    ws = FakeWs()
    await server._dispatch(ws, '{"jsonrpc":"2.0","id":"1","method":"conversation.say"}')
    assert ws.sent[0]["error"]["data"]["reason"] == "not_running"


@pytest.mark.asyncio
async def test_notification_gets_no_response() -> None:
    server = JsonRpcServer()

    @server.method("conversation.ping")
    def _ping(_params: dict[str, Any]) -> None:
        return None

    ws = FakeWs()
    # No id -> notification -> no reply, even on success.
    await server._dispatch(ws, '{"jsonrpc":"2.0","method":"conversation.ping"}')
    assert ws.sent == []


@pytest.mark.asyncio
async def test_broadcast_reaches_all_clients() -> None:
    server = JsonRpcServer()
    a, b = FakeWs(), FakeWs()
    server._clients.add(a)  # type: ignore[arg-type]
    server._clients.add(b)  # type: ignore[arg-type]
    await server.broadcast("conversation.turn", {"state": "speaking"})
    for ws in (a, b):
        assert ws.sent[0]["method"] == "conversation.turn"
        assert ws.sent[0]["params"] == {"state": "speaking"}
