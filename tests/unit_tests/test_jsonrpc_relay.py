"""Unit tests for the JSON-RPC envelope + daemon relay routing.

Covers the three things the relay must get right:
* ``apps.*`` handled locally against the AppManager,
* other namespaces relayed to the app with the caller's ``id`` restored,
* app notifications (no ``id``) fanned out to every client.
"""

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest

from reachy_mini.daemon import jsonrpc_relay
from reachy_mini.daemon.jsonrpc_relay import JsonRpcRelay
from reachy_mini.io.jsonrpc import (
    looks_like_jsonrpc,
    make_error,
    parse_request,
)


# ---------------------------------------------------------------- envelope


def test_looks_like_jsonrpc() -> None:
    assert looks_like_jsonrpc({"jsonrpc": "2.0", "method": "x"})
    assert not looks_like_jsonrpc({"type": "set_target"})
    assert not looks_like_jsonrpc("not a dict")


def test_parse_request_notification_vs_call() -> None:
    call = parse_request('{"jsonrpc":"2.0","id":"1","method":"conversation.say"}')
    assert not call.is_notification
    assert call.namespace == "conversation"
    note = parse_request('{"jsonrpc":"2.0","method":"conversation.turn"}')
    assert note.is_notification


def test_error_reason_lives_in_data() -> None:
    # The deliberate spec deviation: the string code the UI branches on is in
    # error.data.reason, error.code stays an int.
    err = make_error(None, message="busy", reason="robot_busy", code=-32000)
    assert err["error"]["code"] == -32000
    assert err["error"]["data"]["reason"] == "robot_busy"


# ------------------------------------------------------------------ fakes


class FakeAppManager:
    """Minimal AppManager stand-in for relay routing tests."""

    def __init__(self, url: str | None = "http://0.0.0.0:7860/") -> None:
        self._url = url
        self.current_app: Any = None
        self.started: list[str] = []
        self.stopped = 0

    def is_app_running(self) -> bool:
        return self.current_app is not None

    def get_running_app_url(self) -> str | None:
        return self._url

    async def start_app(self, name: str, keep_remote: bool = False) -> Any:
        self.started.append(name)
        self.last_keep_remote = keep_remote
        self.current_app = SimpleNamespace(
            status=SimpleNamespace(
                state="running",
                error=None,
                info=SimpleNamespace(name=name, description="", url=None),
            )
        )
        return self.current_app.status

    async def stop_current_app(self) -> None:
        self.stopped += 1
        self.current_app = None


class FakeAppWs:
    """Fake app /rpc connection: records sends, streams injected frames."""

    def __init__(self) -> None:
        self.sent: list[str] = []
        self._incoming: asyncio.Queue[str | None] = asyncio.Queue()
        self.closed = False

    async def send(self, text: str) -> None:
        self.sent.append(text)

    async def close(self) -> None:
        self.closed = True
        await self._incoming.put(None)  # end the reader

    def inject(self, obj: dict[str, Any]) -> None:
        self._incoming.put_nowait(json.dumps(obj))

    def __aiter__(self) -> "FakeAppWs":
        return self

    async def __anext__(self) -> str:
        item = await self._incoming.get()
        if item is None:
            raise StopAsyncIteration
        return item


# ------------------------------------------------------------- apps.* local


@pytest.mark.asyncio
async def test_apps_status_idle() -> None:
    relay = JsonRpcRelay(FakeAppManager(), broadcast=lambda _: None)
    replies: list[dict[str, Any]] = []
    await relay.handle(
        '{"jsonrpc":"2.0","id":"1","method":"apps.status"}', replies.append
    )
    assert replies[0]["id"] == "1"
    assert replies[0]["result"]["state"] == "idle"


@pytest.mark.asyncio
async def test_apps_start_and_stop() -> None:
    apps = FakeAppManager()
    relay = JsonRpcRelay(apps, broadcast=lambda _: None)
    replies: list[dict[str, Any]] = []

    await relay.handle(
        '{"jsonrpc":"2.0","id":"1","method":"apps.start","params":{"name":"conv"}}',
        replies.append,
    )
    assert apps.started == ["conv"]
    assert apps.last_keep_remote is True
    assert replies[-1]["result"]["state"] == "running"
    assert replies[-1]["result"]["info"]["name"] == "conv"

    await relay.handle(
        '{"jsonrpc":"2.0","id":"2","method":"apps.stop"}', replies.append
    )
    assert apps.stopped == 1
    assert replies[-1]["result"] == {"stopped": True}


@pytest.mark.asyncio
async def test_apps_start_requires_name() -> None:
    relay = JsonRpcRelay(FakeAppManager(), broadcast=lambda _: None)
    replies: list[dict[str, Any]] = []
    await relay.handle(
        '{"jsonrpc":"2.0","id":"1","method":"apps.start"}', replies.append
    )
    assert replies[0]["error"]["data"]["reason"] == "invalid_params"


# ------------------------------------------------------------- relay to app


@pytest.mark.asyncio
async def test_relay_forwards_and_restores_id(monkeypatch: pytest.MonkeyPatch) -> None:
    ws = FakeAppWs()

    async def fake_connect(url: str) -> FakeAppWs:
        return ws

    monkeypatch.setattr(jsonrpc_relay, "connect", fake_connect)

    broadcasted: list[str] = []
    relay = JsonRpcRelay(FakeAppManager(), broadcast=broadcasted.append)
    replies: list[dict[str, Any]] = []

    await relay.handle(
        '{"jsonrpc":"2.0","id":"caller-9","method":"conversation.say",'
        '"params":{"text":"hi"}}',
        replies.append,
    )

    # The frame reached the app with a relay-assigned id (not the caller's).
    assert len(ws.sent) == 1
    sent = json.loads(ws.sent[0])
    assert sent["method"] == "conversation.say"
    assert sent["id"] != "caller-9"
    relay_id = sent["id"]

    # App answers on the relay id; the reader restores the caller's id.
    ws.inject({"jsonrpc": "2.0", "id": relay_id, "result": {"accepted": True}})
    await asyncio.sleep(0.05)
    assert replies == [
        {"jsonrpc": "2.0", "id": "caller-9", "result": {"accepted": True}}
    ]

    # A notification (no id) from the app is broadcast to all clients.
    ws.inject(
        {
            "jsonrpc": "2.0",
            "method": "conversation.turn",
            "params": {"state": "speaking"},
        }
    )
    await asyncio.sleep(0.05)
    assert len(broadcasted) == 1
    assert json.loads(broadcasted[0])["method"] == "conversation.turn"

    await relay.aclose()


@pytest.mark.asyncio
async def test_relay_no_app_running() -> None:
    relay = JsonRpcRelay(FakeAppManager(url=None), broadcast=lambda _: None)
    replies: list[dict[str, Any]] = []
    await relay.handle(
        '{"jsonrpc":"2.0","id":"1","method":"conversation.say","params":{"text":"x"}}',
        replies.append,
    )
    assert replies[0]["error"]["data"]["reason"] == "not_running"
