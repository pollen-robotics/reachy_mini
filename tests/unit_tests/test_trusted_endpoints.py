"""Authenticated clients reject redirects across the actual HTTP boundary."""

import asyncio
from collections.abc import AsyncIterator, Callable

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestServer
from fastapi import APIRouter
from fastapi.testclient import TestClient

from reachy_mini.apps.sources import hf_auth as auth_source
from reachy_mini.daemon.app.routers import hf_auth
from reachy_mini.media import central_signaling_relay, webrtc_utils
from reachy_mini.media.central_consumer import ReachyCentralConsumer
from reachy_mini.media.central_signaling_relay import CentralSignalingRelay, RelayState

EndpointServer = tuple[TestServer, list[web.Request], dict[str, tuple[int, object]]]


@pytest_asyncio.fixture
async def endpoint_server(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[EndpointServer]:
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    monkeypatch.setattr("huggingface_hub.get_token", lambda: "hf_test")
    requests: list[web.Request] = []
    responses: dict[str, tuple[int, object]] = {}
    finished = asyncio.Event()

    async def handle(request: web.Request) -> web.StreamResponse:
        requests.append(request)
        if request.path in responses:
            status, payload = responses[request.path]
            return web.json_response(
                payload, status=status, headers={"Location": "/redirected"}
            )
        if request.path == "/local":
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            await ws.send_json({"type": "welcome", "peerId": "local"})
            async for message in ws:
                assert message.json()["type"] == "setPeerStatus"
            return ws
        if request.path == "/base/events":
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            await response.write(b'data: {"type":"welcome","peerId":"peer"}\n\n')
            await finished.wait()
            return response
        if request.path == "/base/api/robot-status":
            return web.json_response({"robots": [{"peerId": "peer"}]})
        if request.path == "/base/send":
            return web.json_response({"type": "sessionStarted", "sessionId": "session"})
        return web.Response(status=404)

    app = web.Application()
    app.router.add_route("*", "/{path:.*}", handle)
    server = TestServer(app)
    async with server:
        try:
            yield server, requests, responses
        finally:
            finished.set()
    assert not any(request.path == "/redirected" for request in requests)
    for request in requests:
        if request.path != "/local":
            assert request.headers.get("Authorization") == "Bearer hf_test"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["consumer", "relay"])
@pytest.mark.parametrize(
    ("path", "status"),
    [("/send", 200)]
    + [
        (path, status)
        for path in ("/events", "/send", "/api/robot-status")
        for status in (302, 307, 308)
    ],
)
async def test_signaling_authenticates_without_following_redirects(
    endpoint_server: EndpointServer,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    kind: str,
    path: str,
    status: int,
) -> None:
    server, requests, responses = endpoint_server
    if status != 200:
        responses[f"/base{path}"] = (
            status,
            {
                "robots": [{"peerId": "untrusted"}],
                "type": "sessionStarted",
                "sessionId": "untrusted",
            },
        )
    central_url = str(server.make_url("/base/"))
    client: ReachyCentralConsumer | CentralSignalingRelay
    if kind == "consumer":
        client = ReachyCentralConsumer(hf_token="hf_test", central_url=central_url)
    else:
        monkeypatch.setattr(
            central_signaling_relay, "PRODUCER_HEALTH_CHECK_INITIAL_DELAY", 0
        )
        monkeypatch.setattr(
            central_signaling_relay, "PRODUCER_HEALTH_CHECK_INTERVAL", 0.02
        )
        client = CentralSignalingRelay(
            central_uri=central_url,
            local_uri=str(server.make_url("/local").with_scheme("ws")),
            hf_token="hf_test",
        )
    caplog.set_level("DEBUG", logger=central_signaling_relay.__name__)
    await client.start()
    try:
        async with asyncio.timeout(5):
            if status != 200:
                while f"HTTP {status}" not in caplog.text:
                    await asyncio.sleep(0.01)
            elif isinstance(client, ReachyCentralConsumer):
                while client.status()["session_id"] is None:
                    await asyncio.sleep(0.01)
            else:
                while client.state != RelayState.CONNECTED:
                    await asyncio.sleep(0.01)
        assert any(request.path == f"/base{path}" for request in requests)
        if isinstance(client, ReachyCentralConsumer):
            if status == 200:
                assert client.status()["session_id"] == "session"
                assert client.status()["robot_peer_id"] == "peer"
            else:
                assert client.status()["session_id"] is None
                assert client.status()["robot_peer_id"] != "untrusted"
    finally:
        if isinstance(client, CentralSignalingRelay):
            await asyncio.to_thread(asyncio.run, client.stop())
        else:
            await client.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/credentials", "/credentials/"])
@pytest.mark.parametrize("status", [302, 307, 308, 401, 503])
async def test_turn_keeps_credentials_when_refresh_fails(
    endpoint_server: EndpointServer,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    path: str,
    status: int,
) -> None:
    server, requests, responses = endpoint_server
    responses[path] = (
        200,
        {
            "iceServers": [
                {"urls": "turn:previous.example", "username": "u", "credential": "p"}
            ]
        },
    )
    monkeypatch.setattr(webrtc_utils, "_TURN_REFRESH_RATIO", 0.0001)
    credentials = webrtc_utils.TurnCredentials(url=str(server.make_url(path)), ttl=600)
    credentials.start()
    try:
        async with asyncio.timeout(5):
            while not credentials.turn_uris():
                await asyncio.sleep(0.01)
            responses[path] = (
                status,
                {
                    "iceServers": [
                        {
                            "urls": "turn:untrusted.example",
                            "username": "u",
                            "credential": "p",
                        }
                    ]
                },
            )
            while f"HTTP {status}" not in caplog.text:
                await asyncio.sleep(0.01)
        assert credentials.turn_uris() == ["turn://u:p@previous.example"]
        assert all(request.path == path for request in requests)
        assert all(request.query == {"ttl": "600"} for request in requests)
    finally:
        credentials.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200, 302, 307, 308, 401, 503])
async def test_router_reports_authenticated_response(
    endpoint_server: EndpointServer,
    monkeypatch: pytest.MonkeyPatch,
    router_app: Callable[[APIRouter], TestClient],
    status: int,
) -> None:
    server, requests, responses = endpoint_server
    path = "/base/api/robot-status"
    responses[path] = (
        status,
        {"robots": [{"peerId": "peer"}]},
    )
    monkeypatch.setattr(hf_auth, "CENTRAL_ROBOT_STATUS_URL", str(server.make_url(path)))
    monkeypatch.setattr(auth_source, "get_hf_token", lambda: "hf_test")
    response = await asyncio.to_thread(
        router_app(hf_auth.router).get, "/hf-auth/central-robot-status"
    )
    assert response.status_code == 200
    assert len(requests) == 1
    if status == 200:
        assert response.json() == {"available": True, "robots": [{"peerId": "peer"}]}
    else:
        reason = "token_invalid" if status == 401 else f"http_{status}"
        assert response.json() == {"available": False, "robots": [], "reason": reason}
