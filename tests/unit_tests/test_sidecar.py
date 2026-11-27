"""The Rust daemon's sidecar: the daemon's app it asks about, the JSON-RPC it serves."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient

from reachy_mini.daemon.app.sidecar import DaemonApps, create_app

RUNNING: dict[str, Any] = {
    "info": {
        "name": "app_a",
        "source_kind": "installed",
        "description": "",
        "url": None,
        "extra": {},
    },
    "state": "running",
    "error": None,
}


@pytest.fixture
def daemon() -> Iterator[tuple[str, dict[str, Any]]]:
    """Serve a fake daemon /api/apps: its status, and a stop that clears it."""
    state: dict[str, Any] = {"status": RUNNING, "stops": 0}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: Any) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            self._send(200, state["status"])

        def do_POST(self) -> None:
            state["stops"] += 1
            if state["status"] is None:
                return self._send(400, {"detail": "No app is currently running"})
            state["status"] = None
            self._send(200, None)

        def log_message(self, *args: Any) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", state
    server.shutdown()


@pytest.mark.asyncio
async def test_the_running_app_is_the_daemons(
    daemon: tuple[str, dict[str, Any]],
) -> None:
    """Running app and stop come from the daemon; starting isn't the sidecar's."""
    url, state = daemon
    apps = DaemonApps(
        url, wireless_version=False, desktop_app_daemon=False, daemon=None
    )

    assert apps.is_app_running()
    assert apps.current_app.status.info.name == "app_a"
    await apps.stop_current_app()
    assert not apps.is_app_running()
    with pytest.raises(RuntimeError, match="No app is currently running"):
        await apps.stop_current_app()
    assert state["stops"] == 2
    with pytest.raises(RuntimeError):
        await apps.start_app("app_a")
    # A daemon gone reads as no app, not an error.
    assert DaemonApps("http://127.0.0.1:9", daemon=None).current_app is None


def test_rpc_replies_go_back_to_their_client() -> None:
    """A client's frame is answered to that client."""
    app = create_app(False, False, "http://127.0.0.1:9")
    with TestClient(app) as client, client.websocket_connect("/internal/rpc") as ws:
        frame = {"jsonrpc": "2.0", "id": 7, "method": "apps.status"}
        ws.send_text(json.dumps({"client": "peer-1", "frame": json.dumps(frame)}))
        out = json.loads(ws.receive_text())
        assert out["client"] == "peer-1"
        reply = json.loads(out["frame"])
        assert reply["id"] == 7 and "result" in reply
