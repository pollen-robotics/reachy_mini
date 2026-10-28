"""The Rust daemon's sidecar: the app lock it asks, the JSON-RPC it serves."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient

from reachy_mini.daemon.app.sidecar import RemoteRobotAppLock, create_app


@pytest.fixture
def daemon() -> Iterator[tuple[str, list[dict[str, Any]], list[int]]]:
    """Serve a fake /internal/app-lock, answering each POST with the next status."""
    seen: list[dict[str, Any]] = []
    statuses: list[int] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers["Content-Length"]))
            seen.append({"path": self.path, **json.loads(body)})
            self.send_response(statuses.pop(0) if statuses else 200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args: Any) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", seen, statuses
    server.shutdown()


@pytest.mark.asyncio
async def test_the_lock_is_the_daemons(
    daemon: tuple[str, list[dict[str, Any]], list[int]],
) -> None:
    """Each AppManager call is a request to the daemon's lock."""
    url, seen, statuses = daemon
    lock = RemoteRobotAppLock(url)

    assert lock.try_acquire_local("a")
    statuses.append(409)
    assert not lock.try_acquire_local("b")
    statuses.append(409)
    with pytest.raises(RuntimeError):
        await lock.acquire_local_evicting_remote("c")
    lock.acquire_local_keeping_remote("d")
    lock.release_local("d")
    assert [(s["path"], s.get("mode")) for s in seen] == [
        ("/internal/app-lock/acquire", "try"),
        ("/internal/app-lock/acquire", "try"),
        ("/internal/app-lock/acquire", "evict"),
        ("/internal/app-lock/acquire", "keep"),
        ("/internal/app-lock/release", None),
    ]
    # Releasing runs as an app exits: a daemon gone must not raise.
    RemoteRobotAppLock("http://127.0.0.1:9").release_local("d")


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
