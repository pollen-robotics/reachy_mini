"""Unit tests for LocalNetworkGuardMiddleware (Host/Origin validation).

The guard blocks the browser vectors that reach the unauthenticated API from
the internet (CAN-2026-2032024): DNS rebinding (untrusted Host header),
preflight-free cross-site writes (untrusted Origin on state-changing
methods), and cross-site WebSocket hijacking (browsers apply no CORS to
WebSockets). Everything a legitimate client does must keep working:
same-origin dashboard calls, Origin-less curl/SDK requests and WS
connections, the desktop app webviews, and access by raw LAN IP or .local
mDNS name.
"""

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from reachy_mini.daemon.app.middleware import LocalNetworkGuardMiddleware


@pytest.fixture
def client():
    """TestClient factory for a minimal guarded app, parametrized by base_url."""

    def _make(base_url: str = "http://reachy-mini.local:8000") -> TestClient:
        app = FastAPI()

        @app.post("/write")
        async def write() -> dict[str, str]:
            return {"status": "ok"}

        @app.get("/read")
        async def read() -> dict[str, str]:
            return {"status": "ok"}

        @app.websocket("/ws")
        async def ws(websocket: WebSocket) -> None:
            await websocket.accept()
            await websocket.send_text("connected")
            await websocket.close()

        app.add_middleware(LocalNetworkGuardMiddleware)
        return TestClient(app, base_url=base_url)

    return _make


def test_same_origin_dashboard_post(client):
    """The dashboard's own POSTs (matching Origin) go through."""
    resp = client().post("/write", headers={"origin": "http://reachy-mini.local:8000"})
    assert resp.status_code == 200


def test_originless_clients_pass(client):
    """curl / the Python SDK send no Origin header and are unaffected."""
    resp = client().post("/write")
    assert resp.status_code == 200


@pytest.mark.parametrize(
    "origin",
    [
        "https://evil.example.com",
        "http://8.8.8.8",  # public IP literal
        "null",  # sandboxed iframe / data: URL
        "file://evil",
    ],
)
def test_untrusted_origins_rejected_on_writes(client, origin):
    """Cross-site state-changing requests are rejected server-side (CSRF)."""
    resp = client().post("/write", headers={"origin": origin})
    assert resp.status_code == 403
    assert resp.json() == {"detail": "Cross-origin request rejected"}


@pytest.mark.parametrize(
    "origin",
    [
        "http://localhost:3000",  # local dev tooling
        "http://192.168.1.50:8042",  # app frontend on another LAN device/port
        "http://another-robot.local",  # mDNS neighbour
        "tauri://localhost",  # desktop app webview (macOS/Linux)
        "http://tauri.localhost",  # desktop app webview (Windows)
        "capacitor://localhost",  # mobile app webview
    ],
)
def test_local_origins_allowed_on_writes(client, origin):
    """Local, private-network, and native-webview origins stay allowed."""
    resp = client().post("/write", headers={"origin": origin})
    assert resp.status_code == 200


def test_untrusted_origin_reads_pass_guard(client):
    """GETs are side-effect free: the guard defers to CORS for read privacy."""
    resp = client().get("/read", headers={"origin": "https://evil.example.com"})
    assert resp.status_code == 200


def test_dns_rebinding_host_rejected(client):
    """A public DNS name in Host (rebound to the robot's IP) is rejected."""
    resp = client("http://rebind.evil.example.com:8000").post("/write")
    assert resp.status_code == 400
    assert resp.json() == {"detail": "Untrusted Host header"}
    # Even reads are rejected: the rebound page is same-origin and could
    # otherwise exfiltrate GET responses.
    resp = client("http://rebind.evil.example.com:8000").get("/read")
    assert resp.status_code == 400


@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost:8000",  # Lite / desktop proxy
        "http://127.0.0.1:8000",
        "http://reachy-mini.local:8000",  # wireless mDNS
        "http://192.168.1.42:8000",  # wireless raw IP
    ],
)
def test_legitimate_hosts_accepted(client, base_url):
    """All documented ways of addressing the robot keep working."""
    resp = client(base_url).post("/write", headers={"origin": base_url})
    assert resp.status_code == 200


def test_public_ip_origin_rejected_even_when_host_matches(client):
    """No same-as-Host shortcut: a public-IP origin stays untrusted.

    If the robot were reachable at a public IP, a page served from that IP
    must not gain write access just because Origin == Host.
    """
    resp = client("http://8.8.8.8:8000").post(
        "/write", headers={"origin": "http://8.8.8.8:8000"}
    )
    assert resp.status_code == 403


def test_duplicate_host_or_origin_headers_rejected(client):
    """Duplicate Host/Origin headers are denied outright (smuggling seam).

    uvicorn forwards requests with duplicate Host headers even though
    RFC 9112 mandates rejection; picking either copy invites disagreement
    with any proxy in front. Browsers cannot emit duplicates, so nothing
    legitimate is lost.
    """
    resp = client().post(
        "/write",
        headers=[("host", "reachy-mini.local:8000"), ("host", "evil.example.com")],
    )
    assert resp.status_code == 400
    assert resp.json() == {"detail": "Duplicate Host or Origin header"}

    resp = client().post(
        "/write",
        headers=[
            ("origin", "http://reachy-mini.local:8000"),
            ("origin", "https://evil.example.com"),
        ],
    )
    assert resp.status_code == 400


def test_webview_scheme_trusted_for_localhost_only(client):
    """tauri:// and capacitor:// are only trusted with a localhost host."""
    for origin in ("tauri://evil.example.com", "capacitor://evil.example.com"):
        resp = client().post("/write", headers={"origin": origin})
        assert resp.status_code == 403, origin


def test_trailing_dot_fqdn_local_accepted(client):
    """Browsers may send the FQDN form 'reachy-mini.local.' -- still local."""
    resp = client("http://reachy-mini.local.:8000").post(
        "/write", headers={"origin": "http://reachy-mini.local.:8000"}
    )
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# WebSocket handshakes (browsers apply no CORS to WebSockets, so the guard is
# the only cross-site defence). NOTE: TestClient hardcodes ``host: testserver``
# on WS handshakes regardless of base_url, so every case sets Host explicitly.
# ---------------------------------------------------------------------------

_WS_LOCAL_HOST = {"host": "reachy-mini.local:8000"}


def test_ws_originless_sdk_connects(client):
    """The Python SDK opens WS connections with no Origin header."""
    with client().websocket_connect("/ws", headers=_WS_LOCAL_HOST) as ws:
        assert ws.receive_text() == "connected"


@pytest.mark.parametrize(
    "origin",
    [
        "http://reachy-mini.local:8000",  # dashboard same-origin
        "http://localhost:3000",
        "tauri://localhost",
        "capacitor://localhost",
    ],
)
def test_ws_local_origins_connect(client, origin):
    """Local and webview origins can open WebSockets."""
    headers = {**_WS_LOCAL_HOST, "origin": origin}
    with client().websocket_connect("/ws", headers=headers) as ws:
        assert ws.receive_text() == "connected"


@pytest.mark.parametrize(
    "origin",
    ["https://evil.example.com", "http://8.8.8.8", "null"],
)
def test_ws_cross_site_hijack_rejected(client, origin):
    """A drive-by page cannot open a WebSocket (would be readable + writable)."""
    headers = {**_WS_LOCAL_HOST, "origin": origin}
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client().websocket_connect("/ws", headers=headers):
            pass
    assert exc_info.value.code == 1008


def test_ws_dns_rebinding_host_rejected(client):
    """A rebound public DNS name cannot open a WebSocket, even Origin-less."""
    headers = {"host": "rebind.evil.example.com:8000"}
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client().websocket_connect("/ws", headers=headers):
            pass
    assert exc_info.value.code == 1008
