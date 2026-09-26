"""Unit tests for LocalNetworkGuardMiddleware (Host/Origin validation).

The guard blocks the two browser vectors that reach the unauthenticated API
from the internet (CAN-2026-2032024): DNS rebinding (untrusted Host header)
and preflight-free cross-site writes (untrusted Origin on state-changing
methods). Everything a legitimate client does must keep working: same-origin
dashboard calls, Origin-less curl/SDK requests, the desktop app webviews, and
access by raw LAN IP or .local mDNS name.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

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
