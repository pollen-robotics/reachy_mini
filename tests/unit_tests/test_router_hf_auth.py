"""Unit tests for the HuggingFace auth router (delegating to apps.sources.hf_auth)."""

import types

import pytest

from reachy_mini.apps.sources import hf_auth as src
from reachy_mini.daemon.app.routers import hf_auth

# The router does ``from reachy_mini.apps.sources import hf_auth`` and calls
# ``hf_auth.<fn>``, so the delegated functions live on the source module (``src``);
# patch there, not on the router namespace.


def _daemon(wireless_version=False):
    """Minimal app.state.daemon stub the handlers read."""
    return types.SimpleNamespace(wireless_version=wireless_version)


def test_save_token_success(monkeypatch, router_app):
    """Valid token -> 200 with username echoed back."""
    monkeypatch.setattr(
        src,
        "save_hf_token",
        lambda tok: {"status": "success", "username": "alice"},
    )
    client = router_app(hf_auth.router)

    resp = client.post("/hf-auth/save-token", json={"token": "hf_x"})

    assert resp.status_code == 200
    assert resp.json() == {"status": "success", "username": "alice", "message": None}


def test_save_token_failure(monkeypatch, router_app):
    """Rejected token -> 400 with error detail."""
    monkeypatch.setattr(
        src,
        "save_hf_token",
        lambda tok: {"status": "error", "message": "bad token"},
    )
    client = router_app(hf_auth.router)

    resp = client.post("/hf-auth/save-token", json={"token": "nope"})

    assert resp.status_code == 400
    assert resp.json()["detail"] == "bad token"


def test_status(monkeypatch, router_app):
    """/status returns whatever check_token_status reports."""
    payload = {"authenticated": True, "username": "bob"}
    monkeypatch.setattr(src, "check_token_status", lambda: payload)
    client = router_app(hf_auth.router)

    resp = client.get("/hf-auth/status")

    assert resp.status_code == 200
    assert resp.json() == payload


def test_relay_status_lite_early_return(router_app):
    """Non-wireless daemon short-circuits to the 'coming soon' payload."""
    client = router_app(hf_auth.router, daemon=_daemon(wireless_version=False))

    resp = client.get("/hf-auth/relay-status")

    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "unavailable"
    assert body["is_connected"] is False


def test_relay_status_wireless(monkeypatch, router_app):
    """Wireless daemon delegates to central_signaling_relay.get_relay_status."""
    from reachy_mini.media import central_signaling_relay

    monkeypatch.setattr(
        central_signaling_relay,
        "get_relay_status",
        lambda: {"state": "connected", "is_connected": True},
    )
    client = router_app(hf_auth.router, daemon=_daemon(wireless_version=True))

    resp = client.get("/hf-auth/relay-status")

    assert resp.status_code == 200
    assert resp.json() == {"state": "connected", "is_connected": True}


def test_delete_token_success(monkeypatch, router_app):
    """Delete succeeds -> 200 success."""
    monkeypatch.setattr(src, "delete_hf_token", lambda: True)
    client = router_app(hf_auth.router)

    resp = client.delete("/hf-auth/token")

    assert resp.status_code == 200
    assert resp.json() == {"status": "success"}


def test_delete_token_failure(monkeypatch, router_app):
    """Delete fails -> 500."""
    monkeypatch.setattr(src, "delete_hf_token", lambda: False)
    client = router_app(hf_auth.router)

    resp = client.delete("/hf-auth/token")

    assert resp.status_code == 500


def test_oauth_configured(monkeypatch, router_app):
    """/oauth/configured wraps is_oauth_configured in a dict."""
    monkeypatch.setattr(src, "is_oauth_configured", lambda: True)
    client = router_app(hf_auth.router)

    resp = client.get("/hf-auth/oauth/configured")

    assert resp.status_code == 200
    assert resp.json() == {"configured": True}


def test_oauth_start_success(monkeypatch, router_app):
    """/oauth/start returns the created session on success."""
    result = {"status": "success", "session_id": "s1", "auth_url": "https://hf/authz"}
    monkeypatch.setattr(src, "create_oauth_session", lambda **kw: result)
    client = router_app(hf_auth.router, daemon=_daemon(wireless_version=True))

    resp = client.get("/hf-auth/oauth/start")

    assert resp.status_code == 200
    assert resp.json() == result


def test_oauth_start_error(monkeypatch, router_app):
    """/oauth/start surfaces a session-creation error as 500."""
    monkeypatch.setattr(
        src,
        "create_oauth_session",
        lambda **kw: {"status": "error", "message": "no config"},
    )
    client = router_app(hf_auth.router, daemon=_daemon(wireless_version=False))

    resp = client.get("/hf-auth/oauth/start")

    assert resp.status_code == 500
    assert resp.json()["detail"] == "no config"


def test_oauth_begin_redirects(monkeypatch, router_app):
    """/oauth/begin 307-redirects to the session auth_url."""
    monkeypatch.setattr(
        src,
        "create_oauth_session",
        lambda **kw: {"status": "success", "auth_url": "https://hf/authz"},
    )
    client = router_app(hf_auth.router, daemon=_daemon(wireless_version=True))

    resp = client.get("/hf-auth/oauth/begin", follow_redirects=False)

    assert resp.status_code == 307
    assert resp.headers["location"] == "https://hf/authz"


def test_oauth_status(monkeypatch, router_app):
    """/oauth/status/{id} passes the id through to get_oauth_session_status."""
    monkeypatch.setattr(
        src,
        "get_oauth_session_status",
        lambda sid: {"session_id": sid, "status": "pending"},
    )
    client = router_app(hf_auth.router)

    resp = client.get("/hf-auth/oauth/status/abc")

    assert resp.status_code == 200
    assert resp.json() == {"session_id": "abc", "status": "pending"}


def test_oauth_session_delete_success(monkeypatch, router_app):
    """Cancelling a known session -> 200 success."""
    monkeypatch.setattr(src, "cancel_oauth_session", lambda sid: True)
    client = router_app(hf_auth.router)

    resp = client.delete("/hf-auth/oauth/session/abc")

    assert resp.status_code == 200
    assert resp.json() == {"status": "success"}


def test_oauth_session_delete_not_found(monkeypatch, router_app):
    """Cancelling an unknown session -> 404."""
    monkeypatch.setattr(src, "cancel_oauth_session", lambda sid: False)
    client = router_app(hf_auth.router)

    resp = client.delete("/hf-auth/oauth/session/missing")

    assert resp.status_code == 404


def test_central_robot_status_no_token(monkeypatch, router_app):
    """No stored token -> early return, no aiohttp/network involved."""
    monkeypatch.setattr(src, "get_hf_token", lambda: None)
    client = router_app(hf_auth.router)

    resp = client.get("/hf-auth/central-robot-status")

    assert resp.status_code == 200
    assert resp.json() == {
        "available": False,
        "robots": [],
        "reason": "not_authenticated",
    }


def test_oauth_callback_error_param(monkeypatch, router_app):
    """Callback with an OAuth error renders the failure page (no network)."""
    monkeypatch.setattr(src, "get_session_by_state", lambda state: None)
    client = router_app(hf_auth.router)

    resp = client.get(
        "/hf-auth/oauth/callback",
        params={"error": "access_denied", "error_description": "user said no"},
    )

    assert resp.status_code == 200
    assert "user said no" in resp.text


def test_oauth_callback_missing_code(router_app):
    """Callback without code/state -> 400 failure page (no network)."""
    client = router_app(hf_auth.router)

    resp = client.get("/hf-auth/oauth/callback")

    assert resp.status_code == 400
    assert "Missing authorization code or state" in resp.text
