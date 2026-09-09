"""Offline tests for Hugging Face authentication and OAuth session handling."""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from huggingface_hub.errors import HfHubHTTPError

from reachy_mini.apps.sources import hf_auth
from reachy_mini.media import central_signaling_relay


@pytest.fixture(autouse=True)
def _clear_sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the module-global session dict before every test."""
    monkeypatch.setattr(hf_auth, "_oauth_sessions", {})


def test_generate_user_code_format() -> None:
    """User code is 4 letters, dash, 4 digits, no ambiguous letters."""
    code = hf_auth._generate_user_code()
    letters, numbers = code.split("-")
    assert len(letters) == 4 and letters.isalpha()
    assert len(numbers) == 4 and numbers.isdigit()
    assert not set(letters) & set("IO")


def test_generate_pkce_pair_distinct_urlsafe() -> None:
    """PKCE pair is two distinct URL-safe strings (no padding on challenge)."""
    verifier, challenge = hf_auth._generate_pkce_pair()
    assert verifier != challenge
    assert len(verifier) >= 43
    assert not challenge.endswith("=")


def test_get_oauth_redirect_uri_variants() -> None:
    """Redirect URI honours wireless flag and localhost override."""
    assert hf_auth.get_oauth_redirect_uri(True) == hf_auth.OAUTH_REDIRECT_URI_WIRELESS
    assert hf_auth.get_oauth_redirect_uri(False) == hf_auth.OAUTH_REDIRECT_URI_LITE
    assert (
        hf_auth.get_oauth_redirect_uri(True, use_localhost=True)
        == hf_auth.OAUTH_REDIRECT_URI_LITE
    )


def test_is_oauth_configured_toggle(monkeypatch: pytest.MonkeyPatch) -> None:
    """is_oauth_configured reflects the client-id constant."""
    monkeypatch.setattr(hf_auth, "OAUTH_CLIENT_ID", "some-id")
    assert hf_auth.is_oauth_configured() is True
    monkeypatch.setattr(hf_auth, "OAUTH_CLIENT_ID", "")
    assert hf_auth.is_oauth_configured() is False


def test_configure_oauth_sets_globals(monkeypatch: pytest.MonkeyPatch) -> None:
    """configure_oauth overwrites the module OAuth globals."""
    # Guard originals so mutation of module state doesn't leak.
    monkeypatch.setattr(hf_auth, "OAUTH_CLIENT_ID", None)
    monkeypatch.setattr(hf_auth, "OAUTH_CLIENT_SECRET", None)
    monkeypatch.setattr(hf_auth, "OAUTH_SCOPES", "")
    hf_auth.configure_oauth("cid", client_secret="secret", scopes="openid")
    assert hf_auth.OAUTH_CLIENT_ID == "cid"
    assert hf_auth.OAUTH_CLIENT_SECRET == "secret"
    assert hf_auth.OAUTH_SCOPES == "openid"


def test_create_oauth_session_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """A configured session yields an auth URL and registers the session."""
    monkeypatch.setattr(hf_auth, "OAUTH_CLIENT_ID", "cid")
    result = hf_auth.create_oauth_session(wireless_version=True)
    assert result["status"] == "success"
    assert result["auth_url"].startswith("https://huggingface.co/oauth/authorize?")
    assert result["redirect_uri"] == hf_auth.OAUTH_REDIRECT_URI_WIRELESS
    assert result["expires_in"] == 600
    assert result["session_id"] in hf_auth._oauth_sessions


def test_create_oauth_session_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """No client id short-circuits with an error and stores nothing."""
    monkeypatch.setattr(hf_auth, "OAUTH_CLIENT_ID", "")
    result = hf_auth.create_oauth_session(wireless_version=False)
    assert result["status"] == "error"
    assert "OAuth not configured" in result["message"]
    assert hf_auth._oauth_sessions == {}


def test_get_oauth_session_and_by_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sessions are retrievable by id and by state."""
    monkeypatch.setattr(hf_auth, "OAUTH_CLIENT_ID", "cid")
    sid = hf_auth.create_oauth_session(wireless_version=True)["session_id"]
    session = hf_auth.get_oauth_session(sid)
    assert session is not None
    assert hf_auth.get_session_by_state(session.state) is session
    assert hf_auth.get_oauth_session("nope") is None
    assert hf_auth.get_session_by_state("nope") is None


def test_get_oauth_session_status_states(monkeypatch: pytest.MonkeyPatch) -> None:
    """Status polling surfaces pending, authorized (+username) and error."""
    monkeypatch.setattr(hf_auth, "OAUTH_CLIENT_ID", "cid")
    sid = hf_auth.create_oauth_session(wireless_version=True)["session_id"]

    assert hf_auth.get_oauth_session_status(sid) == {"status": "pending"}

    session = hf_auth.get_oauth_session(sid)
    assert session is not None
    session.status = "authorized"
    session.username = "alice"
    assert hf_auth.get_oauth_session_status(sid) == {
        "status": "authorized",
        "username": "alice",
    }

    session.status = "error"
    session.error_message = "boom"
    assert hf_auth.get_oauth_session_status(sid) == {
        "status": "error",
        "message": "boom",
    }

    assert hf_auth.get_oauth_session_status("missing")["status"] == "expired"


def test_cancel_oauth_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancelling removes the session; a second cancel returns False."""
    monkeypatch.setattr(hf_auth, "OAUTH_CLIENT_ID", "cid")
    sid = hf_auth.create_oauth_session(wireless_version=True)["session_id"]
    assert hf_auth.cancel_oauth_session(sid) is True
    assert sid not in hf_auth._oauth_sessions
    assert hf_auth.cancel_oauth_session(sid) is False


def test_cleanup_expired_sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Expired sessions are pruned; live ones remain."""
    monkeypatch.setattr(hf_auth, "OAUTH_CLIENT_ID", "cid")
    live = hf_auth.create_oauth_session(wireless_version=True)["session_id"]
    stale = hf_auth.create_oauth_session(wireless_version=True)["session_id"]
    hf_auth._oauth_sessions[stale].expires_at = time.time() - 1

    hf_auth._cleanup_expired_sessions()
    assert live in hf_auth._oauth_sessions
    assert stale not in hf_auth._oauth_sessions


def test_expired_session_not_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    """Getters trigger cleanup, so an expired session reads as gone."""
    monkeypatch.setattr(hf_auth, "OAUTH_CLIENT_ID", "cid")
    sid = hf_auth.create_oauth_session(wireless_version=True)["session_id"]
    hf_auth._oauth_sessions[sid].expires_at = time.time() - 1
    assert hf_auth.get_oauth_session(sid) is None
    assert hf_auth.get_oauth_session_status(sid)["status"] == "expired"


@pytest.mark.asyncio
async def test_exchange_code_invalid_session() -> None:
    """Unknown state returns an invalid-session error before any network."""
    result = await hf_auth.exchange_code_for_token("code", "unknown-state", True)
    assert result["status"] == "error"
    assert "Invalid or expired session" in result["message"]


@pytest.mark.asyncio
async def test_exchange_code_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """A valid session but missing client id fails as not configured."""
    monkeypatch.setattr(hf_auth, "OAUTH_CLIENT_ID", "cid")
    sid = hf_auth.create_oauth_session(wireless_version=True)["session_id"]
    session = hf_auth.get_oauth_session(sid)
    assert session is not None

    monkeypatch.setattr(hf_auth, "OAUTH_CLIENT_ID", "")
    result = await hf_auth.exchange_code_for_token("code", session.state, True)
    assert result == {"status": "error", "message": "OAuth not configured"}
    assert session.status == "error"


@pytest.mark.parametrize(
    ("status", "body", "message"),
    [
        (400, "provider-secret-marker", hf_auth.AUTHENTICATION_FAILED_MESSAGE),
        (
            200,
            '{"error": "provider-secret-marker"}',
            hf_auth.AUTHENTICATION_FAILED_MESSAGE,
        ),
        (200, "provider-secret-marker", hf_auth.AUTHENTICATION_UNAVAILABLE_MESSAGE),
        (200, '["provider-secret-marker"]', hf_auth.AUTHENTICATION_UNAVAILABLE_MESSAGE),
    ],
)
@pytest.mark.asyncio
async def test_exchange_code_failure_never_surfaces_provider_text(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    status: int,
    body: str,
    message: str,
) -> None:
    """A failed exchange reports a stable message, not the provider response."""
    response = MagicMock(status=status)
    response.json = AsyncMock(side_effect=lambda **kwargs: json.loads(body))
    client = MagicMock()
    http_session = client.return_value.__aenter__.return_value
    http_session.post = MagicMock()
    http_session.post.return_value.__aenter__.return_value = response
    monkeypatch.setattr(hf_auth, "OAUTH_CLIENT_ID", "cid")
    monkeypatch.setattr(hf_auth.aiohttp, "ClientSession", client)
    sid = hf_auth.create_oauth_session(wireless_version=True)["session_id"]
    session = hf_auth.get_oauth_session(sid)
    assert session is not None

    result = await hf_auth.exchange_code_for_token("code", session.state, True)

    assert result == {
        "status": "error",
        "message": message,
    }
    assert session.error_message == message
    assert (
        "provider-secret-marker" not in hf_auth.get_oauth_session_status(sid)["message"]
    )
    assert "provider-secret-marker" not in caplog.text
    if status != 200:
        response.json.assert_not_awaited()
    else:
        response.json.assert_awaited_once_with(content_type=None)


@pytest.mark.asyncio
async def test_save_token_handles_background_relay_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A failed notification leaves login successful without an unhandled task."""
    api = MagicMock()
    api.whoami.return_value = {"name": "alice"}
    monkeypatch.setattr(hf_auth, "HfApi", MagicMock(return_value=api))
    login = MagicMock()
    monkeypatch.setattr(hf_auth, "login", login)
    notify = AsyncMock(side_effect=RuntimeError("provider-secret-marker"))
    monkeypatch.setattr(central_signaling_relay, "notify_token_change", notify)
    loop = asyncio.get_running_loop()
    unhandled = MagicMock()
    original_handler = loop.get_exception_handler()
    loop.set_exception_handler(unhandled)
    try:
        result = hf_auth.save_hf_token("hf_test")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(original_handler)

    assert result == {"status": "success", "username": "alice"}
    login.assert_called_once_with(token="hf_test", add_to_git_credential=False)
    notify.assert_awaited_once_with("hf_test")
    unhandled.assert_not_called()
    assert "RuntimeError" in caplog.text
    assert "provider-secret-marker" not in caplog.text


@pytest.mark.parametrize("relay_fails", [False, True])
def test_save_hf_token_success(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    relay_fails: bool,
) -> None:
    """Valid token validates, persists via login, and returns the username."""
    api = MagicMock()
    api.whoami.return_value = {"name": "alice"}
    hf_api = MagicMock(return_value=api)
    login = MagicMock()
    monkeypatch.setattr(hf_auth, "HfApi", hf_api)
    monkeypatch.setattr(hf_auth, "login", login)
    notify = AsyncMock(
        side_effect=RuntimeError("provider-secret-marker") if relay_fails else None
    )
    monkeypatch.setattr(central_signaling_relay, "notify_token_change", notify)

    result = hf_auth.save_hf_token("tok")
    assert result == {"status": "success", "username": "alice"}
    hf_api.assert_called_once_with(token="tok")
    login.assert_called_once_with(token="tok", add_to_git_credential=False)
    notify.assert_awaited_once_with("tok")
    assert "provider-secret-marker" not in caplog.text
    if relay_fails:
        assert "RuntimeError" in caplog.text


def test_save_hf_token_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    """HfHubHTTPError/ValueError maps to the generic invalid-token message."""
    api = MagicMock()
    api.whoami.side_effect = ValueError("bad")
    monkeypatch.setattr(hf_auth, "HfApi", MagicMock(return_value=api))
    monkeypatch.setattr(hf_auth, "login", MagicMock())
    monkeypatch.setattr(hf_auth, "_notify_relay_of_token_change", lambda *a: None)

    result = hf_auth.save_hf_token("tok")
    assert result == {"status": "error", "message": "Invalid token or network error"}


def test_save_hf_token_unexpected_error_is_redacted(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Unexpected provider text stays out of the response and log."""
    api = MagicMock()
    api.whoami.side_effect = RuntimeError("provider-secret-marker")
    monkeypatch.setattr(hf_auth, "HfApi", MagicMock(return_value=api))
    monkeypatch.setattr(hf_auth, "login", MagicMock())
    monkeypatch.setattr(hf_auth, "_notify_relay_of_token_change", lambda *a: None)

    with caplog.at_level("WARNING", logger=hf_auth.__name__):
        result = hf_auth.save_hf_token("tok")

    assert result == {
        "status": "error",
        "message": hf_auth.CREDENTIAL_SAVE_FAILED_MESSAGE,
    }
    assert "provider-secret-marker" not in caplog.text
    assert "RuntimeError" in caplog.text


def test_save_hf_token_hfhub_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """HfHubHTTPError from login maps to the invalid-token message."""
    api = MagicMock()
    api.whoami.return_value = {"name": "alice"}
    login = MagicMock(side_effect=HfHubHTTPError("nope", response=MagicMock()))
    monkeypatch.setattr(hf_auth, "HfApi", MagicMock(return_value=api))
    monkeypatch.setattr(hf_auth, "login", login)
    monkeypatch.setattr(hf_auth, "_notify_relay_of_token_change", lambda *a: None)

    result = hf_auth.save_hf_token("tok")
    assert result == {"status": "error", "message": "Invalid token or network error"}


def test_get_hf_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_hf_token delegates to huggingface_hub.get_token."""
    monkeypatch.setattr(hf_auth, "get_token", MagicMock(return_value="tok"))
    assert hf_auth.get_hf_token() == "tok"


def test_delete_hf_token_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Successful logout returns True and notifies the relay of no token."""
    logout = MagicMock()
    notify = MagicMock()
    monkeypatch.setattr(hf_auth, "logout", logout)
    monkeypatch.setattr(hf_auth, "_notify_relay_of_token_change", notify)
    assert hf_auth.delete_hf_token() is True
    logout.assert_called_once_with()
    notify.assert_called_once_with(None)


def test_delete_hf_token_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A logout error is swallowed and returns False."""
    monkeypatch.setattr(hf_auth, "logout", MagicMock(side_effect=RuntimeError("x")))
    monkeypatch.setattr(hf_auth, "_notify_relay_of_token_change", lambda *a: None)
    assert hf_auth.delete_hf_token() is False


def test_check_token_status_no_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """No stored token means logged out."""
    monkeypatch.setattr(hf_auth, "get_token", MagicMock(return_value=None))
    assert hf_auth.check_token_status() == {"is_logged_in": False, "username": None}


def test_check_token_status_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    """A valid token reports logged in with the username."""
    monkeypatch.setattr(hf_auth, "get_token", MagicMock(return_value="tok"))
    monkeypatch.setattr(hf_auth, "whoami", MagicMock(return_value={"name": "alice"}))
    assert hf_auth.check_token_status() == {"is_logged_in": True, "username": "alice"}


def test_check_token_status_whoami_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A whoami failure downgrades to logged out."""
    monkeypatch.setattr(hf_auth, "get_token", MagicMock(return_value="tok"))
    monkeypatch.setattr(hf_auth, "whoami", MagicMock(side_effect=RuntimeError("x")))
    assert hf_auth.check_token_status() == {"is_logged_in": False, "username": None}
