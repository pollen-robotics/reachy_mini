"""Tests for the HuggingFace auth source module (in-memory, no network).

Covers the OAuth-session lifecycle on the module-global `_oauth_sessions`
dict, the pure helpers, and the token functions with `huggingface_hub`
symbols monkeypatched. The real aiohttp token POST in
`exchange_code_for_token` is not exercised; only its early error branches.
"""

import time
from unittest.mock import MagicMock

import pytest
from huggingface_hub.errors import HfHubHTTPError

from reachy_mini.apps.sources import hf_auth


@pytest.fixture(autouse=True)
def _clear_sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the module-global session dict before every test."""
    monkeypatch.setattr(hf_auth, "_oauth_sessions", {})


# ---- Pure helpers


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


# ---- OAuth-session lifecycle


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


# ---- exchange_code_for_token early error branches (no aiohttp)


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


# ---- Token functions (huggingface_hub monkeypatched)


def test_save_hf_token_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Valid token validates, persists via login, and returns the username."""
    api = MagicMock()
    api.whoami.return_value = {"name": "alice"}
    hf_api = MagicMock(return_value=api)
    login = MagicMock()
    monkeypatch.setattr(hf_auth, "HfApi", hf_api)
    monkeypatch.setattr(hf_auth, "login", login)
    monkeypatch.setattr(hf_auth, "_notify_relay_of_token_change", lambda *a: None)

    result = hf_auth.save_hf_token("tok")
    assert result == {"status": "success", "username": "alice"}
    hf_api.assert_called_once_with(token="tok")
    login.assert_called_once_with(token="tok", add_to_git_credential=False)


def test_save_hf_token_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    """HfHubHTTPError/ValueError maps to the generic invalid-token message."""
    api = MagicMock()
    api.whoami.side_effect = ValueError("bad")
    monkeypatch.setattr(hf_auth, "HfApi", MagicMock(return_value=api))
    monkeypatch.setattr(hf_auth, "login", MagicMock())
    monkeypatch.setattr(hf_auth, "_notify_relay_of_token_change", lambda *a: None)

    result = hf_auth.save_hf_token("tok")
    assert result == {"status": "error", "message": "Invalid token or network error"}


def test_save_hf_token_unexpected_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Other exceptions surface their string in the error message."""
    api = MagicMock()
    api.whoami.side_effect = RuntimeError("kaboom")
    monkeypatch.setattr(hf_auth, "HfApi", MagicMock(return_value=api))
    monkeypatch.setattr(hf_auth, "login", MagicMock())
    monkeypatch.setattr(hf_auth, "_notify_relay_of_token_change", lambda *a: None)

    result = hf_auth.save_hf_token("tok")
    assert result == {"status": "error", "message": "kaboom"}


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
