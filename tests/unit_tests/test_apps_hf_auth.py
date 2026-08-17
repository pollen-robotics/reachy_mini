"""Tests for the HuggingFace auth source module (in-memory, no network).

Covers the OAuth-session lifecycle on the module-global `_oauth_sessions`
dict, the pure helpers, and the token functions with `huggingface_hub`
symbols monkeypatched. The real aiohttp token POST in
`exchange_code_for_token` is not exercised; only its early error branches.
"""

import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from reachy_mini.apps.sources import hf_auth


@pytest.fixture(autouse=True)
def _isolated_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point the credential store at a throwaway directory."""
    monkeypatch.setattr(hf_auth, "HF_TOKEN_PATH", str(tmp_path / "token"))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(hf_auth, "_notify_relay_of_token_change", lambda *_a: None)


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


# ---- Token functions (against the daemon's own store)


def _validating_api(monkeypatch: pytest.MonkeyPatch, name: str = "alice") -> MagicMock:
    api = MagicMock()
    api.whoami.return_value = {"name": name}
    monkeypatch.setattr(hf_auth, "HfApi", MagicMock(return_value=api))
    return api


def test_save_then_read_round_trips_through_the_owned_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A saved token is validated, stored, and read back without huggingface_hub."""
    _validating_api(monkeypatch)

    assert hf_auth.save_hf_token("tok") == {"status": "success", "username": "alice"}
    assert hf_auth.get_hf_token() == "tok"
    assert hf_auth.current_lifecycle_generation() == 1


def test_save_rejects_an_invalid_token_without_storing_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A token the Hub rejects is never written to the store."""
    api = MagicMock()
    api.whoami.side_effect = ValueError("bad")
    monkeypatch.setattr(hf_auth, "HfApi", MagicMock(return_value=api))

    assert hf_auth.save_hf_token("tok") == {
        "status": "error",
        "message": "Invalid token or network error",
    }
    assert hf_auth.get_hf_token() is None


def test_save_reports_a_write_failure_without_leaking_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A store write failure returns a stable message rather than raising."""
    _validating_api(monkeypatch)
    monkeypatch.setattr(
        hf_auth, "_write_store", MagicMock(side_effect=OSError("secret-marker"))
    )

    result = hf_auth.save_hf_token("tok")

    assert result == {
        "status": "error",
        "message": hf_auth.CREDENTIAL_SAVE_FAILED_MESSAGE,
    }
    assert "secret-marker" not in result["message"]


def test_sign_out_leaves_credentials_the_daemon_does_not_own(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Signing out tombstones our record and never touches the CLI token file."""
    cli_token = Path(hf_auth.HF_TOKEN_PATH)
    cli_token.parent.mkdir(parents=True, exist_ok=True)
    cli_token.write_text("cli-token", encoding="utf-8")
    _validating_api(monkeypatch)
    assert hf_auth.save_hf_token("tok")["status"] == "success"

    assert hf_auth.delete_hf_token() is True

    assert hf_auth.get_hf_token() is None
    assert cli_token.read_text(encoding="utf-8") == "cli-token"
    assert hf_auth.current_lifecycle_generation() == 2


def test_sign_out_reports_a_write_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A store write failure during sign-out is reported, not raised."""
    monkeypatch.setattr(hf_auth, "_write_store", MagicMock(side_effect=OSError("nope")))
    assert hf_auth.delete_hf_token() is False


def test_a_login_completing_after_sign_out_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A login that lost its race with sign-out must not authorize the robot."""
    generation = hf_auth.current_lifecycle_generation()
    assert hf_auth.delete_hf_token() is True

    assert hf_auth._persist_login({"access_token": "late-token"}, generation) is False
    assert hf_auth.get_hf_token() is None


@pytest.mark.parametrize(
    ("stored", "whoami_result", "expected"),
    [
        (None, None, {"is_logged_in": False, "username": None}),
        ("tok", {"name": "alice"}, {"is_logged_in": True, "username": "alice"}),
        ("tok", RuntimeError("x"), {"is_logged_in": False, "username": None}),
    ],
)
def test_check_token_status_reflects_the_store(
    monkeypatch: pytest.MonkeyPatch,
    stored: str | None,
    whoami_result: object,
    expected: dict[str, object],
) -> None:
    """Status is derived from our record plus a Hub identity check."""
    if stored:
        _validating_api(monkeypatch)
        assert hf_auth.save_hf_token(stored)["status"] == "success"
    if isinstance(whoami_result, Exception):
        monkeypatch.setattr(hf_auth, "whoami", MagicMock(side_effect=whoami_result))
    else:
        monkeypatch.setattr(hf_auth, "whoami", MagicMock(return_value=whoami_result))

    assert hf_auth.check_token_status() == expected


def test_an_expiring_token_is_refreshed_and_rotated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A token close to expiry is exchanged, and the rotated refresh token sticks."""
    _validating_api(monkeypatch)
    assert hf_auth.save_hf_token("first")["status"] == "success"
    hf_auth._write_store(
        replace(
            hf_auth._read_store(),
            refresh_token="r1",
            expires_at=int(time.time()) + 5,
        )
    )
    exchanged: list[str] = []

    def _refresh(refresh_token: str) -> dict[str, object]:
        exchanged.append(refresh_token)
        return {"access_token": "second", "refresh_token": "r2", "expires_in": 3600}

    monkeypatch.setattr(
        "huggingface_hub.utils._oauth_device.refresh_access_token", _refresh
    )

    assert hf_auth.get_hf_token() == "second"
    assert exchanged == ["r1"]
    assert hf_auth._read_store().refresh_token == "r2"


def test_a_failed_refresh_does_not_hand_out_an_expired_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the exchange fails an already-expired token is withheld, not returned."""
    _validating_api(monkeypatch)
    assert hf_auth.save_hf_token("first")["status"] == "success"
    hf_auth._write_store(
        replace(
            hf_auth._read_store(),
            refresh_token="r1",
            expires_at=int(time.time()) - 1,
        )
    )

    def _boom(_refresh_token: str) -> dict[str, object]:
        raise RuntimeError("secret-marker")

    monkeypatch.setattr(
        "huggingface_hub.utils._oauth_device.refresh_access_token", _boom
    )

    assert hf_auth.get_hf_token() is None
