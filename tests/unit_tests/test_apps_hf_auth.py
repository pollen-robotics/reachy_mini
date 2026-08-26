"""Tests for Hugging Face authentication."""

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


def test_is_oauth_configured_toggle(monkeypatch: pytest.MonkeyPatch) -> None:
    """is_oauth_configured reflects the client-id constant."""
    monkeypatch.setattr(hf_auth, "OAUTH_CLIENT_ID", "some-id")
    assert hf_auth.is_oauth_configured() is True
    monkeypatch.setattr(hf_auth, "OAUTH_CLIENT_ID", "")
    assert hf_auth.is_oauth_configured() is False


@pytest.mark.parametrize(
    ("wireless_version", "use_localhost", "expected_redirect"),
    [
        (False, False, hf_auth.OAUTH_REDIRECT_URI_LITE),
        (True, False, hf_auth.OAUTH_REDIRECT_URI_WIRELESS),
        (True, True, hf_auth.OAUTH_REDIRECT_URI_LITE),
    ],
)
def test_create_oauth_session_selects_redirect(
    monkeypatch: pytest.MonkeyPatch,
    wireless_version: bool,
    use_localhost: bool,
    expected_redirect: str,
) -> None:
    """Lite, Wireless, and desktop-proxied Wireless use the registered URI."""
    monkeypatch.setattr(hf_auth, "OAUTH_CLIENT_ID", "cid")
    result = hf_auth.create_oauth_session(wireless_version, use_localhost)
    assert result["status"] == "success"
    assert result["auth_url"].startswith("https://huggingface.co/oauth/authorize?")
    assert result["redirect_uri"] == expected_redirect
    assert result["expires_in"] == 600
    assert hf_auth._oauth_sessions[result["session_id"]].redirect_uri == expected_redirect


def test_create_oauth_session_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """No client id short-circuits with an error and stores nothing."""
    monkeypatch.setattr(hf_auth, "OAUTH_CLIENT_ID", "")
    result = hf_auth.create_oauth_session(wireless_version=False)
    assert result["status"] == "error"
    assert "OAuth not configured" in result["message"]
    assert hf_auth._oauth_sessions == {}


def test_get_oauth_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sessions are retrievable by their state-backed ID."""
    monkeypatch.setattr(hf_auth, "OAUTH_CLIENT_ID", "cid")
    sid = hf_auth.create_oauth_session(wireless_version=True)["session_id"]
    assert hf_auth.get_oauth_session(sid) is not None
    assert hf_auth.get_oauth_session("nope") is None


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
    result = await hf_auth.exchange_code_for_token("code", "unknown-state")
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
    result = await hf_auth.exchange_code_for_token("code", session.session_id)
    assert result == {"status": "error", "message": "OAuth not configured"}
    assert session.status == "error"


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
    assert hf_auth.get_hf_credential().lifecycle_generation == 1


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
    assert hf_auth.get_hf_credential().lifecycle_generation == 2


def test_sign_out_reports_a_write_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A store write failure during sign-out is reported, not raised."""
    monkeypatch.setattr(hf_auth, "_write_store", MagicMock(side_effect=OSError("nope")))
    assert hf_auth.delete_hf_token() is False


def test_manual_login_completing_after_sign_out_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A login that lost its race with sign-out must not authorize the robot."""
    api = _validating_api(monkeypatch)

    def _sign_out_during_validation() -> dict[str, str]:
        assert hf_auth.delete_hf_token() is True
        return {"name": "alice"}

    api.whoami.side_effect = _sign_out_during_validation

    assert hf_auth.save_hf_token("late-token") == {
        "status": "error",
        "message": hf_auth.AUTHENTICATION_FAILED_MESSAGE,
    }
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
