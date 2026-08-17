"""Tests for Hugging Face authentication persistence."""

import os
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from reachy_mini.apps.sources import hf_auth
from reachy_mini.media import central_signaling_relay


class _TokenResponse:
    status = 200

    async def __aenter__(self) -> "_TokenResponse":
        return self

    async def __aexit__(self, *_args: object) -> None:
        pass

    async def text(self) -> str:
        return '{"access_token": "oauth-token"}'


class _ClientSession:
    async def __aenter__(self) -> "_ClientSession":
        return self

    async def __aexit__(self, *_args: object) -> None:
        pass

    def post(self, _url: str, **_kwargs: object) -> _TokenResponse:
        return _TokenResponse()


@pytest.mark.asyncio
async def test_oauth_token_is_stored_in_the_daemon_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Redirect OAuth persists to the daemon's own store, not the shared token file."""
    token_path = tmp_path / "private-hf-home" / "token"
    session = hf_auth.OAuthSession(
        session_id="session",
        user_code="",
        state="state",
        code_verifier="verifier",
        wireless_version=False,
    )
    hf_auth._oauth_sessions[session.session_id] = session
    monkeypatch.setattr(hf_auth, "HF_TOKEN_PATH", str(token_path))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(hf_auth.aiohttp, "ClientSession", _ClientSession)
    monkeypatch.setattr(hf_auth, "whoami", lambda **_kwargs: {"name": "tester"})
    monkeypatch.setattr(central_signaling_relay, "notify_token_change", AsyncMock())

    try:
        result = await hf_auth.exchange_code_for_token("code", "state", False)
    finally:
        hf_auth._oauth_sessions.clear()

    assert result == {"status": "success", "username": "tester"}
    assert hf_auth.get_hf_token() == "oauth-token"
    assert not token_path.exists()
    assert hf_auth._store_path().stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_lite_first_run_ignores_credentials_on_the_same_machine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """On Lite the daemon must not adopt the user's own CLI or environment token."""
    token_path = tmp_path / "user-hf-home" / "token"
    token_path.parent.mkdir(parents=True)
    token_path.write_text("the-users-own-token", encoding="utf-8")
    monkeypatch.setattr(hf_auth, "HF_TOKEN_PATH", str(token_path))
    monkeypatch.setenv("HF_TOKEN", "the-users-env-token")
    monkeypatch.setattr(hf_auth.aiohttp, "ClientSession", _ClientSession)
    monkeypatch.setattr(hf_auth, "whoami", lambda **_kwargs: {"name": "tester"})
    monkeypatch.setattr(central_signaling_relay, "notify_token_change", AsyncMock())

    assert hf_auth.get_hf_token() is None

    start = hf_auth.create_oauth_session(wireless_version=False, use_localhost=True)
    try:
        result = await hf_auth.exchange_code_for_token(
            "code", start["session_id"], False
        )
    finally:
        hf_auth._oauth_sessions.clear()

    assert result == {"status": "success", "username": "tester"}
    assert hf_auth.get_hf_token() == "oauth-token"
    assert token_path.read_text(encoding="utf-8") == "the-users-own-token"
