"""Tests for Hugging Face authentication persistence."""

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
async def test_oauth_token_uses_huggingface_configured_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """OAuth persistence should honor the path selected by huggingface_hub."""
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
    monkeypatch.setattr(hf_auth.aiohttp, "ClientSession", _ClientSession)
    monkeypatch.setattr(hf_auth, "whoami", lambda **_kwargs: {"name": "tester"})
    monkeypatch.setattr(central_signaling_relay, "notify_token_change", AsyncMock())

    try:
        result = await hf_auth.exchange_code_for_token("code", "state", False)
    finally:
        hf_auth._oauth_sessions.clear()

    assert result == {"status": "success", "username": "tester"}
    assert token_path.read_text() == "oauth-token"
    assert token_path.stat().st_mode & 0o777 == 0o600
