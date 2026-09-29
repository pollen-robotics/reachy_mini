"""Tests for Hugging Face authentication persistence."""

import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from reachy_mini.apps.sources import hf_auth
from reachy_mini.media import central_signaling_relay


class _TokenResponse:
    status = 200

    async def __aenter__(self) -> "_TokenResponse":
        return self

    async def __aexit__(self, *_args: object) -> None:
        pass

    async def json(self, **_kwargs: object) -> dict[str, str]:
        return {"access_token": "oauth-token"}


class _ClientSession:
    """Stub aiohttp session recording constructor and request kwargs."""

    last_kwargs: dict[str, object] = {}
    last_post_kwargs: dict[str, object] = {}

    def __init__(self, **kwargs: object) -> None:
        _ClientSession.last_kwargs = kwargs

    async def __aenter__(self) -> "_ClientSession":
        return self

    async def __aexit__(self, *_args: object) -> None:
        pass

    def post(self, _url: str, **_kwargs: object) -> _TokenResponse:
        _ClientSession.last_post_kwargs = _kwargs
        return _TokenResponse()


@pytest.mark.asyncio
async def test_oauth_token_is_stored_in_the_daemon_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Redirect OAuth persists to the daemon's own store, not the shared token file."""
    token_path = tmp_path / "private-hf-home" / "token"
    session = hf_auth.OAuthSession(
        session_id="state",
        code_verifier="verifier",
        redirect_uri=hf_auth.OAUTH_REDIRECT_URI_LITE,
    )
    hf_auth._oauth_sessions[session.session_id] = session
    monkeypatch.setattr(hf_auth, "HF_TOKEN_PATH", str(token_path))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(hf_auth.aiohttp, "ClientSession", _ClientSession)
    monkeypatch.setattr(hf_auth, "whoami", lambda **_kwargs: {"name": "tester"})
    monkeypatch.setattr(central_signaling_relay, "notify_token_change", AsyncMock())

    try:
        result = await hf_auth.exchange_code_for_token("code", "state")
    finally:
        hf_auth._oauth_sessions.clear()

    assert result == {"status": "success", "username": "tester"}
    # No trust_env (it would read ~/.netrc); the proxy is passed per request.
    assert "trust_env" not in _ClientSession.last_kwargs
    assert "proxy" in _ClientSession.last_post_kwargs
    assert hf_auth.get_hf_token() == "oauth-token"
    assert not token_path.exists()
    if os.name != "nt":
        assert hf_auth._store_path().stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_cancelling_redirect_oauth_while_exchanging_refuses_the_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A redirect login cancelled mid-exchange never stores the late token."""

    async def cancel_then_return_token(
        _response: _TokenResponse, **_kwargs: object
    ) -> dict[str, str]:
        assert hf_auth.cancel_oauth_session("state") is True
        return {"access_token": "late-token"}

    session = hf_auth.OAuthSession(
        session_id="state",
        code_verifier="verifier",
        redirect_uri=hf_auth.OAUTH_REDIRECT_URI_LITE,
    )
    hf_auth._oauth_sessions[session.session_id] = session
    monkeypatch.setattr(_TokenResponse, "json", cancel_then_return_token)
    monkeypatch.setattr(hf_auth.aiohttp, "ClientSession", _ClientSession)
    monkeypatch.setattr(hf_auth, "whoami", lambda **_kwargs: {"name": "tester"})
    relay_notify = AsyncMock()
    monkeypatch.setattr(central_signaling_relay, "notify_token_change", relay_notify)

    try:
        result = await hf_auth.exchange_code_for_token("code", "state")
    finally:
        hf_auth._oauth_sessions.clear()

    assert result == {
        "status": "error",
        "message": hf_auth._CANCELLED_SESSION_MESSAGE,
    }
    assert hf_auth.get_hf_token() is None
    relay_notify.assert_not_awaited()


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
        result = await hf_auth.exchange_code_for_token("code", start["session_id"])
    finally:
        hf_auth._oauth_sessions.clear()

    assert result == {"status": "success", "username": "tester"}
    assert hf_auth.get_hf_token() == "oauth-token"
    assert token_path.read_text(encoding="utf-8") == "the-users-own-token"


@pytest.mark.asyncio
async def test_save_token_keeps_relay_task_until_finished(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """Retain relay work until it completes."""
    monkeypatch.setattr(hf_auth, "HF_TOKEN_PATH", str(tmp_path / "token"))
    api = MagicMock()
    api.whoami.return_value = {"name": "alice"}
    monkeypatch.setattr(hf_auth, "HfApi", MagicMock(return_value=api))
    started = asyncio.Event()
    finish = asyncio.Event()

    async def notify(_token: str) -> None:
        started.set()
        await finish.wait()
        raise RuntimeError("provider-secret-marker")

    monkeypatch.setattr(central_signaling_relay, "notify_token_change", notify)

    assert hf_auth.save_hf_token("tok")["status"] == "success"
    await started.wait()
    assert len(hf_auth._relay_tasks) == 1
    finish.set()
    await asyncio.gather(*tuple(hf_auth._relay_tasks))
    await asyncio.sleep(0)

    assert not hf_auth._relay_tasks
    assert "RuntimeError" in caplog.text
    assert "provider-secret-marker" not in caplog.text
