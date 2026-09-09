"""Tests for Hugging Face authentication persistence."""

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from reachy_mini.apps.sources import hf_auth
from reachy_mini.media import central_signaling_relay


@pytest.fixture
def token_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Stub the token provider while keeping exchange and persistence real."""
    client = MagicMock()
    session = client.return_value.__aenter__.return_value
    session.post = MagicMock()
    response = session.post.return_value.__aenter__.return_value
    response.status = 200
    response.json = AsyncMock(return_value={"access_token": "oauth-token"})
    monkeypatch.setattr(hf_auth.aiohttp, "ClientSession", client)
    monkeypatch.setattr(hf_auth, "_oauth_sessions", {})
    monkeypatch.setattr(hf_auth, "OAUTH_CLIENT_ID", "cid")
    return client


@pytest.mark.asyncio
@pytest.mark.parametrize("relay_fails", [False, True])
async def test_oauth_token_uses_huggingface_configured_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    token_client: MagicMock,
    relay_fails: bool,
) -> None:
    """OAuth persistence should honor the path selected by huggingface_hub."""
    token_path = tmp_path / "private-hf-home" / "token"
    sid = hf_auth.create_oauth_session(wireless_version=False)["session_id"]
    session = hf_auth.get_oauth_session(sid)
    assert session is not None
    monkeypatch.setattr(hf_auth, "HF_TOKEN_PATH", str(token_path))
    monkeypatch.setattr(hf_auth, "whoami", lambda **_kwargs: {"name": "tester"})
    notify = AsyncMock(
        side_effect=RuntimeError("provider-secret-marker") if relay_fails else None
    )
    monkeypatch.setattr(central_signaling_relay, "notify_token_change", notify)
    real_replace = os.replace
    replacement_observation: dict[str, int] = {}

    def observe_secure_replace(source: str | Path, destination: str | Path) -> None:
        replacement_observation["mode"] = Path(source).stat().st_mode & 0o777
        real_replace(source, destination)

    monkeypatch.setattr(hf_auth.os, "replace", observe_secure_replace)

    result = await hf_auth.exchange_code_for_token("code", session.state, False)

    assert result == {"status": "success", "username": "tester"}
    # No trust_env (it would read ~/.netrc); the proxy is passed per request.
    assert "trust_env" not in token_client.call_args.kwargs
    assert (
        "proxy"
        in token_client.return_value.__aenter__.return_value.post.call_args.kwargs
    )
    assert replacement_observation == {"mode": 0o600}
    assert token_path.read_text() == "oauth-token"
    assert token_path.stat().st_mode & 0o777 == 0o600
    notify.assert_awaited_once_with("oauth-token")
    assert hf_auth.get_oauth_session_status(sid)["status"] == "authorized"
    assert "provider-secret-marker" not in caplog.text
    if relay_fails:
        assert "RuntimeError" in caplog.text


@pytest.mark.asyncio
async def test_oauth_token_is_not_retained_when_permissions_cannot_be_enforced(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    token_client: MagicMock,
) -> None:
    """Permission failure must happen before token bytes reach persistent storage."""
    token_path = tmp_path / "private-hf-home" / "token"
    sid = hf_auth.create_oauth_session(wireless_version=False)["session_id"]
    session = hf_auth.get_oauth_session(sid)
    assert session is not None
    monkeypatch.setattr(hf_auth, "HF_TOKEN_PATH", str(token_path))

    def deny_chmod(_path: Path, _mode: int) -> None:
        raise PermissionError("provider-secret-marker")

    monkeypatch.setattr(Path, "chmod", deny_chmod)

    result = await hf_auth.exchange_code_for_token("code", session.state, False)

    assert result["status"] == "error"
    assert result["message"] == hf_auth.CREDENTIAL_SAVE_FAILED_MESSAGE
    assert not token_path.exists()
    assert list(token_path.parent.iterdir()) == []
    assert hf_auth.get_oauth_session_status(sid)["message"] == result["message"]
    assert "provider-secret-marker" not in caplog.text
    assert "PermissionError" in caplog.text
