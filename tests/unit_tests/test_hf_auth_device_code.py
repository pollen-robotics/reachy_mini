"""Offline tests for device-code OAuth sessions and failure handling."""

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from huggingface_hub import _login
from huggingface_hub.errors import DeviceCodeError
from huggingface_hub.utils import _oauth_device

from reachy_mini.apps.sources import hf_auth
from reachy_mini.media import central_signaling_relay


@pytest.fixture(autouse=True)
def _clear_sessions() -> Any:
    """Each test starts with an empty session registry."""
    hf_auth._device_code_sessions.clear()
    yield
    hf_auth._device_code_sessions.clear()


_DEVICE_INFO = {
    "device_code": "dev-123",
    "user_code": "ABCD-1234",
    "verification_uri": "https://hf.co/oauth/device",
    "verification_uri_complete": "https://hf.co/oauth/device?user_code=ABCD-1234",
    "interval": 5,
    "expires_in": 900,
}


def test_start_returns_user_code_and_registers_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _oauth_device, "request_device_code", lambda: dict(_DEVICE_INFO)
    )

    # Stub the background poll so the test does not depend on its timing.
    async def _noop_poll(session: Any, device_info: Any) -> None:
        return None

    monkeypatch.setattr(hf_auth, "_run_device_code_poll", _noop_poll)

    async def scenario() -> dict[str, Any]:
        result = await hf_auth.start_device_code_login()
        return result

    result = asyncio.run(scenario())

    assert result["status"] == "pending"
    assert result["user_code"] == "ABCD-1234"
    assert result["verification_uri"] == "https://hf.co/oauth/device"
    assert result["verification_uri_complete"].endswith("user_code=ABCD-1234")
    sid = result["session_id"]
    assert sid in hf_auth._device_code_sessions
    assert hf_auth._device_code_sessions[sid].status == "pending"


def test_start_returns_error_when_request_fails(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def _boom() -> dict[str, Any]:
        raise RuntimeError("provider-secret-marker")

    monkeypatch.setattr(_oauth_device, "request_device_code", _boom)

    result = asyncio.run(hf_auth.start_device_code_login())

    assert result["status"] == "error"
    assert result["message"] == hf_auth.AUTHENTICATION_UNAVAILABLE_MESSAGE
    assert "provider-secret-marker" not in result["message"]
    assert hf_auth._device_code_sessions == {}
    assert "provider-secret-marker" not in caplog.text
    assert "RuntimeError" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "status", "message"),
    [
        (
            DeviceCodeError("provider-secret-marker", error_code="expired_token"),
            "expired",
            hf_auth.LOGIN_EXPIRED_MESSAGE,
        ),
        (
            DeviceCodeError(
                "expired provider-secret-marker", error_code="access_denied"
            ),
            "error",
            hf_auth.AUTHENTICATION_FAILED_MESSAGE,
        ),
        (
            RuntimeError("provider-secret-marker"),
            "error",
            hf_auth.AUTHENTICATION_UNAVAILABLE_MESSAGE,
        ),
    ],
)
async def test_device_login_classifies_errors_without_provider_text(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    error: Exception,
    status: str,
    message: str,
) -> None:
    """Poll results use structured error codes and keep provider text private."""
    monkeypatch.setattr(
        _oauth_device, "request_device_code", lambda: dict(_DEVICE_INFO)
    )
    monkeypatch.setattr(
        _oauth_device, "poll_device_token", MagicMock(side_effect=error)
    )
    caplog.set_level("INFO", logger=hf_auth.__name__)

    login = await hf_auth.start_device_code_login()
    async with asyncio.timeout(2):
        while (result := hf_auth.get_device_code_session_status(login["session_id"]))[
            "status"
        ] == "pending":
            await asyncio.sleep(0.01)

    assert result == {"status": status, "message": message}
    assert type(error).__name__ in caplog.text
    assert "provider-secret-marker" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("save_fails", [False, True])
async def test_device_login_persists_before_notifying_relay(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    save_fails: bool,
) -> None:
    """Persistence errors fail login; relay errors preserve the saved login."""
    token_response = {"access_token": "hf_new_token", "refresh_token": "refresh-xyz"}
    monkeypatch.setattr(
        _oauth_device, "request_device_code", lambda: dict(_DEVICE_INFO)
    )
    monkeypatch.setattr(
        _oauth_device, "poll_device_token", MagicMock(return_value=token_response)
    )
    save = MagicMock(
        return_value=("oauth-alice", "alice"),
        side_effect=OSError("provider-secret-marker") if save_fails else None,
    )
    monkeypatch.setattr(_login, "_save_oauth_token", save)
    notify = AsyncMock(side_effect=RuntimeError("provider-secret-marker"))
    monkeypatch.setattr(central_signaling_relay, "notify_token_change", notify)

    login = await hf_auth.start_device_code_login()
    async with asyncio.timeout(2):
        while (result := hf_auth.get_device_code_session_status(login["session_id"]))[
            "status"
        ] == "pending":
            await asyncio.sleep(0.01)

    save.assert_called_once_with(token_response)
    if save_fails:
        assert result == {
            "status": "error",
            "message": hf_auth.CREDENTIAL_SAVE_FAILED_MESSAGE,
        }
        notify.assert_not_awaited()
        assert "OSError" in caplog.text
    else:
        assert result == {"status": "authorized", "username": "alice"}
        notify.assert_awaited_once_with("hf_new_token")
        assert "RuntimeError" in caplog.text
    assert "provider-secret-marker" not in caplog.text


def test_status_unknown_session_is_expired() -> None:
    assert hf_auth.get_device_code_session_status("nope")["status"] == "expired"


def test_status_authorized_includes_username() -> None:
    session = hf_auth.DeviceCodeSession(
        session_id="s4",
        user_code="ABCD-1234",
        verification_uri="https://hf.co/oauth/device",
        verification_uri_complete="https://hf.co/oauth/device",
        status="authorized",
        username="bob",
    )
    hf_auth._device_code_sessions["s4"] = session

    result = hf_auth.get_device_code_session_status("s4")
    assert result == {"status": "authorized", "username": "bob"}


def test_consume_relay_pending_fires_once() -> None:
    session = hf_auth.DeviceCodeSession(
        session_id="s5",
        user_code="ABCD-1234",
        verification_uri="https://hf.co/oauth/device",
        verification_uri_complete="https://hf.co/oauth/device",
        status="authorized",
    )
    hf_auth._device_code_sessions["s5"] = session

    assert hf_auth.consume_device_session_relay_pending("s5") is True
    assert hf_auth.consume_device_session_relay_pending("s5") is False


def test_consume_relay_pending_false_while_pending() -> None:
    session = hf_auth.DeviceCodeSession(
        session_id="s6",
        user_code="ABCD-1234",
        verification_uri="https://hf.co/oauth/device",
        verification_uri_complete="https://hf.co/oauth/device",
        status="pending",
    )
    hf_auth._device_code_sessions["s6"] = session

    assert hf_auth.consume_device_session_relay_pending("s6") is False


def test_cancel_session_removes_it() -> None:
    session = hf_auth.DeviceCodeSession(
        session_id="s7",
        user_code="ABCD-1234",
        verification_uri="https://hf.co/oauth/device",
        verification_uri_complete="https://hf.co/oauth/device",
    )
    hf_auth._device_code_sessions["s7"] = session

    assert hf_auth.cancel_device_code_session("s7") is True
    assert "s7" not in hf_auth._device_code_sessions
    assert hf_auth.cancel_device_code_session("s7") is False


def test_cancel_signals_the_polling_thread() -> None:
    """Cancel must set the event the polling thread observes, not just drop it."""
    session = hf_auth.DeviceCodeSession(
        session_id="s8",
        user_code="ABCD-1234",
        verification_uri="https://hf.co/oauth/device",
        verification_uri_complete="https://hf.co/oauth/device",
    )
    hf_auth._device_code_sessions["s8"] = session

    assert session.cancel_event.is_set() is False
    assert hf_auth.cancel_device_code_session("s8") is True
    # We keep the local reference, so we can assert the thread would stop.
    assert session.cancel_event.is_set() is True


def test_poll_aborts_when_cancel_event_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cancelled session unwinds poll_device_token via its on_pending hook."""
    calls = {"on_pending": 0}

    def _fake_poll(device_info: Any, *, on_pending: Any = None) -> dict[str, Any]:
        # Mimic the hub: invoke on_pending each pending cycle. With the event
        # already set it raises _DeviceCodeCancelled on the first call, so the
        # (real) blocking loop never runs.
        for _ in range(1000):
            if on_pending is not None:
                calls["on_pending"] += 1
                on_pending()
        raise AssertionError("poll should have been cancelled before returning")

    monkeypatch.setattr(_oauth_device, "poll_device_token", _fake_poll)

    session = hf_auth.DeviceCodeSession(
        session_id="s9",
        user_code="ABCD-1234",
        verification_uri="https://hf.co/oauth/device",
        verification_uri_complete="https://hf.co/oauth/device",
    )
    session.cancel_event.set()

    asyncio.run(hf_auth._run_device_code_poll(session, dict(_DEVICE_INFO)))

    assert calls["on_pending"] == 1
    assert session.status == "cancelled"


def test_authorized_session_gets_bounded_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    """On success the session's expires_at is shortened to the authorized TTL."""
    token_response = {"access_token": "hf_x", "refresh_token": "r", "expires_in": 10}
    monkeypatch.setattr(
        _oauth_device, "poll_device_token", lambda info, **kw: token_response
    )
    monkeypatch.setattr(_login, "_save_oauth_token", lambda resp: ("name", "user"))

    session = hf_auth.DeviceCodeSession(
        session_id="s10",
        user_code="ABCD-1234",
        verification_uri="https://hf.co/oauth/device",
        verification_uri_complete="https://hf.co/oauth/device",
        expires_at=time.time() + 900,  # original device-code expiry
    )

    before = time.time()
    asyncio.run(hf_auth._run_device_code_poll(session, dict(_DEVICE_INFO)))

    assert session.status == "authorized"
    assert session.expires_at <= before + hf_auth._AUTHORIZED_SESSION_TTL_S + 1


def test_cleanup_prunes_authorized_after_expiry() -> None:
    """Authorized sessions are reclaimed once past expires_at (no leak)."""
    live = hf_auth.DeviceCodeSession(
        session_id="live",
        user_code="A",
        verification_uri="u",
        verification_uri_complete="u",
        status="authorized",
        expires_at=time.time() + 300,
    )
    stale = hf_auth.DeviceCodeSession(
        session_id="stale",
        user_code="B",
        verification_uri="u",
        verification_uri_complete="u",
        status="authorized",
        expires_at=time.time() - 1,
    )
    hf_auth._device_code_sessions["live"] = live
    hf_auth._device_code_sessions["stale"] = stale

    hf_auth._cleanup_expired_device_sessions()

    assert "live" in hf_auth._device_code_sessions
    assert "stale" not in hf_auth._device_code_sessions
