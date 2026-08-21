"""Tests for the device-code OAuth flow."""

import asyncio
import sys
import time
import types
from typing import Any
from unittest.mock import AsyncMock

import pytest
from huggingface_hub.errors import DeviceCodeError

from reachy_mini.apps.sources import hf_auth


@pytest.fixture(autouse=True)
def _clear_sessions() -> Any:
    """Each test starts with an empty session registry."""
    hf_auth._device_code_sessions.clear()
    yield
    hf_auth._device_code_sessions.clear()


def _install_fake_oauth_device(
    monkeypatch: pytest.MonkeyPatch,
    *,
    request_device_code: Any = None,
    poll_device_token: Any = None,
) -> None:
    """Inject a fake ``huggingface_hub.utils._oauth_device`` module."""
    fake = types.ModuleType("huggingface_hub.utils._oauth_device")
    if request_device_code is not None:
        fake.request_device_code = request_device_code  # type: ignore[attr-defined]
    if poll_device_token is not None:
        fake.poll_device_token = poll_device_token  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils._oauth_device", fake)


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
    _install_fake_oauth_device(
        monkeypatch, request_device_code=lambda: dict(_DEVICE_INFO)
    )

    monkeypatch.setattr(hf_auth, "_run_device_code_poll", AsyncMock())

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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom() -> dict[str, Any]:
        raise RuntimeError("network down")

    _install_fake_oauth_device(monkeypatch, request_device_code=_boom)

    result = asyncio.run(hf_auth.start_device_code_login())

    assert result["status"] == "error"
    assert result["message"] == hf_auth.AUTHENTICATION_UNAVAILABLE_MESSAGE
    assert "network down" not in result["message"]
    assert hf_auth._device_code_sessions == {}


def test_poll_success_persists_token_and_notifies_relay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token_response = {
        "access_token": "hf_new_token",
        "refresh_token": "refresh-xyz",
        "expires_in": 2_592_000,
    }
    _install_fake_oauth_device(
        monkeypatch, poll_device_token=lambda info, **kw: token_response
    )

    monkeypatch.setattr(hf_auth, "whoami", lambda **_kwargs: {"name": "alice"})

    notified: dict[str, Any] = {}

    async def _fake_notify(token: str | None) -> None:
        notified["token"] = token

    import reachy_mini.media.central_signaling_relay as relay_module

    monkeypatch.setattr(relay_module, "notify_token_change", _fake_notify)

    session = hf_auth.DeviceCodeSession(session_id="s1")
    hf_auth._device_code_sessions["s1"] = session

    asyncio.run(hf_auth._run_device_code_poll(session, dict(_DEVICE_INFO)))

    assert session.status == "authorized"
    assert session.username == "alice"
    assert hf_auth.get_hf_token() == "hf_new_token"
    assert notified["token"] == "hf_new_token"


def test_poll_expired_maps_to_expired_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _expire(info: Any, **kw: Any) -> dict[str, Any]:
        raise DeviceCodeError("Device code expired. Please try again.")

    _install_fake_oauth_device(monkeypatch, poll_device_token=_expire)

    session = hf_auth.DeviceCodeSession(session_id="s2")

    asyncio.run(hf_auth._run_device_code_poll(session, dict(_DEVICE_INFO)))

    assert session.status == "expired"
    assert "expired" in (session.error_message or "").lower()


def test_poll_denied_maps_to_error_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _deny(info: Any, **kw: Any) -> dict[str, Any]:
        raise DeviceCodeError("Authorization was denied. Please try again.")

    _install_fake_oauth_device(monkeypatch, poll_device_token=_deny)

    session = hf_auth.DeviceCodeSession(session_id="s3")

    asyncio.run(hf_auth._run_device_code_poll(session, dict(_DEVICE_INFO)))

    assert session.status == "error"


def test_status_unknown_session_is_expired() -> None:
    assert hf_auth.get_device_code_session_status("nope")["status"] == "expired"


def test_status_authorized_includes_username() -> None:
    session = hf_auth.DeviceCodeSession(
        session_id="s4",
        status="authorized",
        username="bob",
    )
    hf_auth._device_code_sessions["s4"] = session

    result = hf_auth.get_device_code_session_status("s4")
    assert result == {"status": "authorized", "username": "bob"}


def test_consume_relay_pending_fires_once() -> None:
    session = hf_auth.DeviceCodeSession(
        session_id="s5",
        status="authorized",
    )
    hf_auth._device_code_sessions["s5"] = session

    assert hf_auth.consume_device_session_relay_pending("s5") is True
    assert hf_auth.consume_device_session_relay_pending("s5") is False


def test_consume_relay_pending_false_while_pending() -> None:
    session = hf_auth.DeviceCodeSession(
        session_id="s6",
        status="pending",
    )
    hf_auth._device_code_sessions["s6"] = session

    assert hf_auth.consume_device_session_relay_pending("s6") is False


def test_cancel_session_removes_it() -> None:
    session = hf_auth.DeviceCodeSession(session_id="s7")
    hf_auth._device_code_sessions["s7"] = session

    assert hf_auth.cancel_device_code_session("s7") is True
    assert "s7" not in hf_auth._device_code_sessions
    assert hf_auth.cancel_device_code_session("s7") is False


def test_cancel_signals_the_polling_thread() -> None:
    """Cancel must set the event the polling thread observes, not just drop it."""
    session = hf_auth.DeviceCodeSession(session_id="s8")
    hf_auth._device_code_sessions["s8"] = session

    assert session.cancel_event.is_set() is False
    assert hf_auth.cancel_device_code_session("s8") is True
    assert session.cancel_event.is_set() is True


def test_poll_aborts_when_cancel_event_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cancelled session unwinds poll_device_token via its on_pending hook."""
    calls = {"on_pending": 0}

    def _fake_poll(device_info: Any, *, on_pending: Any = None) -> dict[str, Any]:
        for _ in range(1000):
            if on_pending is not None:
                calls["on_pending"] += 1
                on_pending()
        raise AssertionError("poll should have been cancelled before returning")

    _install_fake_oauth_device(monkeypatch, poll_device_token=_fake_poll)

    session = hf_auth.DeviceCodeSession(session_id="s9")
    session.cancel_event.set()

    asyncio.run(hf_auth._run_device_code_poll(session, dict(_DEVICE_INFO)))

    assert calls["on_pending"] == 1
    assert session.status == "cancelled"


def test_cancel_after_device_poll_refuses_the_returned_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _return_token_after_cancel(
        device_info: Any, *, on_pending: Any = None
    ) -> dict[str, Any]:
        (session_id,) = hf_auth._device_code_sessions
        assert hf_auth.cancel_device_code_session(session_id) is True
        return {"access_token": "late-token"}

    _install_fake_oauth_device(
        monkeypatch,
        request_device_code=lambda: dict(_DEVICE_INFO),
        poll_device_token=_return_token_after_cancel,
    )
    monkeypatch.setattr(hf_auth, "whoami", lambda **_kwargs: {"name": "alice"})
    import reachy_mini.media.central_signaling_relay as relay_module

    relay_notify = AsyncMock()
    monkeypatch.setattr(relay_module, "notify_token_change", relay_notify)

    async def scenario() -> hf_auth.DeviceCodeSession:
        start = await hf_auth.start_device_code_login()
        session = hf_auth._device_code_sessions[start["session_id"]]
        assert session.task is not None
        await session.task
        return session

    session = asyncio.run(scenario())

    assert session.status == "cancelled"
    assert hf_auth.get_hf_token() is None
    relay_notify.assert_not_awaited()


def test_authorized_session_gets_bounded_ttl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On success the session's expires_at is shortened to the authorized TTL."""
    token_response = {"access_token": "hf_x", "refresh_token": "r", "expires_in": 10}
    _install_fake_oauth_device(
        monkeypatch, poll_device_token=lambda info, **kw: token_response
    )
    monkeypatch.setattr(hf_auth, "whoami", lambda **_kwargs: {"name": "user"})

    session = hf_auth.DeviceCodeSession(
        session_id="s10",
        expires_at=time.time() + 900,
    )
    hf_auth._device_code_sessions[session.session_id] = session

    before = time.time()
    asyncio.run(hf_auth._run_device_code_poll(session, dict(_DEVICE_INFO)))

    assert session.status == "authorized"
    assert session.expires_at <= before + hf_auth._AUTHORIZED_SESSION_TTL_S + 1


def test_cleanup_prunes_authorized_after_expiry() -> None:
    """Authorized sessions are reclaimed once past expires_at (no leak)."""
    live = hf_auth.DeviceCodeSession(
        session_id="live",
        status="authorized",
        expires_at=time.time() + 300,
    )
    stale = hf_auth.DeviceCodeSession(
        session_id="stale",
        status="authorized",
        expires_at=time.time() - 1,
    )
    hf_auth._device_code_sessions["live"] = live
    hf_auth._device_code_sessions["stale"] = stale

    hf_auth._cleanup_expired_device_sessions()

    assert "live" in hf_auth._device_code_sessions
    assert "stale" not in hf_auth._device_code_sessions


def test_wireless_first_run_links_the_robot_with_a_refreshable_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh robot finishes the mobile device-code flow holding its own token."""
    _install_fake_oauth_device(
        monkeypatch,
        request_device_code=lambda: dict(_DEVICE_INFO),
        poll_device_token=lambda info, **kw: {
            "access_token": "device-token",
            "refresh_token": "r1",
            "expires_in": 3600,
        },
    )

    def _fail_username_lookup(**_kwargs: Any) -> None:
        raise RuntimeError

    monkeypatch.setattr(hf_auth, "whoami", _fail_username_lookup)
    monkeypatch.setattr(hf_auth, "_notify_relay_of_token_change", lambda *_a: None)

    assert hf_auth.check_token_status() == {"is_logged_in": False, "username": None}

    async def scenario() -> dict[str, Any]:
        start = await hf_auth.start_device_code_login()
        task = hf_auth._device_code_sessions[start["session_id"]].task
        assert task is not None
        await task
        return hf_auth.get_device_code_session_status(start["session_id"])

    assert asyncio.run(scenario()) == {"status": "authorized", "username": ""}
    assert hf_auth.get_hf_token() == "device-token"
    assert hf_auth._read_store().refresh_token == "r1"
