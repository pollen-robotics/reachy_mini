"""Unit tests for the device-code OAuth flow in ``apps.sources.hf_auth``.

These cover only our orchestration (session lifecycle, status polling, relay
notification, error mapping); the Hugging Face device-code protocol itself lives
in ``huggingface_hub`` and is stubbed here so the tests are deterministic and run
regardless of the installed ``huggingface_hub`` version.
"""

from __future__ import annotations

import asyncio
import sys
import types
from typing import Any

import pytest

from reachy_mini.apps.sources import hf_auth


@pytest.fixture(autouse=True)
def _clear_sessions() -> Any:
    """Each test starts with an empty session registry."""
    hf_auth._device_code_sessions.clear()
    yield
    hf_auth._device_code_sessions.clear()


@pytest.fixture
def device_code_error() -> type[Exception]:
    """Return ``huggingface_hub``'s ``DeviceCodeError``, creating a stand-in on
    older hub versions and wiring it where ``hf_auth`` imports it from."""
    try:
        from huggingface_hub.errors import DeviceCodeError  # type: ignore

        return DeviceCodeError
    except ImportError:
        import huggingface_hub.errors as hub_errors  # type: ignore

        class DeviceCodeError(Exception):  # noqa: N818 — mirror hub naming
            def __init__(self, message: str, error_code: str | None = None) -> None:
                super().__init__(message)
                self.error_code = error_code

        hub_errors.DeviceCodeError = DeviceCodeError  # type: ignore[attr-defined]
        return DeviceCodeError


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


# --------------------------------------------------------------------------- #
# start_device_code_login
# --------------------------------------------------------------------------- #


def test_start_returns_user_code_and_registers_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_oauth_device(
        monkeypatch, request_device_code=lambda: dict(_DEVICE_INFO)
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom() -> dict[str, Any]:
        raise RuntimeError("network down")

    _install_fake_oauth_device(monkeypatch, request_device_code=_boom)

    result = asyncio.run(hf_auth.start_device_code_login())

    assert result["status"] == "error"
    assert "network down" in result["message"]
    assert hf_auth._device_code_sessions == {}


# --------------------------------------------------------------------------- #
# _run_device_code_poll
# --------------------------------------------------------------------------- #


def test_poll_success_persists_token_and_notifies_relay(
    monkeypatch: pytest.MonkeyPatch, device_code_error: type[Exception]
) -> None:
    token_response = {
        "access_token": "hf_new_token",
        "refresh_token": "refresh-xyz",
        "expires_in": 2_592_000,
    }
    _install_fake_oauth_device(
        monkeypatch, poll_device_token=lambda info, **kw: token_response
    )

    persisted: dict[str, Any] = {}

    def _fake_persist(response: dict[str, Any]) -> tuple[str, str]:
        persisted["response"] = response
        return ("oauth-alice", "alice")

    monkeypatch.setattr(hf_auth, "_persist_device_oauth_token", _fake_persist)

    notified: dict[str, Any] = {}

    async def _fake_notify(token: str | None) -> None:
        notified["token"] = token

    import reachy_mini.media.central_signaling_relay as relay_module

    monkeypatch.setattr(relay_module, "notify_token_change", _fake_notify)

    session = hf_auth.DeviceCodeSession(
        session_id="s1",
        user_code="ABCD-1234",
        verification_uri="https://hf.co/oauth/device",
        verification_uri_complete="https://hf.co/oauth/device",
    )
    hf_auth._device_code_sessions["s1"] = session

    asyncio.run(hf_auth._run_device_code_poll(session, dict(_DEVICE_INFO)))

    assert session.status == "authorized"
    assert session.username == "alice"
    assert persisted["response"] is token_response
    assert notified["token"] == "hf_new_token"


def test_poll_expired_maps_to_expired_status(
    monkeypatch: pytest.MonkeyPatch, device_code_error: type[Exception]
) -> None:
    def _expire(info: Any, **kw: Any) -> dict[str, Any]:
        raise device_code_error("Device code expired. Please try again.")

    _install_fake_oauth_device(monkeypatch, poll_device_token=_expire)

    session = hf_auth.DeviceCodeSession(
        session_id="s2",
        user_code="ABCD-1234",
        verification_uri="https://hf.co/oauth/device",
        verification_uri_complete="https://hf.co/oauth/device",
    )

    asyncio.run(hf_auth._run_device_code_poll(session, dict(_DEVICE_INFO)))

    assert session.status == "expired"
    assert "expired" in (session.error_message or "").lower()


def test_poll_denied_maps_to_error_status(
    monkeypatch: pytest.MonkeyPatch, device_code_error: type[Exception]
) -> None:
    def _deny(info: Any, **kw: Any) -> dict[str, Any]:
        raise device_code_error("Authorization was denied. Please try again.")

    _install_fake_oauth_device(monkeypatch, poll_device_token=_deny)

    session = hf_auth.DeviceCodeSession(
        session_id="s3",
        user_code="ABCD-1234",
        verification_uri="https://hf.co/oauth/device",
        verification_uri_complete="https://hf.co/oauth/device",
    )

    asyncio.run(hf_auth._run_device_code_poll(session, dict(_DEVICE_INFO)))

    assert session.status == "error"


# --------------------------------------------------------------------------- #
# get_device_code_session_status / consume_device_session_relay_pending
# --------------------------------------------------------------------------- #


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
