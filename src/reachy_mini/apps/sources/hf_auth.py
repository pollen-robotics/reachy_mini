"""Hugging Face authentication for private resources."""

import asyncio
import base64
import hashlib
import json
import logging
import os
import secrets
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import aiohttp
from huggingface_hub import HfApi, whoami
from huggingface_hub.constants import HF_TOKEN_PATH
from huggingface_hub.errors import HfHubHTTPError

logger = logging.getLogger(__name__)

# The OAuth app registers both redirects; HF_OAUTH_CLIENT_ID can override it.
_DEFAULT_OAUTH_CLIENT_ID = "71146982-8184-45a2-b05a-d561b3cd701d"

OAUTH_CLIENT_ID: str | None = os.environ.get(
    "HF_OAUTH_CLIENT_ID", _DEFAULT_OAUTH_CLIENT_ID
)
# The robot only reads at runtime. Publishing an app happens on a dev machine.
OAUTH_SCOPES = "openid profile read-repos"

# Fixed redirect URIs (must match what's registered with HuggingFace)
OAUTH_REDIRECT_URI_WIRELESS = "http://reachy-mini.local:8000/api/hf-auth/oauth/callback"
OAUTH_REDIRECT_URI_LITE = "http://localhost:8000/api/hf-auth/oauth/callback"

# Reach HTTP responses, so never carry provider text, exception detail, or tokens.
AUTHENTICATION_FAILED_MESSAGE = "Authentication failed. Please try again."
AUTHENTICATION_UNAVAILABLE_MESSAGE = (
    "Hugging Face authentication is unavailable. Please try again."
)
LOGIN_EXPIRED_MESSAGE = "Login expired. Please try again."
_CANCELLED_SESSION_MESSAGE = "Sign-in was cancelled. Please try again."
CREDENTIAL_SAVE_FAILED_MESSAGE = "Could not save credentials. Please try again."

# CLI credentials belong to the user, so the daemon never reads them (RFC 9700).
_STORE_FILENAME = "reachy_mini_daemon_credentials.json"
_STORE_VERSION = 1
_REFRESH_MARGIN_S = 300
_OAUTH_SESSION_TTL_S = 600

_store_lock = threading.RLock()


@dataclass(frozen=True)
class HfCredential:
    """The daemon bearer and the lifecycle it belongs to, read as one value."""

    token: str | None = field(repr=False)
    lifecycle_generation: int


@dataclass(frozen=True)
class _Stored:
    version: int = _STORE_VERSION
    signed_out: bool = False
    access_token: str | None = None
    refresh_token: str | None = None
    expires_at: int | None = None
    lifecycle_generation: int = 0


def _store_path() -> Path:
    return Path(HF_TOKEN_PATH).with_name(_STORE_FILENAME)


def _write_store(stored: _Stored) -> None:
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # mkstemp makes the file owner-only (0600 on POSIX).
    descriptor, name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", text=True
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(asdict(stored), output, separators=(",", ":"))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def _read_store() -> _Stored:
    path = _store_path()
    if not path.exists():
        return _Stored(signed_out=True)
    try:
        stored = _Stored(**json.loads(path.read_text(encoding="utf-8")))
        if stored.version != _STORE_VERSION or stored.lifecycle_generation < 0:
            raise ValueError("unsupported credential record")
        if not stored.signed_out and not stored.access_token:
            raise ValueError("credential record has no token")
        return stored
    except (OSError, TypeError, ValueError) as error:
        logger.warning(
            "[HF Auth] Unreadable credential store (%s)", type(error).__name__
        )
        return _Stored(signed_out=True)


def _token_fields(response: Any, fallback_refresh: str | None) -> dict[str, Any]:
    expires_in = response.get("expires_in")
    return {
        "signed_out": False,
        "access_token": response["access_token"],
        "refresh_token": response.get("refresh_token") or fallback_refresh,
        "expires_at": int(time.time()) + int(expires_in) if expires_in else None,
    }


def _persist_login(fields: dict[str, Any], expected_generation: int) -> bool:
    with _store_lock:
        current = _read_store()
        if current.lifecycle_generation != expected_generation:
            return False
        _write_store(
            replace(
                _Stored(),
                **fields,
                lifecycle_generation=current.lifecycle_generation + 1,
            )
        )
        return True


@dataclass
class OAuthSession:
    """A pending redirect OAuth login."""

    session_id: str
    code_verifier: str
    redirect_uri: str
    status: str = "pending"
    username: str | None = None
    error_message: str | None = None
    lifecycle_generation: int = 0
    expires_at: float = field(
        default_factory=lambda: time.time() + _OAUTH_SESSION_TTL_S
    )


_oauth_sessions: dict[str, OAuthSession] = {}


def _cleanup_expired_sessions() -> None:
    now = time.time()
    expired = [
        session_id
        for session_id, session in _oauth_sessions.items()
        if session.expires_at < now
    ]
    for session_id in expired:
        del _oauth_sessions[session_id]


def create_oauth_session(
    wireless_version: bool, use_localhost: bool = False
) -> dict[str, Any]:
    """Create a redirect OAuth session."""
    _cleanup_expired_sessions()

    if not OAUTH_CLIENT_ID:
        return {
            "status": "error",
            "message": "OAuth not configured. Set HF_OAUTH_CLIENT_ID environment variable.",
        }

    redirect_uri = (
        OAUTH_REDIRECT_URI_WIRELESS
        if wireless_version and not use_localhost
        else OAUTH_REDIRECT_URI_LITE
    )
    state = secrets.token_urlsafe(32)
    code_verifier = secrets.token_urlsafe(32)
    digest = hashlib.sha256(code_verifier.encode()).digest()
    code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()

    session = OAuthSession(
        session_id=state,
        code_verifier=code_verifier,
        redirect_uri=redirect_uri,
        lifecycle_generation=_read_store().lifecycle_generation,
    )
    _oauth_sessions[state] = session

    params = {
        "client_id": OAUTH_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "scope": OAUTH_SCOPES,
        "response_type": "code",
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    auth_url = f"https://huggingface.co/oauth/authorize?{urlencode(params)}"

    return {
        "status": "success",
        "session_id": state,
        "auth_url": auth_url,
        "redirect_uri": redirect_uri,
        "expires_in": _OAUTH_SESSION_TTL_S,
    }


def get_oauth_session(session_id: str) -> OAuthSession | None:
    """Return an active redirect OAuth session."""
    _cleanup_expired_sessions()
    return _oauth_sessions.get(session_id)


async def exchange_code_for_token(
    code: str,
    state: str,
) -> dict[str, Any]:
    """Exchange an OAuth authorization code for daemon credentials."""
    session = get_oauth_session(state)
    if session is None:
        return {
            "status": "error",
            "message": "Invalid or expired session. Please try again.",
        }

    if not OAUTH_CLIENT_ID:
        session.status = "error"
        session.error_message = "OAuth not configured"
        return {"status": "error", "message": "OAuth not configured"}

    token_url = "https://huggingface.co/oauth/token"
    data = {
        "grant_type": "authorization_code",
        "client_id": OAUTH_CLIENT_ID,
        "code": code,
        "redirect_uri": session.redirect_uri,
        "code_verifier": session.code_verifier,
    }

    try:
        async with aiohttp.ClientSession() as http_session:
            async with http_session.post(token_url, data=data) as response:
                response_text = await response.text()
                if response.status != 200:
                    logger.warning(
                        "[HF Auth] OAuth token exchange returned HTTP %s",
                        response.status,
                    )
                    session.status = "error"
                    session.error_message = AUTHENTICATION_FAILED_MESSAGE
                    return {"status": "error", "message": session.error_message}

                token_data = json.loads(response_text)

        access_token = token_data.get("access_token") or token_data.get("accessToken")
        if not access_token:
            logger.warning("[HF Auth] OAuth response did not include an access token")
            session.status = "error"
            session.error_message = AUTHENTICATION_FAILED_MESSAGE
            return {"status": "error", "message": session.error_message}
        token_data["access_token"] = access_token

    except Exception as error:
        logger.warning(
            "[HF Auth] OAuth token request failed (%s)", type(error).__name__
        )
        session.status = "error"
        session.error_message = AUTHENTICATION_UNAVAILABLE_MESSAGE
        return {"status": "error", "message": session.error_message}

    try:
        with _store_lock:
            if _oauth_sessions.get(session.session_id) is not session:
                landed = False
            else:
                landed = _persist_login(
                    _token_fields(token_data, None), session.lifecycle_generation
                )
    except OSError as error:
        logger.warning(
            "[HF Auth] Could not save OAuth credentials (%s)", type(error).__name__
        )
        session.status = "error"
        session.error_message = CREDENTIAL_SAVE_FAILED_MESSAGE
        return {"status": "error", "message": session.error_message}
    if not landed:
        session.status = "error"
        session.error_message = _CANCELLED_SESSION_MESSAGE
        return {"status": "error", "message": session.error_message}

    username = ""
    try:
        user_info = whoami(token=access_token)
        if isinstance(user_info, dict):
            username = user_info.get("name", "") or user_info.get("fullname", "")
    except Exception as error:  # noqa: BLE001 - username is optional
        logger.debug(
            "[HF Auth] Could not resolve OAuth username (%s)", type(error).__name__
        )

    session.status = "authorized"
    session.username = username

    try:
        from reachy_mini.media.central_signaling_relay import notify_token_change

        await notify_token_change(access_token)
        logger.info("[HF Auth] Notified central relay of OAuth login")
    except Exception as error:  # noqa: BLE001 - relay notification is best effort
        logger.debug(
            "[HF Auth] Could not notify relay (%s)", type(error).__name__
        )

    return {
        "status": "success",
        "username": username,
    }


def get_oauth_session_status(session_id: str) -> dict[str, Any]:
    """Return the status of a redirect OAuth session."""
    session = get_oauth_session(session_id)
    if session is None:
        return {"status": "expired", "message": "Session expired or not found"}

    result: dict[str, Any] = {"status": session.status}

    if session.status == "authorized":
        result["username"] = session.username
    elif session.status == "error":
        result["message"] = session.error_message

    return result


def cancel_oauth_session(session_id: str) -> bool:
    """Cancel an OAuth session."""
    with _store_lock:
        return _oauth_sessions.pop(session_id, None) is not None


def is_oauth_configured() -> bool:
    """Check if OAuth is configured."""
    return bool(OAUTH_CLIENT_ID)


# Device-code OAuth uses private hub helpers to avoid writing shared credentials.
_AUTHORIZED_SESSION_TTL_S = 300


class _DeviceCodeCancelled(Exception):
    """Raised in poll_device_token's on_pending hook to stop its worker thread."""


@dataclass
class DeviceCodeSession:
    """A device-code OAuth login polled in the background."""

    session_id: str
    status: str = "pending"
    username: str | None = None
    error_message: str | None = None
    lifecycle_generation: int = 0
    relay_started: bool = False
    expires_at: float = field(default_factory=lambda: time.time() + 900)
    task: asyncio.Task[None] | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)


_device_code_sessions: dict[str, DeviceCodeSession] = {}


def _cleanup_expired_device_sessions() -> None:
    now = time.time()
    stale = [
        session_id
        for session_id, session in _device_code_sessions.items()
        if session.expires_at < now
    ]
    for session_id in stale:
        _device_code_sessions.pop(session_id, None)


def _complete_device_login(
    response: Any, session: DeviceCodeSession
) -> str | None:
    with _store_lock:
        if (
            session.cancel_event.is_set()
            or _device_code_sessions.get(session.session_id) is not session
        ):
            return None
        if not _persist_login(
            _token_fields(response, None), session.lifecycle_generation
        ):
            return None
        session.status = "authorized"
        session.username = ""
        session.expires_at = time.time() + _AUTHORIZED_SESSION_TTL_S

    try:
        user_info = whoami(token=response["access_token"])
    except Exception as error:  # noqa: BLE001 - username is optional
        logger.debug(
            "[HF Auth] Could not resolve device username (%s)", type(error).__name__
        )
        return ""
    if not isinstance(user_info, dict):
        return ""
    return str(user_info.get("name") or "")


async def start_device_code_login() -> dict[str, Any]:
    """Begin a device-code OAuth login."""
    _cleanup_expired_device_sessions()
    lifecycle_generation = _read_store().lifecycle_generation

    try:
        from huggingface_hub.utils._oauth_device import request_device_code

        device_info = await asyncio.to_thread(request_device_code)
    except Exception as error:  # noqa: BLE001 - callers always get a result
        logger.error(
            "[HF Auth] Failed to request device code (%s)", type(error).__name__
        )
        return {"status": "error", "message": AUTHENTICATION_UNAVAILABLE_MESSAGE}

    session_id = secrets.token_urlsafe(16)
    session = DeviceCodeSession(
        session_id=session_id,
        expires_at=time.time() + int(device_info.get("expires_in", 900)),
        lifecycle_generation=lifecycle_generation,
    )
    _device_code_sessions[session_id] = session
    session.task = asyncio.create_task(_run_device_code_poll(session, device_info))

    return {
        "status": "pending",
        "session_id": session_id,
        "user_code": device_info["user_code"],
        "verification_uri": device_info["verification_uri"],
        "verification_uri_complete": device_info["verification_uri_complete"],
        "interval": int(device_info.get("interval", 5)),
        "expires_in": int(device_info.get("expires_in", 900)),
    }


async def _run_device_code_poll(session: DeviceCodeSession, device_info: Any) -> None:
    from huggingface_hub.errors import DeviceCodeError
    from huggingface_hub.utils._oauth_device import poll_device_token

    def _abort_if_cancelled() -> None:
        if session.cancel_event.is_set():
            raise _DeviceCodeCancelled

    try:
        response = await asyncio.to_thread(
            poll_device_token, device_info, on_pending=_abort_if_cancelled
        )
    except _DeviceCodeCancelled:
        logger.info("[HF Auth] Device-code login cancelled: %s", session.session_id)
        session.status = "cancelled"
        return
    except DeviceCodeError as error:
        logger.info("[HF Auth] Device-code login failed (%s)", type(error).__name__)
        expired = "expired" in str(error).lower()
        session.status = "expired" if expired else "error"
        session.error_message = (
            LOGIN_EXPIRED_MESSAGE if expired else AUTHENTICATION_FAILED_MESSAGE
        )
        return
    except Exception as error:  # noqa: BLE001
        logger.error("[HF Auth] Device-code polling error (%s)", type(error).__name__)
        session.status = "error"
        session.error_message = AUTHENTICATION_UNAVAILABLE_MESSAGE
        return

    try:
        username = await asyncio.to_thread(_complete_device_login, response, session)
    except Exception as error:  # noqa: BLE001
        logger.error(
            "[HF Auth] Failed to persist device-code token (%s)",
            type(error).__name__,
        )
        session.status = "error"
        session.error_message = CREDENTIAL_SAVE_FAILED_MESSAGE
        return

    if username is None:
        logger.info("[HF Auth] Device-code login superseded: %s", session.session_id)
        session.status = "cancelled"
        return
    session.username = username

    try:
        from reachy_mini.media.central_signaling_relay import notify_token_change

        await notify_token_change(response.get("access_token"))
        logger.info("[HF Auth] Notified central relay of device-code login")
    except Exception as error:  # noqa: BLE001
        logger.debug(
            "[HF Auth] Could not notify relay (%s)", type(error).__name__
        )


def get_device_code_session_status(session_id: str) -> dict[str, Any]:
    """Return the status of a device-code OAuth session."""
    _cleanup_expired_device_sessions()
    session = _device_code_sessions.get(session_id)
    if session is None:
        return {"status": "expired", "message": "Session expired or not found"}

    result: dict[str, Any] = {"status": session.status}
    if session.status == "authorized":
        result["username"] = session.username
    elif session.status in ("error", "expired"):
        result["message"] = session.error_message
    return result


def consume_device_session_relay_pending(session_id: str) -> bool:
    """Return whether an authorized session still needs to start the relay."""
    session = _device_code_sessions.get(session_id)
    if session is None or session.status != "authorized" or session.relay_started:
        return False
    session.relay_started = True
    return True


def cancel_device_code_session(session_id: str) -> bool:
    """Cancel a pending device-code session."""
    with _store_lock:
        session = _device_code_sessions.get(session_id)
        if session is None or session.status != "pending":
            return False
        _device_code_sessions.pop(session_id)
        session.cancel_event.set()
        session.status = "cancelled"
        return True


def _notify_relay_of_token_change(new_token: str | None = None) -> None:
    try:
        from reachy_mini.media.central_signaling_relay import notify_token_change

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(notify_token_change(new_token))
        else:
            loop.create_task(notify_token_change(new_token))

        logger.info("[HF Auth] Notified central relay of token change")
    except Exception as error:  # noqa: BLE001 - relay notification is best effort
        logger.debug(
            "[HF Auth] Could not notify relay (%s)", type(error).__name__
        )


def save_hf_token(token: str) -> dict[str, Any]:
    """Validate and store a manually entered Hugging Face token."""
    lifecycle_generation = _read_store().lifecycle_generation
    try:
        user_info = HfApi(token=token).whoami()

        if not _persist_login({"access_token": token}, lifecycle_generation):
            return {"status": "error", "message": AUTHENTICATION_FAILED_MESSAGE}

        _notify_relay_of_token_change(token)

        return {
            "status": "success",
            "username": user_info.get("name", ""),
        }
    except (HfHubHTTPError, ValueError):
        return {
            "status": "error",
            "message": "Invalid token or network error",
        }
    except Exception as error:  # noqa: BLE001 - callers always get a dict
        logger.warning(
            "[HF Auth] Could not save credentials (%s)", type(error).__name__
        )
        return {"status": "error", "message": CREDENTIAL_SAVE_FAILED_MESSAGE}


def get_hf_credential(force_refresh: bool = False) -> HfCredential:
    """Return the daemon bearer and its lifecycle generation as one atomic read."""
    with _store_lock:
        stored = _read_store()
        expires_at = stored.expires_at
        expired = expires_at is not None and expires_at <= time.time()
        # Also the answer when a due refresh fails: never hand out an expired token.
        usable = HfCredential(
            None if expired else stored.access_token, stored.lifecycle_generation
        )
        due = force_refresh or (
            expires_at is not None and expires_at <= time.time() + _REFRESH_MARGIN_S
        )
        if not stored.refresh_token or not due:
            return usable

        from huggingface_hub.utils._oauth_device import refresh_access_token

        try:
            response = refresh_access_token(stored.refresh_token)
        except Exception as error:  # noqa: BLE001 - a stale token must not raise
            logger.warning(
                "[HF Auth] Could not refresh credentials (%s)", type(error).__name__
            )
            return usable
        refreshed = replace(stored, **_token_fields(response, stored.refresh_token))
        _write_store(refreshed)
        return HfCredential(refreshed.access_token, refreshed.lifecycle_generation)


def get_hf_token() -> str | None:
    """Return the daemon-owned token, refreshing it when it is close to expiry."""
    return get_hf_credential().token


def delete_hf_token() -> bool:
    """Sign the robot out, leaving credentials the daemon does not own alone."""
    with _store_lock:
        try:
            current = _read_store()
            _write_store(
                _Stored(
                    signed_out=True,
                    lifecycle_generation=current.lifecycle_generation + 1,
                )
            )
        except OSError as error:
            logger.warning("[HF Auth] Could not sign out (%s)", type(error).__name__)
            return False
        for session in _device_code_sessions.values():
            session.cancel_event.set()
        _oauth_sessions.clear()
        _device_code_sessions.clear()

    _notify_relay_of_token_change(None)
    return True


def check_token_status() -> dict[str, Any]:
    """Return whether the daemon credentials are valid."""
    token = get_hf_token()
    if not token:
        return {"is_logged_in": False, "username": None}

    try:
        user_info = whoami(token=token)
        return {
            "is_logged_in": True,
            "username": user_info.get("name", ""),
        }
    except Exception:
        return {"is_logged_in": False, "username": None}
