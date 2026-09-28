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
from huggingface_hub.errors import DeviceCodeError, HfHubHTTPError
from huggingface_hub.utils._oauth_device import (
    poll_device_token,
    refresh_access_token,
    request_device_code,
)

from reachy_mini.utils.proxy import proxy_for

logger = logging.getLogger(__name__)

# =============================================================================
# OAuth Configuration
# =============================================================================
# Register ONE OAuth app at https://huggingface.co/settings/connected-applications
# with TWO redirect URIs:
#   - http://reachy-mini.local:8000/api/hf-auth/oauth/callback  (wireless)
#   - http://localhost:8000/api/hf-auth/oauth/callback          (lite)
#
# Then set HF_OAUTH_CLIENT_ID on all robots (same value for all).
#
# Environment variables:
#   HF_OAUTH_CLIENT_ID     - Required for OAuth login
#
# Pollen's HuggingFace OAuth app - works for all Reachy Mini robots
_DEFAULT_OAUTH_CLIENT_ID = "71146982-8184-45a2-b05a-d561b3cd701d"

OAUTH_CLIENT_ID: str | None = os.environ.get(
    "HF_OAUTH_CLIENT_ID", _DEFAULT_OAUTH_CLIENT_ID
)
# Read-only: publishing an app happens on a dev machine, never on the robot.
OAUTH_SCOPES = "openid profile read-repos"

# Fixed redirect URIs (must match what's registered with HuggingFace)
OAUTH_REDIRECT_URI_WIRELESS = "http://reachy-mini.local:8000/api/hf-auth/oauth/callback"
OAUTH_REDIRECT_URI_LITE = "http://localhost:8000/api/hf-auth/oauth/callback"

# Returned over HTTP, so never carry provider text, exception detail, or tokens.
AUTHENTICATION_FAILED_MESSAGE = "Authentication failed. Please try again."
AUTHENTICATION_UNAVAILABLE_MESSAGE = (
    "Hugging Face authentication is unavailable. Please try again."
)
LOGIN_EXPIRED_MESSAGE = "Login expired. Please try again."
_CANCELLED_SESSION_MESSAGE = "Sign-in was cancelled. Please try again."
CREDENTIAL_SAVE_FAILED_MESSAGE = "Could not save credentials. Please try again."

# CLI credentials belong to the user, so the daemon keeps and reads only its own.
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
    code_verifier: str  # PKCE code verifier
    redirect_uri: str
    status: str = "pending"  # pending, authorized, expired, error
    username: str | None = None
    error_message: str | None = None
    lifecycle_generation: int = 0
    expires_at: float = field(
        default_factory=lambda: time.time() + _OAUTH_SESSION_TTL_S
    )  # 10 min expiry


# In-memory storage for OAuth sessions (device-flow-like pattern)
_oauth_sessions: dict[str, OAuthSession] = {}


def _drop_expired(sessions: dict[str, Any]) -> None:
    now = time.time()
    for session_id in [sid for sid, s in sessions.items() if s.expires_at < now]:
        del sessions[session_id]


def create_oauth_session(
    wireless_version: bool, use_localhost: bool = False
) -> dict[str, Any]:
    """Create a redirect OAuth session."""
    _drop_expired(_oauth_sessions)

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

    # Generate PKCE pair for secure public client auth
    # Generate code_verifier (43-128 characters, URL-safe)
    code_verifier = secrets.token_urlsafe(32)

    # Generate code_challenge = BASE64URL(SHA256(code_verifier))
    digest = hashlib.sha256(code_verifier.encode()).digest()
    code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()

    session = OAuthSession(
        session_id=state,  # Use state as session ID for simplicity
        code_verifier=code_verifier,
        redirect_uri=redirect_uri,
        lifecycle_generation=_read_store().lifecycle_generation,
    )
    _oauth_sessions[state] = session

    # Build HuggingFace OAuth authorization URL with PKCE
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
        "expires_in": _OAUTH_SESSION_TTL_S,  # 10 minutes
    }


def get_oauth_session(session_id: str) -> OAuthSession | None:
    """Return an active redirect OAuth session."""
    _drop_expired(_oauth_sessions)
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

    # Exchange code for token using PKCE
    token_url = "https://huggingface.co/oauth/token"
    data = {
        "grant_type": "authorization_code",
        "client_id": OAUTH_CLIENT_ID,
        "code": code,
        "redirect_uri": session.redirect_uri,
        "code_verifier": session.code_verifier,  # PKCE verification
    }

    try:
        # Explicit proxy resolution (HTTP_PROXY/HTTPS_PROXY/NO_PROXY) —
        # deliberately NOT trust_env=True, which would also read ~/.netrc
        # and break Authorization-header requests (see utils/proxy.py).
        async with aiohttp.ClientSession() as http_session:
            async with http_session.post(
                token_url, data=data, proxy=proxy_for(token_url)
            ) as response:
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

        # HuggingFace returns accessToken (camelCase)
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

    # Save token directly to the daemon's credential store
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

    # Get username
    username = _resolve_username(access_token)

    # Update session
    session.status = "authorized"
    session.username = username

    # Notify central relay of new token for immediate reconnection
    await _notify_relay(access_token)

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


# =============================================================================
# Device Code OAuth (RFC 8628) — refresh-capable, redirect-free login
# =============================================================================
# Unlike the authorization-code flow above, the device-code flow:
#   - needs NO redirect URI, so it does not depend on the robot being reachable
#     at a fixed hostname (reachy-mini.local) — the phone only displays a short
#     code + URL and the robot polls Hugging Face for the result.
#   - yields a refresh token. The daemon persists it in its own credential
#     store and `get_hf_credential()` transparently renews the access token
#     when it is close to expiry, so a long-running robot never needs the user
#     to re-authenticate by hand.
#
# It uses Hugging Face's first-party device-code OAuth client (shipped in
# huggingface_hub via DEVICE_CODE_OAUTH_CLIENT_ID), not the Pollen OAuth app,
# so it works even when HF_OAUTH_CLIENT_ID is not configured.

# How long an authorized session is kept so the frontend can read the result and
# the relay can be started once, after which _drop_expired removes it (a token-less
# boot polls for a few seconds; 5 min is a generous margin).
_AUTHORIZED_SESSION_TTL_S = 300


class _DeviceCodeCancelled(Exception):
    """Raised in poll_device_token's on_pending hook to stop its worker thread."""


@dataclass
class DeviceCodeSession:
    """A device-code OAuth login polled in the background."""

    session_id: str
    status: str = "pending"  # pending, authorized, error, expired, cancelled
    username: str | None = None
    error_message: str | None = None
    lifecycle_generation: int = 0
    relay_started: bool = False  # set once the central relay has been brought up
    expires_at: float = field(default_factory=lambda: time.time() + 900)
    # Keep a strong reference to the polling task so it is not garbage-collected.
    task: asyncio.Task[None] | None = None
    # Set to request cancellation; observed by the polling thread's on_pending hook.
    cancel_event: threading.Event = field(default_factory=threading.Event)


# In-memory storage for device-code login sessions, polled by the frontend.
_device_code_sessions: dict[str, DeviceCodeSession] = {}


def _complete_device_login(response: Any, session: DeviceCodeSession) -> str | None:
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
        # Bound the authorized session's lifetime so _drop_expired reclaims it after
        # the frontend has read the result (rather than leaking until daemon restart).
        session.expires_at = time.time() + _AUTHORIZED_SESSION_TTL_S

    return _resolve_username(response["access_token"])


async def start_device_code_login() -> dict[str, Any]:
    """Begin a device-code OAuth login."""
    _drop_expired(_device_code_sessions)
    lifecycle_generation = _read_store().lifecycle_generation

    try:
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
    def _abort_if_cancelled() -> None:
        # poll_device_token calls this after each "authorization pending" poll,
        # just before it sleeps; raising here unwinds it and frees the worker
        # thread within ~one poll interval instead of blocking to expiry.
        if session.cancel_event.is_set():
            raise _DeviceCodeCancelled

    try:
        # poll_device_token blocks (time.sleep between polls) — keep it off the loop.
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

    # Notify a *running* central relay so it reconnects with the new token. A
    # token-less boot has no relay instance yet; the status route starts one.
    await _notify_relay(response.get("access_token"))


def get_device_code_session_status(session_id: str) -> dict[str, Any]:
    """Return the status of a device-code OAuth session."""
    _drop_expired(_device_code_sessions)
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


def _resolve_username(token: str) -> str:
    try:
        user_info = whoami(token=token)
    except Exception as error:  # noqa: BLE001 - username is optional
        logger.debug("[HF Auth] Could not resolve username (%s)", type(error).__name__)
        return ""
    if not isinstance(user_info, dict):
        return ""
    return str(user_info.get("name") or user_info.get("fullname") or "")


async def _notify_relay(new_token: str | None) -> None:
    try:
        # Lazy: the relay imports this module and pulls in websockets.
        from reachy_mini.media.central_signaling_relay import notify_token_change

        await notify_token_change(new_token)
    except Exception as error:  # noqa: BLE001 - relay notification is best effort
        logger.debug("[HF Auth] Could not notify relay (%s)", type(error).__name__)


def _notify_relay_of_token_change(new_token: str | None = None) -> None:
    # Try to get the running event loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop - run in new loop (blocking but quick)
        asyncio.run(_notify_relay(new_token))
    else:
        # If we're already in an async context, schedule as task
        loop.create_task(_notify_relay(new_token))


def save_hf_token(token: str) -> dict[str, Any]:
    """Validate and store a manually entered Hugging Face token."""
    lifecycle_generation = _read_store().lifecycle_generation
    try:
        # Validate token first by making an API call
        user_info = HfApi(token=token).whoami()

        # Persist token for future runs
        if not _persist_login({"access_token": token}, lifecycle_generation):
            return {"status": "error", "message": AUTHENTICATION_FAILED_MESSAGE}

        # Notify central relay of new token for immediate reconnection
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


def _usable_credential(stored: _Stored) -> HfCredential:
    expired = stored.expires_at is not None and stored.expires_at <= time.time()
    return HfCredential(
        None if expired else stored.access_token, stored.lifecycle_generation
    )


def get_hf_credential(force_refresh: bool = False) -> HfCredential:
    """Return the daemon bearer and its lifecycle generation as one atomic read."""
    with _store_lock:
        stored = _read_store()
    due = force_refresh or (
        stored.expires_at is not None
        and stored.expires_at <= time.time() + _REFRESH_MARGIN_S
    )
    if not stored.refresh_token or not due:
        return _usable_credential(stored)

    # The network round-trip runs unlocked; the write below is compare-and-set.
    refreshed = None
    try:
        response = refresh_access_token(stored.refresh_token)
        refreshed = replace(stored, **_token_fields(response, stored.refresh_token))
    except Exception as error:  # noqa: BLE001 - refresh is best effort
        logger.warning(
            "[HF Auth] Could not refresh credentials (%s)", type(error).__name__
        )
    with _store_lock:
        current = _read_store()
        if refreshed is not None and current == stored:
            _write_store(refreshed)
            current = refreshed
    return _usable_credential(current)


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

    # Notify central relay that user logged out
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
