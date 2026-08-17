"""HuggingFace authentication management for private spaces."""

import asyncio
import json
import logging
import os
import secrets
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

import aiohttp
from huggingface_hub import HfApi, whoami
from huggingface_hub.constants import HF_TOKEN_PATH
from huggingface_hub.errors import HfHubHTTPError

logger = logging.getLogger(__name__)

# One OAuth app registered at huggingface.co/settings/connected-applications,
# with both redirect URIs below. HF_OAUTH_CLIENT_ID overrides it per robot.
_DEFAULT_OAUTH_CLIENT_ID = "71146982-8184-45a2-b05a-d561b3cd701d"

OAUTH_CLIENT_ID: Optional[str] = os.environ.get(
    "HF_OAUTH_CLIENT_ID", _DEFAULT_OAUTH_CLIENT_ID
)
OAUTH_CLIENT_SECRET: Optional[str] = os.environ.get("HF_OAUTH_CLIENT_SECRET")
# The robot only reads at runtime. Publishing an app happens on a dev machine.
OAUTH_SCOPES = "openid profile read-repos"

# Fixed redirect URIs (must match what's registered with HuggingFace)
OAUTH_REDIRECT_URI_WIRELESS = "http://reachy-mini.local:8000/api/hf-auth/oauth/callback"
OAUTH_REDIRECT_URI_LITE = "http://localhost:8000/api/hf-auth/oauth/callback"

# Reach HTTP responses, so never carry provider text, exception detail, or tokens.
AUTHENTICATION_FAILED_MESSAGE = "Authentication failed. Please try again."
_CANCELLED_SESSION_MESSAGE = "Sign-in was cancelled. Please try again."
CREDENTIAL_SAVE_FAILED_MESSAGE = "Could not save credentials. Please try again."

# The daemon owns this file. Credentials the user holds elsewhere are never read:
# a token issued to the CLI was not issued to the robot (RFC 9700).
_STORE_FILENAME = "reachy_mini_daemon_credentials.json"
_STORE_VERSION = 1
_REFRESH_MARGIN_S = 300

_store_lock = threading.RLock()


@dataclass(frozen=True)
class HfCredential:
    """The daemon bearer and the lifecycle it belongs to, read as one value."""

    token: Optional[str] = field(repr=False)
    lifecycle_generation: int


@dataclass(frozen=True)
class _Stored:
    """On-disk shape of the daemon credential store."""

    version: int = _STORE_VERSION
    signed_out: bool = False
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    expires_at: Optional[int] = None
    lifecycle_generation: int = 0


def _store_path() -> Path:
    return Path(HF_TOKEN_PATH).with_name(_STORE_FILENAME)


def _write_store(stored: _Stored) -> None:
    """Replace the store atomically. mkstemp already restricts the mode to 0600."""
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _token_fields(response: Any, fallback_refresh: Optional[str]) -> dict[str, Any]:
    """Map a Hugging Face token response onto the fields we store."""
    expires_in = response.get("expires_in")
    return {
        "signed_out": False,
        "access_token": response["access_token"],
        "refresh_token": response.get("refresh_token") or fallback_refresh,
        "expires_at": int(time.time()) + int(expires_in) if expires_in else None,
    }


def _persist_login(fields: dict[str, Any], expected_generation: int) -> bool:
    """Store a completed login unless the credential lifecycle moved on meanwhile."""
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


def current_lifecycle_generation() -> int:
    """Return the generation a login must still match when it completes."""
    return _read_store().lifecycle_generation


# In-memory storage for OAuth sessions (device-flow-like pattern)
_oauth_sessions: dict[str, "OAuthSession"] = {}


@dataclass
class OAuthSession:
    """Represents an OAuth authorization session."""

    session_id: str
    user_code: str  # Short code shown to user (e.g., "ABCD-1234")
    state: str  # CSRF protection
    code_verifier: str  # PKCE code verifier
    wireless_version: bool  # To know which redirect URI to use
    use_localhost: bool = False  # Force localhost callback (desktop app proxy)
    status: str = "pending"  # pending, authorized, expired, error
    access_token: Optional[str] = None
    username: Optional[str] = None
    error_message: Optional[str] = None
    lifecycle_generation: int = 0
    created_at: float = field(default_factory=time.time)
    expires_at: float = field(
        default_factory=lambda: time.time() + 600
    )  # 10 min expiry


def _generate_user_code() -> str:
    """Generate a short, easy-to-type user code like 'ABCD-1234'."""
    letters = "".join(secrets.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(4))
    numbers = "".join(secrets.choice("0123456789") for _ in range(4))
    return f"{letters}-{numbers}"


def _generate_pkce_pair() -> tuple[str, str]:
    """Generate PKCE code_verifier and code_challenge.

    Returns:
        Tuple of (code_verifier, code_challenge)

    """
    import base64
    import hashlib

    # Generate code_verifier (43-128 characters, URL-safe)
    code_verifier = secrets.token_urlsafe(32)

    # Generate code_challenge = BASE64URL(SHA256(code_verifier))
    digest = hashlib.sha256(code_verifier.encode()).digest()
    code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()

    return code_verifier, code_challenge


def _cleanup_expired_sessions() -> None:
    """Remove expired OAuth sessions."""
    now = time.time()
    expired = [sid for sid, s in _oauth_sessions.items() if s.expires_at < now]
    for sid in expired:
        del _oauth_sessions[sid]


def get_oauth_redirect_uri(wireless_version: bool, use_localhost: bool = False) -> str:
    """Get the appropriate OAuth redirect URI based on robot type.

    Args:
        wireless_version: True for wireless robots, False for Lite.
        use_localhost: When True, force localhost callback (for desktop app
            proxy — the app forwards localhost:8000 to the robot).

    Returns:
        The redirect URI to use for OAuth.

    """
    if use_localhost:
        return OAUTH_REDIRECT_URI_LITE
    if wireless_version:
        return OAUTH_REDIRECT_URI_WIRELESS
    else:
        return OAUTH_REDIRECT_URI_LITE


def create_oauth_session(
    wireless_version: bool, use_localhost: bool = False
) -> dict[str, Any]:
    """Create a new OAuth authorization session.

    Args:
        wireless_version: True for wireless robots, False for Lite.
        use_localhost: When True, force localhost callback (desktop app proxy).

    Returns:
        Session info including auth_url to redirect the user to.

    """
    _cleanup_expired_sessions()

    if not OAUTH_CLIENT_ID:
        return {
            "status": "error",
            "message": "OAuth not configured. Set HF_OAUTH_CLIENT_ID environment variable.",
        }

    redirect_uri = get_oauth_redirect_uri(wireless_version, use_localhost)
    state = secrets.token_urlsafe(32)

    # Generate PKCE pair for secure public client auth
    code_verifier, code_challenge = _generate_pkce_pair()

    session = OAuthSession(
        session_id=state,  # Use state as session ID for simplicity
        user_code="",  # Not needed for this flow
        state=state,
        code_verifier=code_verifier,
        wireless_version=wireless_version,
        use_localhost=use_localhost,
        lifecycle_generation=current_lifecycle_generation(),
    )
    _oauth_sessions[state] = session

    # Build HuggingFace OAuth authorization URL with PKCE
    from urllib.parse import urlencode

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
        "expires_in": 600,  # 10 minutes
    }


def get_oauth_session(session_id: str) -> Optional[OAuthSession]:
    """Get an OAuth session by ID."""
    _cleanup_expired_sessions()
    return _oauth_sessions.get(session_id)


def get_session_by_state(state: str) -> Optional[OAuthSession]:
    """Get an OAuth session by its state parameter."""
    _cleanup_expired_sessions()
    for session in _oauth_sessions.values():
        if session.state == state:
            return session
    return None


async def exchange_code_for_token(
    code: str,
    state: str,
    wireless_version: bool,
) -> dict[str, Any]:
    """Exchange an authorization code for an access token.

    Args:
        code: The authorization code from HuggingFace
        state: The state parameter for CSRF verification
        wireless_version: True for wireless robots, False for Lite.

    Returns:
        Result dict with status and token/error info

    """
    session = get_session_by_state(state)
    if not session:
        return {
            "status": "error",
            "message": "Invalid or expired session. Please try again.",
        }

    if not OAUTH_CLIENT_ID:
        session.status = "error"
        session.error_message = "OAuth not configured"
        return {"status": "error", "message": "OAuth not configured"}

    redirect_uri = get_oauth_redirect_uri(
        session.wireless_version, session.use_localhost
    )

    # Exchange code for token using PKCE
    token_url = "https://huggingface.co/oauth/token"
    data = {
        "grant_type": "authorization_code",
        "client_id": OAUTH_CLIENT_ID,
        "code": code,
        "redirect_uri": redirect_uri,
        "code_verifier": session.code_verifier,  # PKCE verification
    }

    try:
        async with aiohttp.ClientSession() as http_session:
            async with http_session.post(token_url, data=data) as response:
                response_text = await response.text()
                if response.status != 200:
                    session.status = "error"
                    session.error_message = f"Token exchange failed (HTTP {response.status}): {response_text}"
                    return {"status": "error", "message": session.error_message}

                import json

                token_data = json.loads(response_text)

        # HuggingFace returns accessToken (camelCase)
        access_token = token_data.get("access_token") or token_data.get("accessToken")
        if not access_token:
            session.status = "error"
            session.error_message = f"No access token. Response: {token_data}"
            return {"status": "error", "message": session.error_message}

    except Exception as e:
        session.status = "error"
        session.error_message = f"Token request error: {type(e).__name__}: {e}"
        return {"status": "error", "message": session.error_message}

    try:
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
    username = ""
    try:
        user_info = whoami(token=access_token)
        if isinstance(user_info, dict):
            username = user_info.get("name", "") or user_info.get("fullname", "")
    except Exception:
        pass  # Username is optional

    # Update session
    session.status = "authorized"
    session.access_token = access_token
    session.username = username

    # Notify central relay of new token for immediate reconnection
    try:
        from reachy_mini.media.central_signaling_relay import notify_token_change

        await notify_token_change(access_token)
        logger.info("[HF Auth] Notified central relay of OAuth login")
    except ImportError:
        pass  # Central relay not available
    except Exception as e:
        logger.debug(f"[HF Auth] Could not notify relay: {e}")

    return {
        "status": "success",
        "username": username,
    }


def get_oauth_session_status(session_id: str) -> dict[str, Any]:
    """Check the status of an OAuth session.

    Used for polling from the frontend.

    Args:
        session_id: The session ID to check

    Returns:
        Status dict with authorization state

    """
    session = get_oauth_session(session_id)
    if not session:
        return {"status": "expired", "message": "Session expired or not found"}

    result: dict[str, Any] = {"status": session.status}

    if session.status == "authorized":
        result["username"] = session.username
    elif session.status == "error":
        result["message"] = session.error_message

    return result


def cancel_oauth_session(session_id: str) -> bool:
    """Cancel an OAuth session."""
    if session_id in _oauth_sessions:
        del _oauth_sessions[session_id]
        return True
    return False


def is_oauth_configured() -> bool:
    """Check if OAuth is configured."""
    return bool(OAUTH_CLIENT_ID)


# =============================================================================
# Device Code OAuth (RFC 8628) — refresh-capable, redirect-free login
# =============================================================================
# No redirect URI, so it does not depend on reaching the robot at a fixed
# hostname. The phone shows a short code and the robot polls Hugging Face. It
# uses the hub's first-party device-code client, so it works without
# HF_OAUTH_CLIENT_ID. We call the hub's private request, poll and refresh
# helpers because the public login() would write into the user's shared store.

# In-memory storage for device-code login sessions, polled by the frontend.
_device_code_sessions: dict[str, "DeviceCodeSession"] = {}

# Long enough for the client to read the result and the relay to start once.
_AUTHORIZED_SESSION_TTL_S = 300


class _LoginSuperseded(Exception):
    """Raised when a login completes after sign-out or another login replaced it."""


class _DeviceCodeCancelled(Exception):
    """Raised in poll_device_token's on_pending hook to stop its worker thread."""


@dataclass
class DeviceCodeSession:
    """A pending device-code OAuth login, polled in the background."""

    session_id: str
    user_code: str
    verification_uri: str
    verification_uri_complete: str
    status: str = "pending"  # pending, authorized, error, expired, cancelled
    username: Optional[str] = None
    error_message: Optional[str] = None
    lifecycle_generation: int = 0
    relay_started: bool = False  # set once the central relay has been brought up
    created_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + 900)
    # Keep a strong reference to the polling task so it is not garbage-collected.
    task: Optional["asyncio.Task[None]"] = None
    # Set to request cancellation. The polling thread observes it in on_pending.
    cancel_event: threading.Event = field(default_factory=threading.Event)


def _cleanup_expired_device_sessions() -> None:
    """Remove finished or expired device-code sessions.

    Authorized sessions are included: without this they would live in memory
    until the daemon restarts. `start_device_code_login` sets their `expires_at`
    to a short TTL once authorized so this prunes them shortly after use.
    """
    now = time.time()
    stale = [
        sid
        for sid, s in _device_code_sessions.items()
        if s.expires_at < now
        and s.status in ("pending", "authorized", "expired", "error", "cancelled")
    ]
    for sid in stale:
        _device_code_sessions.pop(sid, None)


def _persist_device_oauth_token(response: Any, expected_generation: int) -> str:
    """Store a device-code token in the daemon's own record and return the username."""
    if not _persist_login(_token_fields(response, None), expected_generation):
        raise _LoginSuperseded
    user_info = whoami(token=response["access_token"])
    if not isinstance(user_info, dict):
        return ""
    return str(user_info.get("name") or "")


async def start_device_code_login() -> dict[str, Any]:
    """Begin a device-code OAuth login and poll for completion in the background.

    Returns immediately with the user code and verification URL to display, plus a
    `session_id` the frontend polls via `get_device_code_session_status`.
    """
    _cleanup_expired_device_sessions()

    try:
        from huggingface_hub.utils._oauth_device import request_device_code

        device_info = await asyncio.to_thread(request_device_code)
    except Exception as e:  # noqa: BLE001 — surface any failure to the caller
        logger.error("[HF Auth] Failed to request device code: %s", e)
        return {
            "status": "error",
            "message": f"Could not start login: {type(e).__name__}: {e}",
        }

    session_id = secrets.token_urlsafe(16)
    session = DeviceCodeSession(
        session_id=session_id,
        user_code=device_info["user_code"],
        verification_uri=device_info["verification_uri"],
        verification_uri_complete=device_info["verification_uri_complete"],
        expires_at=time.time() + int(device_info.get("expires_in", 900)),
        lifecycle_generation=current_lifecycle_generation(),
    )
    _device_code_sessions[session_id] = session
    session.task = asyncio.create_task(_run_device_code_poll(session, device_info))

    return {
        "status": "pending",
        "session_id": session_id,
        "user_code": session.user_code,
        "verification_uri": session.verification_uri,
        "verification_uri_complete": session.verification_uri_complete,
        "interval": int(device_info.get("interval", 5)),
        "expires_in": int(device_info.get("expires_in", 900)),
    }


async def _run_device_code_poll(session: DeviceCodeSession, device_info: Any) -> None:
    """Poll Hugging Face until the user authorizes the device, then persist the token."""
    from huggingface_hub.errors import DeviceCodeError
    from huggingface_hub.utils._oauth_device import poll_device_token

    def _abort_if_cancelled() -> None:
        # Raising from the hub's per-poll hook frees the worker within one interval.
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
    except DeviceCodeError as e:
        logger.info("[HF Auth] Device-code login failed: %s", e)
        session.status = "expired" if "expired" in str(e).lower() else "error"
        session.error_message = str(e)
        return
    except Exception as e:  # noqa: BLE001
        logger.error("[HF Auth] Device-code polling error: %s", e)
        session.status = "error"
        session.error_message = f"{type(e).__name__}: {e}"
        return

    try:
        username = await asyncio.to_thread(
            _persist_device_oauth_token, response, session.lifecycle_generation
        )
    except _LoginSuperseded:
        logger.info("[HF Auth] Device-code login superseded: %s", session.session_id)
        session.status = "cancelled"
        return
    except Exception as error:  # noqa: BLE001
        logger.error(
            "[HF Auth] Failed to persist device-code token (%s)",
            type(error).__name__,
        )
        session.status = "error"
        session.error_message = CREDENTIAL_SAVE_FAILED_MESSAGE
        return

    session.username = username or ""
    session.status = "authorized"
    # Bound the authorized session's lifetime so _cleanup reclaims it after the
    # frontend has read the result (rather than leaking until daemon restart).
    session.expires_at = time.time() + _AUTHORIZED_SESSION_TTL_S

    # Notify a *running* central relay so it reconnects with the new token. A
    # token-less boot has no relay instance yet, so the status route starts one.
    try:
        from reachy_mini.media.central_signaling_relay import notify_token_change

        await notify_token_change(response.get("access_token"))
        logger.info("[HF Auth] Notified central relay of device-code login")
    except ImportError:
        pass  # Central relay not available (e.g. Lite version)
    except Exception as e:  # noqa: BLE001
        logger.debug("[HF Auth] Could not notify relay: %s", e)


def get_device_code_session_status(session_id: str) -> dict[str, Any]:
    """Check the status of a device-code login session (polled by the frontend)."""
    _cleanup_expired_device_sessions()
    session = _device_code_sessions.get(session_id)
    if not session:
        return {"status": "expired", "message": "Session expired or not found"}

    result: dict[str, Any] = {"status": session.status}
    if session.status == "authorized":
        result["username"] = session.username
    elif session.status in ("error", "expired"):
        result["message"] = session.error_message
    return result


def consume_device_session_relay_pending(session_id: str) -> bool:
    """Return True exactly once after a session becomes authorized.

    Lets the HTTP layer (which holds the daemon handle) start the central relay a
    single time on a token-less boot, without the background poll task needing a
    reference to the daemon.
    """
    session = _device_code_sessions.get(session_id)
    if session is None or session.status != "authorized" or session.relay_started:
        return False
    session.relay_started = True
    return True


def cancel_device_code_session(session_id: str) -> bool:
    """Cancel a pending device-code session and stop its polling thread.

    Signals the polling thread via `cancel_event` (observed by `on_pending`),
    which unwinds `poll_device_token` within ~one poll interval and frees the
    worker thread. We deliberately do not call `task.cancel()`: cancelling the
    asyncio task while the worker thread is still running would orphan the
    thread and surface a stray "Future exception was never retrieved" warning.
    While awaiting the thread the task stays referenced by the executor, so it
    is safe to drop the session here and let the poll unwind cooperatively.
    """
    session = _device_code_sessions.pop(session_id, None)
    if session is None:
        return False
    session.cancel_event.set()
    return True


def _notify_relay_of_token_change(new_token: Optional[str] = None) -> None:
    """Notify the central signaling relay of a token change.

    This is called after login/logout to trigger reconnection with the
    new (or no) token. It handles the async call in a background task.
    """
    try:
        from reachy_mini.media.central_signaling_relay import notify_token_change

        # Try to get the running event loop
        try:
            loop = asyncio.get_running_loop()
            # If we're already in an async context, schedule as task
            loop.create_task(notify_token_change(new_token))
        except RuntimeError:
            # No running loop - run in new loop (blocking but quick)
            asyncio.run(notify_token_change(new_token))

        logger.info("[HF Auth] Notified central relay of token change")
    except ImportError:
        # Central relay module not available (e.g., Lite version)
        pass
    except Exception as e:
        logger.debug(f"[HF Auth] Could not notify relay: {e}")


def save_hf_token(token: str) -> dict[str, Any]:
    """Save a HuggingFace access token securely.

    Validates the token against the Hugging Face API and, if valid,
    stores it using the standard Hugging Face authentication mechanism
    for reuse across sessions.

    Args:
        token: The HuggingFace access token to save.

    Returns:
        A dict containing:
        - "status": "success" or "error"
        - "username": the associated Hugging Face username if successful
        - "message": an error description if unsuccessful

    """
    try:
        # Validate token first by making an API call
        api = HfApi(token=token)
        user_info = api.whoami()

        generation = current_lifecycle_generation()
        if not _persist_login({"access_token": token}, generation):
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


def get_hf_token() -> Optional[str]:
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
    """Check if a token is stored and valid.

    Returns:
        Status dict with is_logged_in and username.

    """
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
