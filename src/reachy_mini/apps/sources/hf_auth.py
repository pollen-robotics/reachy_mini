"""HuggingFace authentication management for private spaces."""

import asyncio
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import aiohttp
from huggingface_hub import HfApi, get_token, login, logout, whoami
from huggingface_hub.utils import HfHubHTTPError  # type: ignore

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
#   HF_OAUTH_CLIENT_SECRET - Optional (for confidential clients)
#
# Pollen's HuggingFace OAuth app - works for all Reachy Mini robots
_DEFAULT_OAUTH_CLIENT_ID = "71146982-8184-45a2-b05a-d561b3cd701d"

OAUTH_CLIENT_ID: Optional[str] = os.environ.get(
    "HF_OAUTH_CLIENT_ID", _DEFAULT_OAUTH_CLIENT_ID
)
OAUTH_CLIENT_SECRET: Optional[str] = os.environ.get("HF_OAUTH_CLIENT_SECRET")
OAUTH_SCOPES = os.environ.get(
    "HF_OAUTH_SCOPES",
    "openid profile read-repos write-repos manage-repos inference-api",
)

# Fixed redirect URIs (must match what's registered with HuggingFace)
OAUTH_REDIRECT_URI_WIRELESS = "http://reachy-mini.local:8000/api/hf-auth/oauth/callback"
OAUTH_REDIRECT_URI_LITE = "http://localhost:8000/api/hf-auth/oauth/callback"

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
    status: str = "pending"  # pending, authorized, expired, error
    access_token: Optional[str] = None
    username: Optional[str] = None
    error_message: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    expires_at: float = field(
        default_factory=lambda: time.time() + 600
    )  # 10 min expiry


def configure_oauth(
    client_id: str,
    client_secret: Optional[str] = None,
    scopes: str = "openid profile read-repos",
) -> None:
    """Configure OAuth credentials.

    Args:
        client_id: HuggingFace OAuth client ID
        client_secret: OAuth client secret (optional for public clients)
        scopes: Space-separated OAuth scopes

    """
    global OAUTH_CLIENT_ID, OAUTH_CLIENT_SECRET, OAUTH_SCOPES
    OAUTH_CLIENT_ID = client_id
    OAUTH_CLIENT_SECRET = client_secret
    OAUTH_SCOPES = scopes


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


def get_oauth_redirect_uri(wireless_version: bool) -> str:
    """Get the appropriate OAuth redirect URI based on robot type.

    Args:
        wireless_version: True for wireless robots, False for Lite.

    Returns:
        The redirect URI to use for OAuth.

    """
    if wireless_version:
        return OAUTH_REDIRECT_URI_WIRELESS
    else:
        return OAUTH_REDIRECT_URI_LITE


def create_oauth_session(wireless_version: bool) -> dict[str, Any]:
    """Create a new OAuth authorization session.

    Args:
        wireless_version: True for wireless robots, False for Lite.

    Returns:
        Session info including auth_url to redirect the user to.

    """
    _cleanup_expired_sessions()

    if not OAUTH_CLIENT_ID:
        return {
            "status": "error",
            "message": "OAuth not configured. Set HF_OAUTH_CLIENT_ID environment variable.",
        }

    redirect_uri = get_oauth_redirect_uri(wireless_version)
    state = secrets.token_urlsafe(32)

    # Generate PKCE pair for secure public client auth
    code_verifier, code_challenge = _generate_pkce_pair()

    session = OAuthSession(
        session_id=state,  # Use state as session ID for simplicity
        user_code="",  # Not needed for this flow
        state=state,
        code_verifier=code_verifier,
        wireless_version=wireless_version,
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

    redirect_uri = get_oauth_redirect_uri(session.wireless_version)

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

    # Save token directly to HuggingFace token file
    # (login() doesn't work well with OAuth tokens)
    try:
        from pathlib import Path

        token_path = Path.home() / ".cache" / "huggingface" / "token"
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(access_token)
    except Exception as e:
        session.status = "error"
        session.error_message = f"Failed to save token: {type(e).__name__}: {e}"
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

        # Persist token for future runs (no prompt since token is provided)
        # add_to_git_credential=False keeps it from touching git credentials.
        login(token=token, add_to_git_credential=False)

        # Notify central relay of new token for immediate reconnection
        _notify_relay_of_token_change(token)

        return {
            "status": "success",
            "username": user_info.get("name", ""),
        }
    except (HfHubHTTPError, ValueError):
        # ValueError can be raised by `login()` on invalid token (v1.x behavior)
        return {
            "status": "error",
            "message": "Invalid token or network error",
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }


def get_hf_token() -> Optional[str]:
    """Get stored HuggingFace token.

    Returns:
        The stored token, or None if no token is stored.

    """
    return get_token()


def delete_hf_token() -> bool:
    """Delete stored HuggingFace token(s).

    Note: logout() without arguments logs out from all saved access tokens.
    """
    try:
        logout()
        # Notify central relay that user logged out
        _notify_relay_of_token_change(None)
        return True
    except Exception:
        return False


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
