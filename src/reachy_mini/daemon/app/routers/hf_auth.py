"""HuggingFace authentication API routes."""

import asyncio
import logging
from typing import Any

import aiohttp
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from reachy_mini.apps.sources import hf_auth
from reachy_mini.daemon.app.logging_ctx import kv_log

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/hf-auth")

# Central signaling server that tracks which robot is currently in use by
# which remote JS app. We proxy its /api/robot-status endpoint so the
# desktop frontend never needs to see the raw HF token.
CENTRAL_ROBOT_STATUS_URL = "https://cduss-reachy-mini-central.hf.space/api/robot-status"
CENTRAL_ROBOT_STATUS_TIMEOUT = aiohttp.ClientTimeout(total=5)


class TokenRequest(BaseModel):
    """Request model for saving a HuggingFace token."""

    token: str


class TokenResponse(BaseModel):
    """Response model for token operations."""

    status: str
    username: str | None = None
    message: str | None = None


# =============================================================================
# Token-based Authentication (Manual)
# =============================================================================


@router.post("/save-token")
async def save_token(req: Request, request: TokenRequest) -> TokenResponse:
    """Save HuggingFace token after validation.

    On success this also makes sure the central signaling relay is
    running. The relay is normally started at media-server-acquire
    time, but the daemon may have booted without a token (so the relay
    never came up). Without this ensure-step the robot would remain
    invisible on central until the next daemon restart, even though
    the token is now stored on disk.
    """
    result = hf_auth.save_hf_token(request.token)

    if result["status"] == "error":
        kv_log(
            logger,
            logging.WARNING,
            "auth.save_token.failure",
            reason=result.get("message", "invalid_token"),
        )
        raise HTTPException(
            status_code=400, detail=result.get("message", "Invalid token")
        )

    kv_log(
        logger,
        logging.INFO,
        "auth.save_token.success",
        username=result.get("username"),
    )

    daemon = getattr(req.app.state, "daemon", None)
    if daemon is not None and getattr(daemon, "wireless_version", False):
        try:
            relay_result = await daemon.ensure_central_signaling_relay()
            kv_log(
                logger,
                logging.INFO,
                "auth.save_token.relay_ensured",
                **relay_result,
            )
        except Exception as e:
            logger.warning("[save-token] ensure_central_signaling_relay failed: %s", e)

    return TokenResponse(
        status="success",
        username=result.get("username"),
    )


@router.get("/status")
async def get_auth_status() -> dict[str, Any]:
    """Check if user is authenticated with HuggingFace."""
    return hf_auth.check_token_status()


@router.get("/token")
async def get_token() -> dict[str, Any]:
    """Return the stored HuggingFace token in plaintext.

    This exists so remote clients (e.g. the mobile app) can bridge the
    token into a sandboxed iframe that itself cannot go through the HF
    OAuth flow (hit by ``X-Frame-Options: SAMEORIGIN`` on ``/login``).

    Security note: the daemon's HTTP API is already unauthenticated on
    the local network, so any client that can reach this endpoint can
    also start/stop apps at will. Exposing the token here does not
    meaningfully widen the attack surface, but we deliberately keep the
    endpoint separate from ``/status`` so the desktop frontend - which
    never needs the raw token - is not tempted to consume it.
    """
    token = hf_auth.get_hf_token()
    if not token:
        raise HTTPException(status_code=404, detail="No HF token stored")
    return {"token": token}


@router.get("/relay-status")
async def get_relay_status(request: Request) -> dict[str, Any]:
    """Get the central signaling relay connection status."""
    # Check if this is a Lite version (no WebRTC support)
    daemon = getattr(request.app.state, "daemon", None)
    if daemon and not daemon.wireless_version:
        return {
            "state": "unavailable",
            "message": "Coming soon to Lite version",
            "is_connected": False,
        }

    try:
        from reachy_mini.media.central_signaling_relay import get_relay_status

        return get_relay_status()
    except ImportError:
        return {
            "state": "unavailable",
            "message": "Central relay not available",
            "is_connected": False,
        }


@router.delete("/token")
async def delete_token() -> dict[str, str]:
    """Delete stored HuggingFace token."""
    success = hf_auth.delete_hf_token()

    if not success:
        kv_log(logger, logging.ERROR, "auth.delete_token.failure")
        raise HTTPException(status_code=500, detail="Failed to delete token")

    kv_log(logger, logging.INFO, "auth.delete_token.success")
    return {"status": "success"}


@router.post("/refresh-relay")
async def refresh_relay(request: Request) -> dict[str, Any]:
    """Force the central signaling relay to reconnect with the stored token.

    Drops the relay's current connection and re-registers with the
    currently stored HF token.

    Intended as a recovery handle for clients (e.g. the mobile app) that
    detect a desync between ``/relay-status`` (claims ``connected``) and
    ``/central-robot-status`` (returns ``robots: []``). That combination
    means the relay still holds an SSE channel open with central but is
    no longer registered as a producer for the authenticated user, most
    commonly because the relay attached with a token that has since been
    rotated or because a transient error during ``setPeerStatus`` went
    unnoticed. From the outside this manifests as "the robot is online
    but no one can call it" until someone restarts the daemon.

    Rather than asking users to SSH in and run ``systemctl restart``,
    this endpoint triggers the relay's existing ``_token_updated`` event
    path (the same one ``save-token`` uses on login), which cleanly
    tears down the SSE connection, refreshes the HF token from
    ``huggingface_hub.get_token`` and reconnects. Works with any token
    currently stored (raw user tokens or OAuth access tokens) because
    we go through the relay's reconnect logic rather than re-validating
    the token.

    Response shape:
        { "status": "requested", "token_available": bool, "reason"?: str }

    ``token_available`` is false if the daemon has no HF token stored
    at all (user not logged in). In that case the relay will just
    transition to ``WAITING_FOR_TOKEN`` after the reconnect, which is
    the correct behaviour.
    """
    token = hf_auth.get_hf_token()

    try:
        from reachy_mini.media import central_signaling_relay as _csr
        from reachy_mini.media.central_signaling_relay import notify_force_reconnect

        # If the daemon booted without a token, the relay never came
        # up. In that case `notify_force_reconnect` is a no-op. Try to
        # bring the relay up first so this endpoint actually does
        # something useful for the cold-boot recovery case.
        if _csr._relay_instance is None:
            daemon = getattr(request.app.state, "daemon", None)
            if daemon is not None and getattr(daemon, "wireless_version", False):
                try:
                    await daemon.ensure_central_signaling_relay()
                except Exception as e:
                    logger.warning(
                        "[refresh-relay] ensure_central_signaling_relay failed: %s",
                        e,
                    )

        # Unconditionally drop & reconnect. We used to call
        # `notify_token_change(token)` here, but that path is guarded
        # by an `old_token == new_token` early-return in the relay,
        # which turned this endpoint into a no-op whenever the token
        # had not changed - exactly the case we ship this endpoint
        # for. `notify_force_reconnect` skips that guard.
        await notify_force_reconnect()
    except ImportError:
        return {
            "status": "skipped",
            "token_available": bool(token),
            "reason": "relay_unavailable",
        }
    except Exception as e:
        logger.warning("[refresh-relay] notify_force_reconnect failed: %s", e)
        raise HTTPException(
            status_code=500, detail=f"Failed to refresh relay: {e}"
        ) from e

    return {"status": "requested", "token_available": bool(token)}


@router.get("/central-robot-status")
async def get_central_robot_status() -> dict[str, Any]:
    """Proxy to the central signaling server's /api/robot-status endpoint.

    Uses the stored HF token server-side so the desktop frontend never
    sees the raw token. The frontend polls this to know whether any of
    the user's robots is currently held by a remote JS app, and if so
    which one (so it can show "In use by Hand Tracker" in the UI).

    Response shape:
        { "available": bool, "robots": [...], "reason": str? }

    `available` is false when:
      - no HF token stored (user not logged in)
      - central server is unreachable / returned an error
    Callers should treat `available: false` as "unknown, don't block".
    """
    token = hf_auth.get_hf_token()
    if not token:
        return {"available": False, "robots": [], "reason": "not_authenticated"}

    try:
        async with aiohttp.ClientSession(
            timeout=CENTRAL_ROBOT_STATUS_TIMEOUT
        ) as session:
            # Token goes in the Authorization header, not the URL —
            # otherwise it leaks into central's access logs and any
            # intermediate proxy's logs. The desktop frontend already
            # never sees the raw token (we read it server-side via
            # hf_auth.get_hf_token); header use keeps it off the
            # wire-visible URL as well.
            async with session.get(
                CENTRAL_ROBOT_STATUS_URL,
                headers={"Authorization": f"Bearer {token}"},
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "available": True,
                        "robots": data.get("robots", []),
                    }
                if response.status == 401:
                    return {"available": False, "robots": [], "reason": "token_invalid"}
                logger.warning(
                    "[central-robot-status] unexpected status %s", response.status
                )
                return {
                    "available": False,
                    "robots": [],
                    "reason": f"http_{response.status}",
                }
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.debug("[central-robot-status] central unreachable: %s", e)
        return {"available": False, "robots": [], "reason": "unreachable"}


# =============================================================================
# OAuth Authentication (One-click login)
# =============================================================================
#
# Uses fixed redirect URIs:
#   - Wireless: http://reachy-mini.local:8000/api/hf-auth/oauth/callback
#   - Lite:     http://localhost:8000/api/hf-auth/oauth/callback
#
# Register both URIs with your HuggingFace OAuth app.
# =============================================================================


@router.get("/oauth/configured")
async def is_oauth_configured() -> dict[str, Any]:
    """Check if OAuth is configured."""
    return {
        "configured": hf_auth.is_oauth_configured(),
    }


@router.get("/oauth/start")
async def start_oauth(request: Request, use_localhost: bool = False) -> dict[str, Any]:
    """Start a new OAuth authorization session.

    Returns the auth_url to redirect the user to HuggingFace.

    Args:
        request: The incoming HTTP request.
        use_localhost: When True, use localhost:8000 as the OAuth callback URL.
            Passed by the desktop app which proxies localhost:8000 to the robot.

    """
    # Get wireless_version from app state
    wireless_version = getattr(request.app.state, "daemon", None)
    if wireless_version:
        wireless_version = wireless_version.wireless_version
    else:
        # Fallback: check if accessed via reachy-mini.local
        host = request.headers.get("host", "")
        wireless_version = "reachy-mini.local" in host

    result = hf_auth.create_oauth_session(
        wireless_version=wireless_version,
        use_localhost=use_localhost,
    )

    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result.get("message"))

    return result


@router.get("/oauth/status/{session_id}")
async def get_oauth_status(session_id: str) -> dict[str, Any]:
    """Poll for OAuth session status.

    The frontend polls this endpoint to check if the user has
    completed authorization.
    """
    return hf_auth.get_oauth_session_status(session_id)


@router.delete("/oauth/session/{session_id}")
async def cancel_oauth_session(session_id: str) -> dict[str, str]:
    """Cancel an OAuth session."""
    if hf_auth.cancel_oauth_session(session_id):
        return {"status": "success"}
    raise HTTPException(status_code=404, detail="Session not found")


@router.get("/oauth/callback")
async def oauth_callback(
    request: Request,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    error_description: str | None = None,
) -> HTMLResponse:
    """Handle OAuth callback from HuggingFace.

    This is where HF redirects after user authorizes.
    Shows a success/error page that the user can close.
    """
    if error:
        # OAuth error from HF
        session = hf_auth.get_session_by_state(state) if state else None
        if session:
            session.status = "error"
            session.error_message = error_description or error

        return HTMLResponse(
            content=_oauth_result_page(
                success=False,
                message=error_description or error,
            )
        )

    if not code or not state:
        return HTMLResponse(
            content=_oauth_result_page(
                success=False,
                message="Missing authorization code or state",
            ),
            status_code=400,
        )

    # Determine if wireless based on the callback URL
    host = request.headers.get("host", "")
    wireless_version = "reachy-mini.local" in host

    # Exchange code for token
    result = await hf_auth.exchange_code_for_token(
        code=code,
        state=state,
        wireless_version=wireless_version,
    )

    if result["status"] == "success":
        return HTMLResponse(
            content=_oauth_result_page(
                success=True,
                message=f"Successfully logged in as {result.get('username', 'user')}!",
            )
        )
    else:
        return HTMLResponse(
            content=_oauth_result_page(
                success=False,
                message=result.get("message", "Authorization failed"),
            )
        )


def _oauth_result_page(success: bool, message: str) -> str:
    """Generate a simple HTML page showing OAuth result."""
    icon = "✅" if success else "❌"
    title = "Login Successful" if success else "Login Failed"
    color = "#10b981" if success else "#ef4444"

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{title} - Reachy Mini</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white;
        }}
        .container {{
            text-align: center;
            padding: 2rem;
            max-width: 400px;
        }}
        .icon {{
            font-size: 4rem;
            margin-bottom: 1rem;
        }}
        h1 {{
            color: {color};
            margin-bottom: 0.5rem;
        }}
        p {{
            color: #a0aec0;
            font-size: 1.1rem;
            line-height: 1.5;
        }}
        .hint {{
            margin-top: 2rem;
            padding: 1rem;
            background: rgba(255,255,255,0.1);
            border-radius: 8px;
            font-size: 0.9rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="icon">{icon}</div>
        <h1>{title}</h1>
        <p>{message}</p>
        <div class="hint">
            You can close this window and return to your robot's dashboard.
        </div>
    </div>
    <script>
        // Auto-close after 3 seconds if opened as popup
        if (window.opener) {{
            setTimeout(() => window.close(), 3000);
        }}
    </script>
</body>
</html>"""
