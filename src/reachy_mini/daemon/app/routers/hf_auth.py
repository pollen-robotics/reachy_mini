"""HuggingFace authentication API routes."""

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from reachy_mini.apps.sources import hf_auth

router = APIRouter(prefix="/hf-auth")


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
async def save_token(request: TokenRequest) -> TokenResponse:
    """Save HuggingFace token after validation."""
    result = hf_auth.save_hf_token(request.token)

    if result["status"] == "error":
        raise HTTPException(
            status_code=400, detail=result.get("message", "Invalid token")
        )

    return TokenResponse(
        status="success",
        username=result.get("username"),
    )


@router.get("/status")
async def get_auth_status() -> dict[str, Any]:
    """Check if user is authenticated with HuggingFace."""
    return hf_auth.check_token_status()


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
        raise HTTPException(status_code=500, detail="Failed to delete token")

    return {"status": "success"}


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
async def start_oauth(request: Request) -> dict[str, Any]:
    """Start a new OAuth authorization session.

    Returns the auth_url to redirect the user to HuggingFace.
    """
    # Get wireless_version from app state
    wireless_version = getattr(request.app.state, "daemon", None)
    if wireless_version:
        wireless_version = wireless_version.wireless_version
    else:
        # Fallback: check if accessed via reachy-mini.local
        host = request.headers.get("host", "")
        wireless_version = "reachy-mini.local" in host

    result = hf_auth.create_oauth_session(wireless_version=wireless_version)

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
