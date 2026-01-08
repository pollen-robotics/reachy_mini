"""HuggingFace authentication API routes."""

from typing import Any

from fastapi import APIRouter, HTTPException
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


@router.delete("/token")
async def delete_token() -> dict[str, str]:
    """Delete stored HuggingFace token."""
    success = hf_auth.delete_hf_token()

    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete token")

    return {"status": "success"}
