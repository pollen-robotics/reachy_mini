"""HuggingFace authentication management for private spaces."""

from typing import Any, Optional

from huggingface_hub import HfApi, get_token, login, logout, whoami
from huggingface_hub.utils import HfHubHTTPError  # type: ignore


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
