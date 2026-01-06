"""HuggingFace authentication management for private spaces."""

from typing import Any, Optional

from huggingface_hub import HfApi, HfFolder, whoami
from huggingface_hub.utils import HfHubHTTPError  # type: ignore


def save_hf_token(token: str) -> dict[str, Any]:
    """Save HuggingFace token securely.

    Uses HfFolder.save_token() which stores in ~/.cache/huggingface/token
    This is the standard HF location with 600 permissions.

    Args:
        token: The HuggingFace token to save

    Returns:
        dict with status and username if valid, or error message

    """
    try:
        # Validate token first by making an API call
        api = HfApi(token=token)
        user_info = api.whoami()

        # If valid, save it
        HfFolder.save_token(token)

        return {
            "status": "success",
            "username": user_info.get("name", ""),
        }
    except HfHubHTTPError:
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
    return HfFolder.get_token()


def delete_hf_token() -> bool:
    """Delete stored HuggingFace token.

    Returns:
        True if successfully deleted.

    """
    try:
        HfFolder.delete_token()
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
