"""User personalities sync to HuggingFace dataset."""

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/personalities")

PERSONALITIES_DIR = Path(
    "/venvs/apps_venv/lib/python3.12/site-packages/"
    "reachy_mini_conversation_app/profiles/user_personalities"
)

logger = logging.getLogger(__name__)


@router.get("/list")
def list_personalities() -> dict[str, Any]:
    """List local user personalities."""
    if not PERSONALITIES_DIR.exists():
        return {"personalities": [], "exists": False}

    personalities = []
    for d in sorted(PERSONALITIES_DIR.iterdir()):
        if d.is_dir():
            files = [f.name for f in d.iterdir() if f.is_file()]
            personalities.append({"name": d.name, "files": files})

    return {
        "personalities": personalities,
        "exists": True,
    }


@router.post("/sync")
def sync_to_hf() -> dict[str, Any]:
    """Sync user personalities to a HuggingFace dataset.

    Creates the dataset if it doesn't exist, then uploads all personality folders.
    """
    try:
        from huggingface_hub import HfApi, get_token, whoami
    except ImportError:
        raise HTTPException(status_code=500, detail="huggingface_hub not installed")

    token = get_token()
    if not token:
        raise HTTPException(status_code=401, detail="Not logged in to HuggingFace. Please login first.")

    try:
        user_info = whoami(token=token)
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid HF token: {e}")

    username = user_info.get("name", "")
    if not username:
        raise HTTPException(status_code=500, detail="Could not determine HF username")

    repo_id = f"{username}/user_personalities"
    api = HfApi(token=token)

    if not PERSONALITIES_DIR.exists():
        raise HTTPException(status_code=404, detail=f"Personalities directory not found: {PERSONALITIES_DIR}")

    # Create dataset repo if it doesn't exist
    created = False
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
    except Exception:
        try:
            api.create_repo(repo_id=repo_id, repo_type="dataset", private=False)
            created = True
            logger.info(f"Created HF dataset: {repo_id}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create dataset: {e}")

    # Upload entire personalities folder
    try:
        api.upload_folder(
            folder_path=str(PERSONALITIES_DIR),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Sync user personalities from Reachy Mini",
        )
        logger.info(f"Synced personalities to {repo_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

    return {
        "status": "success",
        "repo_id": repo_id,
        "created": created,
        "url": f"https://huggingface.co/datasets/{repo_id}",
    }
