"""Cache management router for Reachy Mini Daemon API."""

import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/cache")
logger = logging.getLogger(__name__)


@router.post("/clear-hf")
def clear_huggingface_cache() -> dict[str, str]:
    """Clear HuggingFace cache directory."""
    try:
        cache_path = Path("/home/pollen/.cache/huggingface")

        if cache_path.exists():
            shutil.rmtree(cache_path)
            logger.info(f"Cleared HuggingFace cache at {cache_path}")
            return {"status": "success", "message": "HuggingFace cache cleared"}
        else:
            logger.info(f"HuggingFace cache directory does not exist: {cache_path}")
            return {"status": "success", "message": "Cache directory already empty"}

    except Exception as e:
        logger.error(f"Failed to clear HuggingFace cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


@router.post("/reset-apps")
def reset_apps() -> dict[str, str]:
    """Remove applications virtual environment directory."""
    try:
        venv_path = Path("/venvs/apps_venv/")

        if venv_path.exists():
            shutil.rmtree(venv_path)
            logger.info(f"Removed applications virtual environment at {venv_path}")
            return {
                "status": "success",
                "message": "Applications virtual environment removed",
            }
        else:
            logger.info(
                f"Applications virtual environment directory does not exist: {venv_path}"
            )
            return {
                "status": "success",
                "message": "Virtual environment directory already empty",
            }

    except Exception as e:
        logger.error(f"Failed to clear applications virtual environment: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to clear virtual environment: {str(e)}"
        )
