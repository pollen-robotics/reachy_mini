"""Update router for Reachy Mini Daemon API.

This module provides endpoints to check for updates, start updates, and monitor update status.
"""

import logging
import threading

import requests
from fastapi import APIRouter, HTTPException, WebSocket

from reachy_mini.daemon.app import bg_job_register
from reachy_mini.daemon.app.bg_job_register import JobInfo
from reachy_mini.utils.wireless_version.update import update_reachy_mini
from reachy_mini.utils.wireless_version.update_available import (
    get_install_source,
    get_local_version,
    get_pypi_version,
    is_update_available,
)

router = APIRouter(prefix="/update")
busy_lock = threading.Lock()


@router.get("/install-source")
def install_source() -> dict[str, str]:
    """Get current install source: version, source type (pypi/git/editable), and git ref if applicable."""
    return get_install_source("reachy_mini")


@router.get("/available")
def available(pre_release: bool = False) -> dict[str, dict[str, dict[str, bool | str]]]:
    """Check if an update is available for Reachy Mini Wireless."""
    if busy_lock.locked():
        raise HTTPException(status_code=400, detail="Update is in progress")

    current_version = str(get_local_version("reachy_mini"))

    try:
        is_available = is_update_available("reachy_mini", pre_release)
        available = str(get_pypi_version("reachy_mini", pre_release))
    except (ConnectionError, requests.exceptions.ConnectionError):
        is_available = False
        available = "unknown"

    return {
        "update": {
            "reachy_mini": {
                "is_available": is_available,
                "current_version": current_version,
                "available_version": available,
            }
        }
    }


@router.post("/start")
def start_update(pre_release: bool = False) -> dict[str, str]:
    """Start the update process for Reachy Mini Wireless version."""
    if busy_lock.locked():
        raise HTTPException(status_code=400, detail="Update already in progress")

    if not is_update_available("reachy_mini", pre_release):
        raise HTTPException(status_code=400, detail="No update available")

    async def update_wrapper(logger: logging.Logger) -> None:
        with busy_lock:
            await update_reachy_mini(logger, pre_release=pre_release)

    job_uuid = bg_job_register.run_command(
        "update_reachy_mini",
        update_wrapper,
    )

    return {"job_id": job_uuid}


# --- Git ref installation (install any tag/branch from GitHub) ---

GITHUB_REPO = "pollen-robotics/reachy_mini"


@router.get("/validate-ref")
def validate_ref(git_ref: str) -> dict[str, bool | str]:
    """Check if a git ref (tag/branch) exists on GitHub."""
    if not git_ref.strip():
        raise HTTPException(status_code=400, detail="Git ref cannot be empty")

    url = f"https://api.github.com/repos/{GITHUB_REPO}/commits/{git_ref}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return {"valid": True, "ref": git_ref}
        return {"valid": False, "ref": git_ref, "error": f"Ref '{git_ref}' not found"}
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"GitHub API error: {e}")


@router.post("/start-from-ref")
def start_update_from_ref(git_ref: str) -> dict[str, str]:
    """Start update from a specific git ref (tag or branch)."""
    if busy_lock.locked():
        raise HTTPException(status_code=400, detail="Update already in progress")

    if not git_ref.strip():
        raise HTTPException(status_code=400, detail="Git ref cannot be empty")

    async def update_wrapper(logger: logging.Logger) -> None:
        with busy_lock:
            await update_reachy_mini(logger, git_ref=git_ref)

    job_uuid = bg_job_register.run_command("update_reachy_mini", update_wrapper)
    return {"job_id": job_uuid}


@router.get("/info")
def get_update_info(job_id: str) -> JobInfo:
    """Get the info of an update job."""
    try:
        return bg_job_register.get_info(job_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket, job_id: str) -> None:
    """WebSocket endpoint to stream update logs in real time."""
    await websocket.accept()
    await bg_job_register.ws_poll_info(websocket, job_id)
    await websocket.close()
