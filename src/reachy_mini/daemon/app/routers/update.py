"""Update router for Reachy Mini Daemon API.

This module provides endpoints to check for updates, start updates, and monitor update status.
"""

import logging
import threading

from fastapi import APIRouter, HTTPException, WebSocket

from reachy_mini.daemon.app import bg_job_register
from reachy_mini.daemon.app.bg_job_register import JobInfo
from reachy_mini.utils.wireless_version.update import update_reachy_mini
from reachy_mini.utils.wireless_version.update_available import is_update_available

router = APIRouter(prefix="/update")
busy_lock = threading.Lock()


@router.get("/available")
def available(pre_release: bool = False) -> dict[str, dict[str, bool]]:
    """Check if an update is available for Reachy Mini Wireless."""
    if busy_lock.locked():
        raise HTTPException(status_code=400, detail="Update is in progress")

    try:
        return {
            "update": {
                "reachy_mini": is_update_available("reachy_mini", pre_release),
            }
        }
    except ConnectionError:
        raise HTTPException(status_code=503, detail="Unable to check for updates")


@router.post("/start")
def start_update(pre_release: bool = False) -> dict[str, str]:
    """Start the update process for Reachy Mini Wireless version."""
    if busy_lock.locked():
        raise HTTPException(status_code=400, detail="Update already in progress")

    if not is_update_available("reachy_mini", pre_release):
        raise HTTPException(status_code=400, detail="No update available")

    async def update_wrapper(logger: logging.Logger) -> None:
        with busy_lock:
            await update_reachy_mini(pre_release, logger)

    job_uuid = bg_job_register.run_command(
        "update_reachy_mini",
        update_wrapper,
    )

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
