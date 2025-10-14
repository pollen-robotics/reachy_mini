"""Update router for Reachy Mini Daemon API.

This module provides endpoints to check for updates, start updates, and monitor update status.
"""

import threading

from fastapi import APIRouter, BackgroundTasks, HTTPException, WebSocket

from reachy_mini.daemon.app import background_tasks_wrapper
from reachy_mini.daemon.app.background_tasks_wrapper import JobInfo
from reachy_mini.utils.wireless_version.update import update_reachy_mini
from reachy_mini.utils.wireless_version.update_available import is_update_available

router = APIRouter(prefix="/update")
busy_lock = threading.Lock()


@router.get("/available")
def available() -> dict[str, dict[str, bool]]:
    """Check if an update is available for Reachy Mini Wireless."""
    if busy_lock.locked():
        raise HTTPException(status_code=400, detail="Update is in progress")

    return {
        "update": {
            "reachy_mini": is_update_available("reachy_mini"),
        }
    }


@router.post("/start")
def start_update(background_tasks: BackgroundTasks) -> dict[str, str]:
    """Start the update process for Reachy Mini Wireless version."""
    if busy_lock.locked():
        raise HTTPException(status_code=400, detail="Update already in progress")

    if not is_update_available("reachy_mini"):
        raise HTTPException(status_code=400, detail="No update available")

    async def update_wrapper(logger) -> None:
        with busy_lock:
            await update_reachy_mini(logger)

    job_uuid = background_tasks_wrapper.run_command(
        background_tasks,
        "update_reachy_mini",
        update_wrapper,
    )

    return {"job_id": job_uuid}


@router.get("/info")
def get_update_info(job_id: str) -> JobInfo:
    """Get the info of an update job."""
    try:
        return background_tasks_wrapper.get_info(job_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket, job_id: str) -> None:
    """WebSocket endpoint to stream update logs in real time."""
    await websocket.accept()
    await background_tasks_wrapper.ws_poll_info(websocket, job_id)
    await websocket.close()
