"""Update router for Reachy Mini Daemon API.

This module provides endpoints to check for updates, start updates, and monitor update status.
"""

import asyncio
import logging
import threading
import uuid
from dataclasses import dataclass
from threading import Thread

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from reachy_mini.utils.wireless_version.update import update_reachy_mini
from reachy_mini.utils.wireless_version.update_available import is_update_available

router = APIRouter(prefix="/update")
busy_lock = threading.Lock()


class JobStatus(BaseModel):
    """Pydantic model for install job status."""

    command: str
    status: str
    logs: list[str]


@dataclass
class JobHandler:
    """Handler for background jobs."""

    status: JobStatus
    new_log_evt: dict[str, asyncio.Event]


jobs: dict[str, JobHandler] = {}


@router.get("/available")
def available() -> dict[str, dict[str, bool]]:
    """Check if an update is available for Reachy Mini Wireless."""
    return {
        "update": {
            "reachy_mini": is_update_available("reachy_mini"),
        }
    }


@router.post("/start")
def start_update() -> dict[str, str]:
    """Start the update process for Reachy Mini Wireless version."""
    if busy_lock.locked():
        raise HTTPException(status_code=400, detail="Update already in progress")

    if not is_update_available("reachy_mini"):
        raise HTTPException(status_code=400, detail="No update available")

    job_id = str(uuid.uuid4())
    jobs[job_id] = JobHandler(
        status=JobStatus(command="update_reachy_mini", status="pending", logs=[]),
        new_log_evt={},
    )

    async def wrapper():
        with busy_lock:
            logger = logging.getLogger(f"update_logs_{job_id}")

            class JobLogger(logging.Handler):
                def emit(self, record: logging.LogRecord) -> None:
                    log_entry = self.format(record)
                    jobs[job_id].status.logs.append(log_entry)
                    for evt in jobs[job_id].new_log_evt.values():
                        evt.set()

            logger.setLevel(logging.INFO)
            logger.addHandler(JobLogger())

            jobs[job_id].status.status = "in_progress"
            try:
                await update_reachy_mini(logger)
                jobs[job_id].status.status = "done"
                logger.info("Update completed successfully.")
            except Exception as e:
                logger.error(f"Update failed: {e}")
                jobs[job_id].status.status = "failed"
                logger.error(f"Update failed: {e}")

    if busy_lock.locked():
        raise HTTPException(status_code=400, detail="Update already in progress")

    t = Thread(target=lambda: asyncio.run(wrapper()))
    t.start()
    # TODO: wait for thread to start to make sure the lock is acquired

    return {"job_id": job_id}


@router.get("/status")
def get_update_status(job_id: str) -> dict[str, str | list[str]]:
    """Get the status and logs of an update job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job ID not found")
    return {
        "status": jobs[job_id].status.status,
        "logs": jobs[job_id].status.logs,
    }


@router.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket, job_id: str):
    """WebSocket endpoint to stream update logs in real time."""
    await websocket.accept()
    last_log_len = 0

    job = jobs.get(job_id)
    if not job:
        await websocket.send_json({"error": "Job ID not found"})
        await websocket.close()
        return

    assert job is not None
    ws_uuid = str(uuid.uuid4())

    try:
        job.new_log_evt[ws_uuid] = asyncio.Event()

        while True:
            await job.new_log_evt[ws_uuid].wait()
            job.new_log_evt[ws_uuid].clear()

            new_logs = job.status.logs[last_log_len:]

            if new_logs:
                for log_entry in new_logs:
                    await websocket.send_text(log_entry)
                last_log_len = len(job.status.logs)
                await websocket.send_json(
                    {
                        "command": job.status.command,
                        "status": job.status.status,
                        "logs": new_logs,
                    }
                )
                if job.status.status in ("done", "failed"):
                    await websocket.close()
                    break
    except WebSocketDisconnect:
        pass
    finally:
        job.new_log_evt.pop(ws_uuid, None)
        await websocket.close()
