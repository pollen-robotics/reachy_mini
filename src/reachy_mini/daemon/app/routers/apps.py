"""Apps router for apps management."""

import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import BaseModel

from reachy_mini.apps import AppInfo, SourceKind
from reachy_mini.apps.manager import AppManager, AppStatus
from reachy_mini.daemon.app.dependencies import get_app_manager

router = APIRouter(prefix="/apps")


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


@router.get("/list-available/{source_kind}")
async def list_available_apps(
    source_kind: SourceKind,
    app_manager: "AppManager" = Depends(get_app_manager),
) -> list[AppInfo]:
    """List available apps (including not installed)."""
    return await app_manager.list_available_apps(source_kind)


@router.get("/list-available")
async def list_all_available_apps(
    app_manager: "AppManager" = Depends(get_app_manager),
) -> list[AppInfo]:
    """List all available apps (including not installed)."""
    return await app_manager.list_all_available_apps()


@router.post("/install")
async def install_app(
    app_info: AppInfo,
    background_tasks: BackgroundTasks,
    app_manager: "AppManager" = Depends(get_app_manager),
) -> dict[str, str]:
    """Install a new app by its info (background, returns job_id)."""
    job_id = start_bg_job(
        "install", background_tasks, app_manager.install_new_app, app_info
    )
    return {"job_id": job_id}


@router.post("/remove/{app_name}")
async def remove_app(
    app_name: str,
    background_tasks: BackgroundTasks,
    app_manager: "AppManager" = Depends(get_app_manager),
) -> dict[str, str]:
    """Remove an installed app by its name (background, returns job_id)."""
    job_id = start_bg_job("remove", background_tasks, app_manager.remove_app, app_name)
    return {"job_id": job_id}


@router.get("/job-status/{job_id}")
async def job_status(job_id: str) -> JobStatus:
    """Get status/logs for a job."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.status


# WebSocket route for live job status/logs
@router.websocket("/ws/apps-manager/{job_id}")
async def ws_apps_manager(websocket: WebSocket, job_id: str) -> None:
    """WebSocket route to stream live job status/logs for a job, sending updates as soon as new logs are available."""
    await websocket.accept()
    last_log_len = 0

    job = jobs.get(job_id)
    if not job:
        await websocket.send_json({"error": "Job not found"})
        await websocket.close()
        return

    assert job is not None  # for mypy

    ws_uuid = str(uuid.uuid4())

    try:
        job.new_log_evt[ws_uuid] = asyncio.Event()
        while True:
            await job.new_log_evt[ws_uuid].wait()
            job.new_log_evt[ws_uuid].clear()
            new_logs = job.status.logs[last_log_len:]
            last_log_len = len(job.status.logs)
            await websocket.send_json(
                {
                    "command": job.status.command,
                    "status": job.status.model_dump_json(),
                    "logs": new_logs,
                }
            )
            if job.status.status.startswith("done") or job.status.status.startswith(
                "error"
            ):
                await websocket.close()
                break
    except WebSocketDisconnect:
        pass

    finally:
        if ws_uuid in job.new_log_evt:
            del job.new_log_evt[ws_uuid]


@router.post("/start-app/{app_name}")
async def start_app(
    app_name: str,
    app_manager: "AppManager" = Depends(get_app_manager),
) -> AppStatus:
    """Start an app by its name."""
    try:
        return await app_manager.start_app(app_name)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/restart-current-app")
async def restart_app(
    app_manager: "AppManager" = Depends(get_app_manager),
) -> AppStatus:
    """Restart the currently running app."""
    try:
        return await app_manager.restart_current_app()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/stop-current-app")
async def stop_app(
    app_manager: "AppManager" = Depends(get_app_manager),
) -> None:
    """Stop the currently running app."""
    try:
        return await app_manager.stop_current_app()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/current-app-status")
async def current_app_status(
    app_manager: "AppManager" = Depends(get_app_manager),
) -> AppStatus | None:
    """Get the status of the currently running app, if any."""
    return await app_manager.current_app_status()


def start_bg_job(
    command: str,
    background_tasks: BackgroundTasks,
    coro_func: Callable[..., Awaitable[None]],
    *args: Any,
) -> str:
    """Start a background job, with a custom logger and return its job_id."""
    job_id = str(uuid.uuid4())
    jobs[job_id] = JobHandler(
        status=JobStatus(
            command=command,
            status="pending",
            logs=[],
        ),
        new_log_evt={},
    )

    async def run_command() -> None:
        jobs[job_id].status.status = "running"

        class JobLogger(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                jobs[job_id].status.logs.append(self.format(record))
                for ws in jobs[job_id].new_log_evt.values():
                    ws.set()

        logger = logging.getLogger(f"logs_job_{job_id}")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        logger.addHandler(JobLogger())
        try:
            await coro_func(*args, logger=logger)
            jobs[job_id].status.status = "done"
            logger.info(f"Job '{command}' completed successfully")
        except Exception as e:
            jobs[job_id].status.status = f"error: {e}"
            logger.error(f"Job '{command}' failed with error: {e}")

    background_tasks.add_task(run_command)
    return job_id
