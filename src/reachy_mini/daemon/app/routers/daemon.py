"""Daemon-related API routes."""

import logging
import threading

from fastapi import APIRouter, Depends, HTTPException, Request

from reachy_mini.daemon.app import bg_job_register

from ...daemon import Daemon, DaemonStatus
from ..dependencies import get_daemon

router = APIRouter(
    prefix="/daemon",
)
busy_lock = threading.Lock()


@router.post("/start")
async def start_daemon(
    request: Request,
    wake_up: bool,
    daemon: Daemon = Depends(get_daemon),
) -> dict[str, str]:
    """Start the daemon."""
    if busy_lock.locked():
        raise HTTPException(status_code=409, detail="Daemon is busy.")

    async def start(logger: logging.Logger) -> None:
        with busy_lock:
            await daemon.start(
                sim=request.app.state.args.sim,
                scene=request.app.state.args.scene,
                headless=request.app.state.args.headless,
                wake_up_on_start=wake_up,
            )

    job_id = bg_job_register.run_command("daemon-start", start)
    return {"job_id": job_id}


@router.post("/stop")
async def stop_daemon(
    goto_sleep: bool, daemon: Daemon = Depends(get_daemon)
) -> dict[str, str]:
    """Stop the daemon, optionally putting the robot to sleep."""
    if busy_lock.locked():
        raise HTTPException(status_code=409, detail="Daemon is busy.")

    async def stop(logger: logging.Logger) -> None:
        with busy_lock:
            await daemon.stop(goto_sleep_on_stop=goto_sleep)

    job_id = bg_job_register.run_command("daemon-stop", stop)
    return {"job_id": job_id}


@router.post("/restart")
async def restart_daemon(
    request: Request, daemon: Daemon = Depends(get_daemon)
) -> dict[str, str]:
    """Restart the daemon."""
    if busy_lock.locked():
        raise HTTPException(status_code=409, detail="Daemon is busy.")

    async def restart(logger: logging.Logger) -> None:
        with busy_lock:
            await daemon.restart(
                sim=request.app.state.args.sim,
                scene=request.app.state.args.scene,
            )

    job_id = bg_job_register.run_command("daemon-restart", restart)
    return {"job_id": job_id}


@router.get("/status")
async def get_daemon_status(daemon: Daemon = Depends(get_daemon)) -> DaemonStatus:
    """Get the current status of the daemon."""
    return daemon.status()
