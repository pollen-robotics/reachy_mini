"""Daemon-related API routes."""

import logging
import threading

from fastapi import APIRouter, Depends, HTTPException, Request

from reachy_mini.daemon.app import bg_job_register
from reachy_mini.daemon.robot_app_lock import RobotAppLockStatus
from reachy_mini.io.protocol import DaemonStatus

from ...daemon import Daemon
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
                serialport=request.app.state.args.serialport,
                scene=request.app.state.args.scene,
                localhost_only=request.app.state.args.localhost_only,
                wake_up_on_start=wake_up,
                check_collision=request.app.state.args.check_collision,
                kinematics_engine=request.app.state.args.kinematics_engine,
                headless=request.app.state.args.headless,
                use_audio=not request.app.state.args.no_media,
                hardware_config_filepath=request.app.state.args.hardware_config_filepath,
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
            await daemon.restart()

    job_id = bg_job_register.run_command("daemon-restart", restart)
    return {"job_id": job_id}


@router.get("/status")
async def get_daemon_status(daemon: Daemon = Depends(get_daemon)) -> DaemonStatus:
    """Get the current status of the daemon."""
    return daemon.status()


@router.get("/robot-app-lock-status")
async def get_robot_app_lock_status(
    daemon: Daemon = Depends(get_daemon),
) -> RobotAppLockStatus:
    """Return the current state of the robot's managed-app lock.

    The daemon's single source of truth for which managed app (if any)
    currently holds the robot:

    - ``free``: no managed app holds the slot.
    - ``local_app``: a Python app launched via AppManager is running.
      ``holder_name`` is the app name.
    - ``remote_session``: a remote WebRTC client is connected via the
      central signaling relay. ``holder_name`` is a generic ``"remote"``
      placeholder (the real consumer app name lives on the central
      server and is surfaced via its own ``/api/robot-status``).

    Note that SDK clients talking to the daemon directly bypass this
    lock; it only reflects the two *managed* app entry points.

    Intended for UI layers (desktop app, dashboard) that want to render
    a busy/free indicator without trying to open a session.
    """
    return daemon.robot_app_lock.status()
