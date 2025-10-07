"""Daemon-related API routes."""

from fastapi import APIRouter, Depends, Request

from ...daemon import Daemon, DaemonStatus
from ..dependencies import get_daemon

router = APIRouter(
    prefix="/daemon",
)


@router.post("/start")
async def start_daemon(
    request: Request,
    wake_up: bool,
    daemon: Daemon = Depends(get_daemon),
) -> DaemonStatus:
    """Start the daemon."""
    await daemon.start(
        sim=request.app.state.args.sim,
        scene=request.app.state.args.scene,
        headless=request.app.state.args.headless,
        wake_up_on_start=wake_up,
    )
    return daemon.status()


@router.post("/stop")
async def stop_daemon(
    goto_sleep: bool, daemon: Daemon = Depends(get_daemon)
) -> DaemonStatus:
    """Stop the daemon, optionally putting the robot to sleep."""
    await daemon.stop(goto_sleep_on_stop=goto_sleep)
    return daemon.status()


@router.post("/restart")
async def restart_daemon(
    request: Request, daemon: Daemon = Depends(get_daemon)
) -> DaemonStatus:
    """Restart the daemon."""
    await daemon.restart(
        sim=request.app.state.args.sim,
        scene=request.app.state.args.scene,
    )
    return daemon.status()


@router.get("/status")
async def get_daemon_status(daemon: Daemon = Depends(get_daemon)) -> DaemonStatus:
    """Get the current status of the daemon."""
    return daemon.status()
