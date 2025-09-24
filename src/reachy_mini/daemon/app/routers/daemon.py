"""Daemon-related API routes."""

import asyncio

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)

from ...daemon import Daemon, DaemonState, DaemonStatus
from ..dependencies import get_daemon, ws_get_daemon

router = APIRouter(
    prefix="/daemon",
)


@router.post("/start")
async def start_daemon(
    request: Request,
    wake_up: bool,
    daemon: Daemon = Depends(get_daemon),
):
    """Start the daemon."""
    state = daemon.status().state
    if state in (DaemonState.STARTING, DaemonState.RUNNING, DaemonState.STOPPING):
        raise HTTPException(status_code=400, detail=f"Daemon is {state}.")

    coro = daemon.start(
        sim=request.app.state.args.sim,
        scene=request.app.state.args.scene,
        headless=request.app.state.args.headless,
        wake_up_on_start=wake_up,
    )
    asyncio.create_task(coro)


@router.post("/stop")
async def stop_daemon(goto_sleep: bool, daemon: Daemon = Depends(get_daemon)):
    """Stop the daemon, optionally putting the robot to sleep."""
    state = daemon.status().state
    if state in (
        DaemonState.NOT_INITIALIZED,
        DaemonState.STARTING,
        DaemonState.STOPPING,
        DaemonState.STOPPED,
    ):
        raise HTTPException(status_code=400, detail=f"Daemon is {state}.")

    coro = daemon.stop(goto_sleep_on_stop=goto_sleep)
    asyncio.create_task(coro)


@router.post("/restart")
async def restart_daemon(request: Request, daemon: Daemon = Depends(get_daemon)):
    """Restart the daemon."""
    coro = daemon.restart(
        sim=request.app.state.args.sim,
        scene=request.app.state.args.scene,
    )
    asyncio.create_task(coro)


@router.get("/status")
async def get_daemon_status(daemon: Daemon = Depends(get_daemon)) -> DaemonStatus:
    """Get the current status of the daemon."""
    return daemon.status()


@router.websocket("/status/ws/notifications")
async def ws_full_state(
    websocket: WebSocket,
    daemon: Daemon = Depends(ws_get_daemon),
):
    """WebSocket endpoint to stream the full state of the robot."""
    await websocket.accept()

    msg_queue = asyncio.Queue()

    async def handler(status: DaemonStatus):
        await msg_queue.put(status)

    uuid = daemon.register_status_notification(handler)

    try:
        while True:
            status = await msg_queue.get()
            await websocket.send_text(status.model_dump_json())

    except asyncio.CancelledError:
        pass
    except WebSocketDisconnect:
        daemon.unregister_status_notification(uuid)
