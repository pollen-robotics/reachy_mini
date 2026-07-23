"""Daemon-related API routes."""

import logging
import threading

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from reachy_mini.daemon.app import bg_job_register
from reachy_mini.daemon.robot_app_lock import RobotAppLockStatus
from reachy_mini.io.protocol import DaemonStatus
from reachy_mini.utils.hardware_id import get_hardware_id
from reachy_mini.utils.robot_name import get_robot_name, set_robot_name

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


class RobotNameRequest(BaseModel):
    """Body for setting the robot display name."""

    name: str = Field(..., min_length=1, max_length=64)


class RobotNameResponse(BaseModel):
    """The persisted robot display name (``null`` when unset)."""

    name: str | None = None


@router.get("/robot-name")
async def get_robot_display_name() -> RobotNameResponse:
    """Return the persisted robot display name (``null`` when unset)."""
    return RobotNameResponse(name=get_robot_name())


@router.post("/robot-name")
async def set_robot_display_name(
    body: RobotNameRequest,
    request: Request,
    daemon: Daemon = Depends(get_daemon),
) -> RobotNameResponse:
    """Persist a new robot display name and apply it live (no restart).

    Persists the name to disk, then refreshes the advertised copies so the
    rename takes effect immediately: the daemon status + central relay (via
    ``Daemon.apply_robot_name``) and the LAN mDNS record. This is the REST
    entry point used by the BLE setup service; the WebRTC ``set_robot_name``
    command goes through the same live-apply path on the backend.
    """
    stored = set_robot_name(body.name)
    if stored is None:
        raise HTTPException(status_code=422, detail="Invalid robot name")

    # Live-apply: daemon status + central relay. Fail-safe so a wiring issue
    # can't turn a successful persist into an HTTP error.
    try:
        daemon.apply_robot_name(stored)
    except Exception as e:
        logging.getLogger(__name__).warning(
            "apply_robot_name failed after rename: %s", e
        )

    # Re-advertise the LAN mDNS record if the app lifespan wired it up.
    mdns = getattr(request.app.state, "mdns", None)
    if mdns is not None:
        try:
            mdns.update_name(stored)
        except Exception as e:
            logging.getLogger(__name__).warning(
                "mDNS update_name failed after rename: %s", e
            )

    return RobotNameResponse(name=stored)


@router.get("/hardware-id")
async def get_robot_hardware_id() -> dict[str, str | None]:
    """Robot-unique hardware ID — the Pollen audio device's USB serial.

    Returns ``{"hardware_id": "<serial>"}`` (or ``null`` when no robot
    is attached, e.g. on a developer machine). Same value across Lite
    and Wireless variants; same value across reboots and OS reinstalls.
    """
    return {"hardware_id": get_hardware_id()}


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
