"""Daemon-related API routes."""

import asyncio
import logging
import re
import threading

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from reachy_mini.daemon.app import bg_job_register
from reachy_mini.daemon.app.config import (
    DEFAULT_ROBOT_NAME as _DEFAULT_ROBOT_NAME,
)
from reachy_mini.daemon.app.config import (
    get_persisted_robot_name,
)
from reachy_mini.daemon.robot_app_lock import RobotAppLockStatus
from reachy_mini.io.protocol import DaemonStatus

from ...daemon import Daemon
from ..dependencies import get_daemon

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/daemon",
)
busy_lock = threading.Lock()

# Single-writer guard for the rename flow: two parallel POSTs would race
# on persistence + ``setPeerStatus`` re-emit. Acquired only inside the
# rename route, never held across awaits longer than necessary.
_robot_name_lock = asyncio.Lock()

# Hard cap chosen so the result fits in mDNS TXT records, the WebSocket
# producer ``meta.name`` field central displays in its dashboard, and
# realistic UI labels. Lower bound rejects ``""`` and pure whitespace.
_ROBOT_NAME_MIN_LEN = 1
_ROBOT_NAME_MAX_LEN = 32

# ASCII printable + space, no control characters. We deliberately keep
# Unicode out for now because a) the BLE LocalName layer can't fit
# arbitrary UTF-8 in a 31-byte advertisement, b) mDNS service names need
# to round-trip through DNS-safe encoding, and c) the daemon HTTP layer
# itself is fine with anything but downstream consumers (logs, dashboards)
# get noisier with non-ASCII. Easy to relax later if real users ask.
_ROBOT_NAME_PATTERN = re.compile(r"^[\x20-\x7E]+$")


class RobotNameResponse(BaseModel):
    """Current robot name and where it was sourced from."""

    name: str
    # ``"persisted"`` -> read from ``daemon.json`` on disk;
    # ``"cli"`` -> ``--robot-name`` flag at daemon launch overrides config;
    # ``"default"`` -> nobody set anything yet, mobile app should prompt.
    source: str


class SetRobotNameRequest(BaseModel):
    """Body for ``POST /api/daemon/robot-name``."""

    name: str = Field(
        min_length=_ROBOT_NAME_MIN_LEN,
        max_length=_ROBOT_NAME_MAX_LEN,
        description="Human-readable label for the robot.",
    )


def _validate_robot_name(raw: str) -> str:
    """Trim, validate, and return a name ready for persistence.

    Raises ``HTTPException(422)`` with a precise reason on failure so
    the mobile app can render a useful inline error.
    """
    name = raw.strip()
    if len(name) < _ROBOT_NAME_MIN_LEN or len(name) > _ROBOT_NAME_MAX_LEN:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Robot name must be {_ROBOT_NAME_MIN_LEN}-{_ROBOT_NAME_MAX_LEN} "
                f"characters after trimming whitespace."
            ),
        )
    if not _ROBOT_NAME_PATTERN.match(name):
        raise HTTPException(
            status_code=422,
            detail=(
                "Robot name must only contain printable ASCII characters "
                "(letters, digits, spaces, common punctuation)."
            ),
        )
    return name


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


@router.get("/version")
async def get_daemon_version() -> dict[str, str]:
    """Return the daemon's package version and a compatibility marker.

    Mobile and remote clients call this early during the connection
    handshake to detect a feature mismatch (e.g. an older daemon that
    doesn't ship a new endpoint yet) and surface a soft warning rather
    than failing later with a cryptic 404.

    The endpoint is deliberately additive: it returns a dict so we can
    grow it (build sha, kinematics engine, ...) without breaking older
    clients that only look up ``version``.
    """
    from reachy_mini import __version__

    return {
        "version": __version__,
        # Bumped whenever the HTTP surface gains/loses an endpoint or
        # a route's response shape changes in a non-additive way.
        # Mobile clients can rely on `api_revision >= N` to know a
        # feature is reachable, regardless of the package version.
        # Revision history:
        #   "1" - initial mobile-app-facing surface (http_proxy, ws_proxy)
        #   "2" - GET / POST /api/daemon/robot-name + persisted daemon.json
        "api_revision": "2",
    }


@router.get("/robot-name", response_model=RobotNameResponse)
async def get_robot_name(
    request: Request, daemon: Daemon = Depends(get_daemon)
) -> RobotNameResponse:
    """Return the current robot name and tell the client where it came from.

    The ``source`` field lets the mobile app decide whether to prompt
    the user for a name on first connection: when the value is
    ``"default"`` the user has never customised it, so we surface the
    naming screen; when it is ``"persisted"`` or ``"cli"`` we trust the
    existing label and skip the prompt.
    """
    args = getattr(request.app.state, "args", None)
    cli_override = args is not None and getattr(args, "_robot_name_cli_explicit", False)

    persisted = get_persisted_robot_name()

    if cli_override:
        source = "cli"
    elif persisted and persisted == daemon.robot_name:
        source = "persisted"
    elif daemon.robot_name == _DEFAULT_ROBOT_NAME and not persisted:
        source = "default"
    else:
        # Live rename in flight, or persisted vs in-memory disagree.
        # ``persisted`` wins as the source label since we just wrote it
        # there before mutating ``daemon.robot_name``.
        source = "persisted" if persisted else "default"

    return RobotNameResponse(name=daemon.robot_name, source=source)


@router.post("/robot-name", response_model=DaemonStatus)
async def set_robot_name(
    body: SetRobotNameRequest,
    daemon: Daemon = Depends(get_daemon),
    request: Request = None,  # type: ignore[assignment]
) -> DaemonStatus:
    """Rename the robot live and persist the new label across restarts.

    The new name is propagated to:

    - in-memory ``Daemon.robot_name`` and ``DaemonStatus.robot_name``;
    - ``~/.config/reachy_mini/daemon.json`` (atomic write);
    - the central signaling relay (live ``setPeerStatus`` re-emit, no
      reconnect required), so the HF fleet listing reflects the change
      within a few seconds without dropping any in-flight WebRTC session;
    - the mDNS service registration, when one is active (LAN listeners
      see the goodbye + new advertisement).

    Idempotent: a no-op when ``name`` already matches the current value
    (after trimming). Concurrency-safe via an ``asyncio.Lock``: parallel
    requests are serialised and the second one observes the new state
    as a no-op.

    Returns the full ``DaemonStatus`` snapshot so the mobile app can
    refresh its connection summary in a single round-trip.
    """
    new_name = _validate_robot_name(body.name)

    async with _robot_name_lock:
        if new_name == daemon.robot_name:
            # Idempotent fast-path: don't touch disk, don't churn central.
            return daemon.status()

        await daemon.set_robot_name(new_name)

        # Forward to the mDNS handle owned by the FastAPI lifespan, when
        # one exists (it doesn't on test harnesses that build the app
        # without going through ``create_app``'s lifespan).
        mdns = getattr(request.app.state, "mdns", None) if request else None
        if mdns is not None:
            try:
                mdns.update_robot_name(new_name)
            except Exception as exc:
                logger.warning("[daemon.rename] mDNS update failed: %s", exc)

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
