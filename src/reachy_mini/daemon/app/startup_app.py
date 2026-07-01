"""Helpers for daemon startup-app installation and idle antenna launch."""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from reachy_mini.apps import SourceKind
from reachy_mini.apps.manager import AppManager
from reachy_mini.daemon.robot_app_lock import RobotAppLockState

if TYPE_CHECKING:
    from reachy_mini.daemon.daemon import Daemon

logger = logging.getLogger(__name__)

STARTUP_APP_ANTENNA_IDLE_POLL_INTERVAL_S = 0.1
STARTUP_APP_ANTENNA_BLOCKED_POLL_INTERVAL_S = 0.5


async def ensure_startup_app_installed(app_manager: AppManager, name: str) -> bool:
    """Install ``name`` from the catalog if it isn't already installed.

    Runs before wake-up so a long download/install doesn't leave the robot
    awake and idle. Installing needs no robot, only the apps venv. Best-effort:
    returns True if the app is ready to start, False (logged) on any failure.
    """
    try:
        installed = await app_manager.list_available_apps(SourceKind.INSTALLED)
        if any(a.name == name for a in installed):
            return True

        catalog = await app_manager.list_available_apps(SourceKind.HF_SPACE)
        match = next((a for a in catalog if a.name == name), None)
        if match is None:
            logger.error(f"Startup app '{name}' not installed and not in catalog")
            return False

        logger.info(f"Installing startup app: {name}")
        await app_manager.install_new_app(match, logger)
        return True
    except Exception as e:
        logger.error(f"Failed to install startup app '{name}': {e}")
        return False


async def start_startup_app(app_manager: AppManager, name: str) -> None:
    """Start ``name`` after wake-up. Best-effort: failures are logged, not raised."""
    try:
        logger.info(f"Auto-starting app: {name}")
        await app_manager.start_app(name)
    except Exception as e:
        logger.error(f"Failed to auto-start app '{name}': {e}")


def make_startup_app_launcher(
    app_manager: AppManager, name: str
) -> Callable[[], None]:
    """Build a one-shot, synchronous callback that launches the startup app.

    Wired into the backend's wake-up hook so the app starts after the robot
    first wakes, however that wake is triggered. Fires once and schedules the
    async start as a task so the wake sequence is not blocked.
    """
    launched = False

    def launch() -> None:
        nonlocal launched
        if launched:
            return
        launched = True
        asyncio.create_task(start_startup_app(app_manager, name))

    return launch


@dataclass
class AntennaTouchDetector:
    """Hysteresis detector for antenna touches relative to commanded target."""

    press_delta_rad: float = 0.25
    release_delta_rad: float = 0.10
    _armed: bool = False

    def reset(self) -> None:
        """Disarm until both antennas are back near their target positions."""
        self._armed = False

    def update(
        self,
        present: tuple[float, float],
        target: tuple[float, float],
    ) -> bool:
        """Return True once when either antenna is pushed away from target."""
        delta = max(abs(p - t) for p, t in zip(present, target))

        if not self._armed:
            if delta <= self.release_delta_rad:
                self._armed = True
            return False

        if delta >= self.press_delta_rad:
            self._armed = False
            return True

        return False


def _as_antenna_pair(values: Any) -> tuple[float, float] | None:
    """Convert a backend antenna vector to a typed pair, or None if missing."""
    if values is None:
        return None

    try:
        return (float(values[0]), float(values[1]))
    except (IndexError, TypeError, ValueError):
        return None


def _read_current_antenna_pair(backend: Any) -> tuple[float, float] | None:
    """Read cached antenna positions when available, falling back to backend API."""
    current = _as_antenna_pair(
        getattr(backend, "current_antenna_joint_positions", None)
    )
    if current is not None:
        return current

    return _as_antenna_pair(backend.get_present_antenna_joint_positions())


def _startup_app_slot_is_free(app_manager: AppManager, daemon: "Daemon") -> bool:
    """Return whether no managed local or remote app currently owns the robot."""
    return (
        not app_manager.is_app_running()
        and daemon.robot_app_lock.status().state == RobotAppLockState.FREE
    )


async def start_startup_app_if_idle(
    app_manager: AppManager,
    daemon: "Daemon",
    name: str,
) -> bool:
    """Start the startup app only if the managed app slot is still free."""
    if not _startup_app_slot_is_free(app_manager, daemon):
        return False

    try:
        logger.info(f"Auto-starting app from antenna touch: {name}")
        await app_manager.start_app(name, evict_remote=False)
        return True
    except Exception as e:
        logger.error(f"Failed to auto-start app '{name}' from antenna touch: {e}")
        return False


async def watch_antennas_for_startup_app(
    app_manager: AppManager,
    daemon: "Daemon",
    name: str,
    *,
    detector: AntennaTouchDetector | None = None,
    idle_poll_interval_s: float = STARTUP_APP_ANTENNA_IDLE_POLL_INTERVAL_S,
    blocked_poll_interval_s: float = STARTUP_APP_ANTENNA_BLOCKED_POLL_INTERVAL_S,
) -> None:
    """Start the startup app when an idle robot receives an antenna touch."""
    detector = detector or AntennaTouchDetector()
    detector.reset()

    while True:
        try:
            backend = daemon.backend
            if backend is None or not backend.ready.is_set():
                detector.reset()
                await asyncio.sleep(blocked_poll_interval_s)
                continue

            if not _startup_app_slot_is_free(app_manager, daemon):
                detector.reset()
                await asyncio.sleep(blocked_poll_interval_s)
                continue

            present = _read_current_antenna_pair(backend)
            target = _as_antenna_pair(backend.target_antenna_joint_positions)
            if present is None or target is None:
                detector.reset()
                await asyncio.sleep(idle_poll_interval_s)
                continue

            if detector.update(present, target):
                await start_startup_app_if_idle(app_manager, daemon, name)
                await asyncio.sleep(blocked_poll_interval_s)
                continue

            await asyncio.sleep(idle_poll_interval_s)
        except asyncio.CancelledError:
            logger.info("Startup app antenna watcher cancelled")
            raise
        except Exception as e:
            logger.warning(f"Startup app antenna watcher error: {e}")
            detector.reset()
            await asyncio.sleep(blocked_poll_interval_s)
