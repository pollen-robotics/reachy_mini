"""App management for Reachy Mini."""

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from importlib.metadata import entry_points
from threading import Thread
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel

from . import AppInfo, SourceKind
from .sources import hf_space, local_common_venv

if TYPE_CHECKING:
    from .app import ReachyMiniApp


class AppState(str, Enum):
    """Status of a running app."""

    STARTING = "starting"
    RUNNING = "running"
    DONE = "done"
    STOPPING = "stopping"
    ERROR = "error"


class AppStatus(BaseModel):
    """Status of an app."""

    info: AppInfo
    state: AppState
    error: str | None = None


@dataclass
class RunningApp:
    """Information about a running app."""

    app: "ReachyMiniApp"
    thread: Thread
    status: AppStatus


class AppManager:
    """Manager for Reachy Mini apps."""

    def __init__(self) -> None:
        """Initialize the AppManager."""
        self.current_app = None  # type: RunningApp | None
        self.logger = logging.getLogger("reachy_mini.apps.manager")

    async def close(self) -> None:
        """Clean up the AppManager, stopping any running app."""
        if self.is_app_running():
            await self.stop_current_app()

    # App lifecycle management
    # Only one app can be started at a time for now
    def is_app_running(self) -> bool:
        """Check if an app is currently running."""
        return self.current_app is not None and self.current_app.status.state in (
            AppState.STARTING,
            AppState.RUNNING,
            AppState.ERROR,
        )

    async def start_app(self, app_name: str, *args: Any, **kwargs: Any) -> AppStatus:
        """Start the app, raises RuntimeError if an app is already running."""
        if self.is_app_running():
            raise RuntimeError("An app is already running")

        (ep,) = entry_points(group="reachy_mini_apps", name=app_name)
        app = ep.load()()

        def wrapped_run() -> None:
            assert self.current_app is not None

            try:
                self.current_app.status.state = AppState.RUNNING
                self.logger.getChild("runner").info(f"App {app_name} is running")
                app.wrapped_run(*args, **kwargs)
                self.current_app.status.state = AppState.DONE
                self.logger.getChild("runner").info(f"App {app_name} finished")
            except Exception as e:
                self.logger.getChild("runner").error(
                    f"An error occurred in the app {app_name}: {e}"
                )
                self.logger.getChild("runner").error(
                    f"Exception details: '{app.error}'",
                )
                self.current_app.status.state = AppState.ERROR
                self.current_app.status.error = str(app.error)

        self.current_app = RunningApp(
            status=AppStatus(
                info=AppInfo(name=app_name, source_kind=SourceKind.INSTALLED),
                state=AppState.STARTING,
                error=None,
            ),
            app=app,
            thread=Thread(target=wrapped_run),
        )
        self.logger.getChild("runner").info(f"Starting app {app_name}")
        self.current_app.thread.start()

        return self.current_app.status

    async def stop_current_app(self, timeout: float | None = 5.0) -> None:
        """Stop the current app."""
        if not self.is_app_running():
            raise RuntimeError("No app is currently running")

        assert self.current_app is not None

        self.current_app.status.state = AppState.STOPPING
        self.logger.getChild("runner").info(
            f"Stopping app {self.current_app.status.info.name}"
        )
        self.current_app.app.stop()
        self.current_app.thread.join(timeout)

        if self.current_app.thread.is_alive():
            self.logger.getChild("runner").warning(
                "The app did not stop within the timeout"
            )
        else:
            self.logger.getChild("runner").info("App stopped successfully")

        self.current_app = None

    async def restart_current_app(self) -> AppStatus:
        """Restart the current app."""
        if not self.is_app_running():
            raise RuntimeError("No app is currently running")

        assert self.current_app is not None

        app_info = self.current_app.status.info

        await self.stop_current_app()
        await self.start_app(app_info.name)

        return self.current_app.status

    async def current_app_status(self) -> Optional[AppStatus]:
        """Get the current status of the app."""
        if self.current_app is not None:
            return self.current_app.status
        return None

    # Apps management interface
    async def list_all_available_apps(self) -> list[AppInfo]:
        """List available apps (parallel async)."""
        results = await asyncio.gather(
            *[self.list_available_apps(kind) for kind in SourceKind]
        )
        return sum(results, [])

    async def list_available_apps(self, source: SourceKind) -> list[AppInfo]:
        """List available apps for given source kind."""
        if source == SourceKind.HF_SPACE:
            return await hf_space.list_available_apps()
        elif source == SourceKind.INSTALLED:
            return await local_common_venv.list_available_apps()
        elif source == SourceKind.LOCAL:
            return []
        else:
            raise NotImplementedError(f"Unknown source kind: {source}")

    async def install_new_app(self, app: AppInfo, logger: logging.Logger) -> None:
        """Install a new app by name."""
        success = await local_common_venv.install_package(app, logger)
        if success != 0:
            raise RuntimeError(f"Failed to install app '{app.name}'")

    async def remove_app(self, app_name: str, logger: logging.Logger) -> None:
        """Remove an installed app by name."""
        success = await local_common_venv.uninstall_package(app_name, logger)
        if success != 0:
            raise RuntimeError(f"Failed to uninstall app '{app_name}'")
