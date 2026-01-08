"""App management for Reachy Mini."""

import asyncio
import logging
import os
import signal
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import psutil
from pydantic import BaseModel

from reachy_mini.daemon.backend.robot import RobotBackend

from . import AppInfo, SourceKind
from .sources import hf_space, local_common_venv

if TYPE_CHECKING:
    from reachy_mini.daemon.daemon import Daemon


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

    process: asyncio.subprocess.Process
    monitor_task: asyncio.Task[None]
    status: AppStatus


class AppManager:
    """Manager for Reachy Mini apps."""

    def __init__(
        self,
        wireless_version: bool = False,
        desktop_app_daemon: bool = False,
        daemon: Optional["Daemon"] = None,
    ) -> None:
        """Initialize the AppManager."""
        self.current_app = None  # type: RunningApp | None
        self.logger = logging.getLogger("reachy_mini.apps.manager")
        self.wireless_version = wireless_version
        self.desktop_app_daemon = desktop_app_daemon
        self.running_on_wireless = wireless_version
        self.daemon = daemon

    async def close(self) -> None:
        """Clean up the AppManager, stopping any running app."""
        if self.is_app_running():
            await self.stop_current_app()

    def _kill_process_tree(self, pid: int) -> None:
        """Kill a process and all its children recursively."""
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            for child in children:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
        except psutil.NoSuchProcess:
            pass

    # App lifecycle management
    # Only one app can be started at a time for now
    def is_app_running(self) -> bool:
        """Check if an app is currently running or stopping."""
        return self.current_app is not None and self.current_app.status.state in (
            AppState.STARTING,
            AppState.RUNNING,
            AppState.ERROR,
            AppState.STOPPING,
        )

    async def start_app(self, app_name: str, *args: Any, **kwargs: Any) -> AppStatus:
        """Start the app as a subprocess, raises RuntimeError if an app is already running."""
        if self.is_app_running():
            raise RuntimeError("An app is already running")

        # Get module name and Python path for subprocess execution
        module_name = local_common_venv.get_app_module(
            app_name, self.wireless_version, self.desktop_app_daemon
        )
        python_path = local_common_venv.get_app_python(
            app_name, self.wireless_version, self.desktop_app_daemon
        )

        # Launch app as subprocess with unbuffered output
        self.logger.getChild("runner").info(f"Starting app {app_name}")
        process = await asyncio.create_subprocess_exec(
            str(python_path),
            "-u",  # Unbuffered stdout/stderr for real-time logging
            "-m",
            module_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Create status and monitor task
        status = AppStatus(
            info=AppInfo(name=app_name, source_kind=SourceKind.INSTALLED),
            state=AppState.STARTING,
            error=None,
        )

        async def monitor_process() -> None:
            """Monitor the subprocess and update status."""
            assert self.current_app is not None
            assert process.stdout is not None
            assert process.stderr is not None

            # Update to RUNNING once process starts
            self.current_app.status.state = AppState.RUNNING
            self.logger.getChild("runner").info(f"App {app_name} is running")

            # Stream stdout
            async def log_stdout() -> None:
                assert process.stdout is not None
                async for line in process.stdout:
                    self.logger.getChild("runner").info(line.decode().rstrip())

            # Stream stderr - log as warning since it often contains errors/exceptions
            stderr_lines: list[str] = []

            async def log_stderr() -> None:
                assert process.stderr is not None
                async for line in process.stderr:
                    decoded = line.decode().rstrip()
                    stderr_lines.append(decoded)
                    # Check if line looks like an error or exception
                    if any(
                        keyword in decoded
                        for keyword in ["Error:", "Exception:", "Traceback", "ERROR"]
                    ):
                        self.logger.getChild("runner").error(decoded)
                    else:
                        # Many libraries write INFO/WARNING to stderr
                        self.logger.getChild("runner").warning(decoded)

            # Run both streams concurrently
            await asyncio.gather(log_stdout(), log_stderr())

            # Wait for process to complete
            returncode = await process.wait()

            # Update status based on exit code
            if self.current_app is not None:
                if returncode == 0:
                    self.current_app.status.state = AppState.DONE
                    self.logger.getChild("runner").info(f"App {app_name} finished")
                else:
                    self.current_app.status.state = AppState.ERROR
                    error_msg = "\n".join(stderr_lines[-10:])  # Last 10 lines
                    self.current_app.status.error = (
                        f"Process exited with code {returncode}\n{error_msg}"
                    )
                    self.logger.getChild("runner").error(
                        f"App {app_name} exited with code {returncode}. "
                        f"Last stderr output:\n{error_msg}"
                    )

        monitor_task = asyncio.create_task(monitor_process())

        self.current_app = RunningApp(
            process=process,
            monitor_task=monitor_task,
            status=status,
        )

        return self.current_app.status

    async def stop_current_app(self, timeout: float | None = 20.0) -> None:
        """Stop the current app subprocess."""
        if self.current_app is None or self.current_app.status.state in (
            AppState.DONE,
            AppState.STOPPING,
        ):
            raise RuntimeError("No app is currently running")

        assert self.current_app is not None

        self.current_app.status.state = AppState.STOPPING
        self.logger.getChild("runner").info(
            f"Stopping app {self.current_app.status.info.name}"
        )

        # Terminate subprocess
        process = self.current_app.process
        if process.returncode is None:
            # Send SIGINT to trigger KeyboardInterrupt (cross-platform, handled by template)
            try:
                if os.name == "posix":
                    # Unix/Linux/Mac: send SIGINT signal
                    os.kill(process.pid, signal.SIGINT)
                else:
                    # Windows: use CTRL_C_EVENT or fallback to terminate
                    process.terminate()

                # Wait for graceful shutdown
                await asyncio.wait_for(process.wait(), timeout=timeout)
                self.logger.getChild("runner").info("App stopped successfully")
            except asyncio.TimeoutError:
                # Force kill if timeout expires - also kill child processes
                self.logger.getChild("runner").warning(
                    "App did not stop within timeout, forcing termination"
                )
                self._kill_process_tree(process.pid)
                process.kill()
                await process.wait()

        # Cancel and wait for monitor task
        if not self.current_app.monitor_task.done():
            self.current_app.monitor_task.cancel()
            try:
                await self.current_app.monitor_task
            except asyncio.CancelledError:
                pass

        # Return robot to zero position after app stops
        if self.daemon is not None and self.daemon.backend is not None:
            if isinstance(self.daemon.backend, RobotBackend):
                self.daemon.backend.enable_motors()

            try:
                from reachy_mini.reachy_mini import INIT_HEAD_POSE

                self.logger.getChild("runner").info("Returning robot to zero position")
                await self.daemon.backend.goto_target(
                    head=INIT_HEAD_POSE,
                    antennas=np.array([0.0, 0.0]),
                    duration=1.0,
                )
            except Exception as e:
                self.logger.getChild("runner").warning(
                    f"Could not return to zero position: {e}"
                )

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
            return await hf_space.list_all_apps()
        elif source == SourceKind.DASHBOARD_SELECTION:
            return await hf_space.list_available_apps()
        elif source == SourceKind.INSTALLED:
            return await local_common_venv.list_available_apps(
                wireless_version=self.wireless_version,
                desktop_app_daemon=self.desktop_app_daemon,
            )
        elif source == SourceKind.LOCAL:
            return []
        else:
            raise NotImplementedError(f"Unknown source kind: {source}")

    async def install_new_app(self, app: AppInfo, logger: logging.Logger) -> None:
        """Install a new app by name."""
        success = await local_common_venv.install_package(
            app,
            logger,
            wireless_version=self.wireless_version,
            desktop_app_daemon=self.desktop_app_daemon,
        )
        if success != 0:
            raise RuntimeError(f"Failed to install app '{app.name}'")

    async def remove_app(self, app_name: str, logger: logging.Logger) -> None:
        """Remove an installed app by name."""
        success = await local_common_venv.uninstall_package(
            app_name,
            logger,
            wireless_version=self.wireless_version,
            desktop_app_daemon=self.desktop_app_daemon,
        )
        if success != 0:
            raise RuntimeError(f"Failed to uninstall app '{app_name}'")
