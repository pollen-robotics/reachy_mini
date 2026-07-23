"""App management for Reachy Mini."""

import asyncio
import logging
import os
import signal
import time
from collections import deque
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, AsyncIterator, Optional

import numpy as np
import psutil
from pydantic import BaseModel

from reachy_mini.daemon import startup_app_config
from reachy_mini.daemon.backend.robot import RobotBackend
from reachy_mini.io.protocol import MotorControlMode
from reachy_mini.utils.interpolation import distance_between_poses

from . import AppInfo, SourceKind
from .app import PARKED_ENV_VAR, PARKED_READY_SENTINEL
from .sources import hf_space, local_common_venv

if TYPE_CHECKING:
    from reachy_mini.daemon.daemon import Daemon

# Sleep-pose proximity in magic-mm (mm + deg), matching Backend.goto_sleep.
SLEEP_POSE_MAGIC_DISTANCE = 10.0


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


_PARK_KEEPER_POLL_S = 10.0
_PARKED_READY_TIMEOUT_S = 90.0  # park sentinel must appear within this window
_PARK_CRASH_WINDOW_S = 600.0  # 3 parked deaths in 10 min => stop trying


@dataclass
class ParkedApp:
    """A pre-warmed app process blocked before robot/media init.

    Holds NO robot lock, NO media pipelines, NO network sessions — only warm
    imports. Never stored in ``AppManager.current_app``, so it is invisible to
    ``is_app_running()``, the status endpoints and the JSON-RPC relay.
    """

    process: asyncio.subprocess.Process
    name: str
    spawned_at: float
    stderr_tail: "deque[str]"
    drain_task: "asyncio.Task[None] | None" = None
    ready: bool = False


def _get_catalog_app_key(app: AppInfo) -> str:
    """Return the Hugging Face space id used to deduplicate catalog entries."""
    value = app.extra.get("id")
    return value if isinstance(value, str) else ""


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
        self.parked_app: ParkedApp | None = None
        self._parked_keeper_task: "asyncio.Task[None] | None" = None
        self._parking_paused = 0
        self._parking_nudge = asyncio.Event()
        self._parked_crash_times: "deque[float]" = deque(maxlen=3)

    async def close(self) -> None:
        """Clean up the AppManager, stopping any running app."""
        if self._parked_keeper_task is not None:
            self._parked_keeper_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._parked_keeper_task
            self._parked_keeper_task = None
        await self._evict_parked_app("daemon shutting down")
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

    def get_running_app_url(self) -> str | None:
        """Return the running app's ``custom_app_url``, or ``None``.

        The JSON-RPC relay uses this to reach the app's ``/rpc`` endpoint. The
        URL is read from the app's ``main.py`` (same cheap scrape the launcher
        uses); the relay normalizes the host (``0.0.0.0`` -> ``127.0.0.1``).
        """
        if self.current_app is None or not self.is_app_running():
            return None
        return local_common_venv._get_custom_app_url_from_file(
            self.current_app.status.info.name,
            self.wireless_version,
            self.desktop_app_daemon,
        )

    def _build_app_env(self, *, parked: bool) -> "dict[str, str]":
        """Environment for an app subprocess.

        Scrub GStreamer env vars that the daemon's own `.venv/.../gstreamer_bundle.pth`
        set pointing at paths inside the daemon's .venv. The app runs in apps_venv and
        its own gstreamer_bundle.pth will set fresh values at Python startup. Leaving
        the parent's values in place is actively harmful:
          * Single-value vars like GST_REGISTRY_1_0 and GST_PLUGIN_SCANNER_1_0 get
            prepended to (via gstreamer_libs.setup_python_environment) producing a
            malformed `apps_venv_path:.venv_path` string that GStreamer can't parse.
          * The app ends up using .venv's plugin scanner binary and registry cache,
            which can mask issues specific to apps_venv's own gstreamer install.
        See pollen-robotics/reachy-mini-desktop-app#185.
        """
        app_env = os.environ.copy()
        for key in (
            "GST_PLUGIN_PATH_1_0",
            "GST_PLUGIN_SYSTEM_PATH_1_0",
            "GST_REGISTRY_1_0",
            "GST_PLUGIN_SCANNER_1_0",
            "GI_TYPELIB_PATH",
            "PYGI_DLL_DIRS",
            "XDG_DATA_DIRS",
            "XDG_CONFIG_DIRS",
        ):
            app_env.pop(key, None)
        if parked:
            app_env[PARKED_ENV_VAR] = "1"
        return app_env

    async def _spawn_app_subprocess(
        self, app_name: str, *, parked: bool
    ) -> asyncio.subprocess.Process:
        """Spawn an app subprocess (unbuffered), optionally in the parked state."""
        module_name = local_common_venv.get_app_module(
            app_name, self.wireless_version, self.desktop_app_daemon
        )
        python_path = local_common_venv.get_app_python(
            app_name, self.wireless_version, self.desktop_app_daemon
        )
        return await asyncio.create_subprocess_exec(
            str(python_path),
            "-u",  # Unbuffered stdout/stderr for real-time logging
            "-m",
            module_name,
            stdin=asyncio.subprocess.PIPE if parked else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._build_app_env(parked=parked),
        )

    # ------------------------------------------------------------------
    # Parked (pre-warmed) app machinery
    # ------------------------------------------------------------------

    async def _activate_parked_app(self) -> "asyncio.subprocess.Process | None":
        """Signal the parked process to resume; ``None`` => plain spawn."""
        parked = self.parked_app
        assert parked is not None
        self.parked_app = None
        if parked.drain_task is not None:
            parked.drain_task.cancel()
            with suppress(asyncio.CancelledError):
                await parked.drain_task
        if parked.process.returncode is not None:
            self.logger.warning(
                f"Parked app '{parked.name}' already exited; falling back to spawn"
            )
            return None
        try:
            assert parked.process.stdin is not None
            parked.process.stdin.write(b"activate\n")
            await parked.process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError, OSError):
            self._kill_process_tree(parked.process.pid)
            with suppress(ProcessLookupError):
                parked.process.kill()
            await parked.process.wait()
            return None
        self._parked_crash_times.clear()
        return parked.process

    async def _evict_parked_app(self, reason: str) -> None:
        """Discard the parked instance. Idempotent, never raises."""
        parked = self.parked_app
        if parked is None:
            return
        self.parked_app = None
        self.logger.info(f"Discarding parked app '{parked.name}' ({reason})")
        if parked.drain_task is not None:
            parked.drain_task.cancel()
            with suppress(asyncio.CancelledError):
                await parked.drain_task
        if parked.process.returncode is None:
            try:
                if parked.process.stdin is not None:
                    parked.process.stdin.close()  # EOF => cooperative exit(0)
                await asyncio.wait_for(parked.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._kill_process_tree(parked.process.pid)
                with suppress(ProcessLookupError):
                    parked.process.kill()
                await parked.process.wait()

    @asynccontextmanager
    async def pause_parking(self, reason: str) -> "AsyncIterator[None]":
        """No parked instance may exist while the apps venv is being mutated.

        Any install/update/remove can change code a parked process already
        imported (including the SDK itself); a stale parked instance would
        then activate with old code.
        """
        self._parking_paused += 1
        try:
            await self._evict_parked_app(reason)
            yield
        finally:
            self._parking_paused -= 1
            self._parked_crash_times.clear()  # fresh code deserves fresh attempts
            self._parking_nudge.set()

    def start_parked_app_keeper(self) -> None:
        """Start the background task that maintains one parked app when idle."""
        if self._parked_keeper_task is None:
            self._parked_keeper_task = asyncio.create_task(self._parked_app_keeper())

    def _parking_crash_looping(self) -> bool:
        return (
            len(self._parked_crash_times) == self._parked_crash_times.maxlen
            and time.monotonic() - self._parked_crash_times[0] < _PARK_CRASH_WINDOW_S
        )

    async def _parked_app_keeper(self) -> None:
        """Keep exactly one parked instance of the prewarm app alive when idle.

        A single poll/nudge loop instead of respawn hooks scattered over every
        stop/crash/update path — whatever killed the parked instance, the next
        iteration restores the invariant.
        """
        log = self.logger.getChild("parked")
        while True:
            try:
                self._parking_nudge.clear()
                parked = self.parked_app
                if (
                    parked is not None
                    and not parked.ready
                    and time.monotonic() - parked.spawned_at > _PARKED_READY_TIMEOUT_S
                ):
                    # Safety backstop: opted in but never parked => misbehaving.
                    await self._evict_parked_app("never reported parked-ready")
                    self._parked_crash_times.append(time.monotonic())
                elif (
                    parked is None
                    and self._parking_paused == 0
                    and not self.is_app_running()
                    and not self._parking_crash_looping()
                ):
                    name = startup_app_config.get_prewarm_app()
                    if name and local_common_venv.app_supports_parking(
                        name, self.wireless_version, self.desktop_app_daemon
                    ):
                        await self._spawn_parked_app(name)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.warning("parked-app keeper iteration failed", exc_info=True)
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(
                    self._parking_nudge.wait(), timeout=_PARK_KEEPER_POLL_S
                )

    async def _spawn_parked_app(self, name: str) -> None:
        log = self.logger.getChild("parked")
        log.info(f"Pre-warming app '{name}' (parked until start-app)")
        try:
            process = await self._spawn_app_subprocess(name, parked=True)
        except Exception:
            log.warning(f"Failed to pre-warm app '{name}'", exc_info=True)
            self._parked_crash_times.append(time.monotonic())
            return
        parked = ParkedApp(
            process=process,
            name=name,
            spawned_at=time.monotonic(),
            stderr_tail=deque(maxlen=10),
        )
        parked.drain_task = asyncio.create_task(self._drain_parked_process(parked))
        self.parked_app = parked

    async def _drain_parked_process(self, parked: ParkedApp) -> None:
        """Drain stdout/stderr (pipes must not fill) and notice unexpected death."""
        log = self.logger.getChild("parked")

        async def drain_stdout() -> None:
            assert parked.process.stdout is not None
            async for line in parked.process.stdout:
                text = line.decode().rstrip()
                if text == PARKED_READY_SENTINEL:
                    parked.ready = True
                    log.info(f"App '{parked.name}' is parked and ready")
                else:
                    log.debug(text)

        async def drain_stderr() -> None:
            assert parked.process.stderr is not None
            async for line in parked.process.stderr:
                text = line.decode().rstrip()
                parked.stderr_tail.append(text)
                log.debug(text)

        await asyncio.gather(drain_stdout(), drain_stderr())
        returncode = await parked.process.wait()
        if self.parked_app is parked:  # died while parked (not evicted/activated)
            self.parked_app = None
            self._parked_crash_times.append(time.monotonic())
            log.warning(
                f"Parked app '{parked.name}' exited (code {returncode}) before "
                "activation. Last stderr:\n" + "\n".join(parked.stderr_tail)
            )
            self._parking_nudge.set()

    async def start_app(
        self,
        app_name: str,
        *args: Any,
        evict_remote: bool = True,
        keep_remote: bool = False,
        **kwargs: Any,
    ) -> AppStatus:
        """Start the app as a subprocess.

        Raises RuntimeError if an app is already running. When
        ``evict_remote`` is false, a remote WebRTC session holding the app slot
        makes the start fail instead of being evicted.

        ``keep_remote`` (used when the start is *requested by* the connected
        remote client, e.g. the mobile app driving a conversation) takes the
        local-app slot **without** evicting the remote session: the client is a
        controller of this app, not a competitor for the robot, so its
        DataChannel must survive. Takes precedence over ``evict_remote``.
        """
        if self.is_app_running():
            raise RuntimeError("An app is already running")

        # Acquire the robot lock before spawning. If a remote WebRTC session
        # currently holds the robot, this notifies the relay so the remote
        # peer gets a clean endSession. Raises if another local app somehow
        # already holds it (belt-and-braces; is_app_running() above covers
        # the normal case, but the lock is the single source of truth
        # shared with the relay thread).
        if self.daemon is not None:
            if keep_remote:
                self.daemon.robot_app_lock.acquire_local_keeping_remote(app_name)
            elif evict_remote:
                await self.daemon.robot_app_lock.acquire_local_evicting_remote(
                    app_name
                )
            elif not self.daemon.robot_app_lock.try_acquire_local(app_name):
                raise RuntimeError("The robot app slot is already in use")

        try:
            process: "asyncio.subprocess.Process | None" = None
            if self.parked_app is not None:
                if self.parked_app.name == app_name:
                    process = await self._activate_parked_app()
                    if process is not None:
                        self.logger.getChild("runner").info(
                            f"Activated pre-warmed app {app_name}"
                        )
                else:
                    # A different app was requested: the parked instance holds
                    # no resources, but discard it so it can't linger.
                    await self._evict_parked_app(f"starting '{app_name}' instead")

            if process is None:
                self.logger.getChild("runner").info(f"Starting app {app_name}")
                process = await self._spawn_app_subprocess(app_name, parked=False)
        except Exception:
            # Release the lock if we failed before the subprocess was created —
            # monitor_process is the normal release path but it depends on
            # the subprocess existing.
            if self.daemon is not None:
                self.daemon.robot_app_lock.release_local(app_name)
            raise

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

            try:
                # Run both streams concurrently
                await asyncio.gather(log_stdout(), log_stderr())

                # Wait for process to complete
                returncode = await process.wait()

                # Update status based on exit code
                if self.current_app is not None:
                    if returncode == 0:
                        self.current_app.status.state = AppState.DONE
                        self.logger.getChild("runner").info(
                            f"App {app_name} finished"
                        )
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
            finally:
                # Always release the robot lock when the subprocess exits, no
                # matter how: clean exit, crash, SIGKILL, OOM, or cancellation
                # of this monitor task. Idempotent — stop_current_app's own
                # release is fine too.
                if self.daemon is not None:
                    self.daemon.robot_app_lock.release_local(app_name)
                # Restore the parked instance promptly after any app exit.
                self._parking_nudge.set()

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

        # Return to zero after an app stops, unless the app left it asleep.
        if self.daemon is not None and self.daemon.backend is not None:
            backend = self.daemon.backend
            _, _, dist_to_sleep = distance_between_poses(
                backend.get_current_head_pose(), backend.SLEEP_HEAD_POSE
            )
            if dist_to_sleep <= SLEEP_POSE_MAGIC_DISTANCE:
                # pose check only; ensure limp (idempotent)
                backend.set_motor_control_mode(MotorControlMode.Disabled)
                self.logger.getChild("runner").info(
                    "Robot is asleep; leaving it limp in the sleep pose."
                )
            else:
                if isinstance(backend, RobotBackend):
                    backend.enable_motors()

                try:
                    from reachy_mini.reachy_mini import (
                        INIT_ANTENNAS_JOINT_POSITIONS,
                        INIT_HEAD_POSE,
                    )

                    self.logger.getChild("runner").info(
                        "Returning robot to zero position"
                    )
                    await backend.goto_target(
                        head=INIT_HEAD_POSE,
                        antennas=np.array(INIT_ANTENNAS_JOINT_POSITIONS),
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
        """List available apps while preserving curated-only entries."""
        (
            hf_space_apps,
            dashboard_selection_apps,
            local_apps,
            installed_apps,
        ) = await asyncio.gather(
            self.list_available_apps(SourceKind.HF_SPACE),
            self.list_available_apps(SourceKind.DASHBOARD_SELECTION),
            self.list_available_apps(SourceKind.LOCAL),
            self.list_available_apps(SourceKind.INSTALLED),
        )

        catalog_apps: list[AppInfo] = []
        seen_catalog_apps: set[str] = set()

        for app in [*dashboard_selection_apps, *hf_space_apps]:
            app_key = _get_catalog_app_key(app)
            if not app_key:
                continue
            if app_key in seen_catalog_apps:
                continue
            seen_catalog_apps.add(app_key)
            catalog_apps.append(app)

        return [*catalog_apps, *local_apps, *installed_apps]

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
        # Any mutation of the shared apps_venv can change code a parked
        # process already imported (including the SDK itself) — never leave
        # a stale parked instance across an install/update/remove.
        async with self.pause_parking(f"installing '{app.name}'"):
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
        async with self.pause_parking(f"removing '{app_name}'"):
            success = await local_common_venv.uninstall_package(
                app_name,
                logger,
                wireless_version=self.wireless_version,
                desktop_app_daemon=self.desktop_app_daemon,
            )
        if success != 0:
            raise RuntimeError(f"Failed to uninstall app '{app_name}'")

    async def update_app(self, app_name: str, logger: logging.Logger) -> None:
        """Update an installed app by reinstalling it from HuggingFace.

        This preserves the original source info and reinstalls to get the latest version.

        Args:
            app_name: Name of the app to update.
            logger: Logger for progress output.

        Raises:
            RuntimeError: If app is running, not found, or update fails.

        """
        # Check if this app is currently running
        if (
            self.is_app_running()
            and self.current_app is not None
            and self.current_app.status.info.name == app_name
        ):
            raise RuntimeError(
                f"Cannot update '{app_name}' while it is running. Please stop it first."
            )

        # Try to get space_id from pip install info (works without stored metadata)
        from .sources.app_update_checker import get_hf_install_info

        hf_info = get_hf_install_info(
            app_name, self.wireless_version, self.desktop_app_daemon
        )

        # Fall back to stored metadata
        metadata = local_common_venv._load_app_metadata(app_name)

        space_id: str | None = None
        if hf_info:
            space_id = hf_info.space_id
        elif metadata:
            space_id = metadata.get("id")

        if not space_id:
            raise RuntimeError(
                f"App '{app_name}' was not installed from HuggingFace - cannot update"
            )

        # Create AppInfo for reinstallation
        app_info = AppInfo(
            name=app_name,
            description=metadata.get("cardData", {}).get("short_description", "")
            if metadata
            else "",
            url=f"https://huggingface.co/spaces/{space_id}",
            source_kind=SourceKind.HF_SPACE,
            extra=metadata if metadata else {"id": space_id},
        )

        logger.info(f"Updating app '{app_name}' from {space_id}")

        async with self.pause_parking(f"updating '{app_name}'"):
            # First uninstall the old version (handles package name changes)
            logger.info(f"Uninstalling old version of '{app_name}'")
            try:
                await local_common_venv.uninstall_package(
                    app_name,
                    logger,
                    wireless_version=self.wireless_version,
                    desktop_app_daemon=self.desktop_app_daemon,
                )
            except Exception as e:
                logger.warning(f"Could not uninstall old version: {e}")

            # Install the new version
            success = await local_common_venv.install_package(
                app_info,
                logger,
                wireless_version=self.wireless_version,
                desktop_app_daemon=self.desktop_app_daemon,
                force_reinstall=True,
            )

        if success != 0:
            raise RuntimeError(f"Failed to update app '{app_name}'")

        logger.info(f"Successfully updated '{app_name}'")
