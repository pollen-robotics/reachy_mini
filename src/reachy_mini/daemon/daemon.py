"""Daemon for Reachy Mini robot.

This module provides a daemon that runs a backend for either a simulated Reachy Mini using Mujoco or a real Reachy Mini robot using a serial connection.
It includes methods to start, stop, and restart the daemon, as well as to check its status.
It also provides a command-line interface for easy interaction.
"""

import asyncio
import json
import logging
import time
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from threading import Event, Thread
from typing import Any, Optional

from reachy_mini.daemon.robot_app_lock import RobotAppLock
from reachy_mini.daemon.utils import (
    SimulationMode,
    find_serial_port,
    get_ip_address,
)
from reachy_mini.io.protocol import DaemonState, DaemonStatus, MotorControlMode
from reachy_mini.io.ws_server import WSServer
from reachy_mini.tools.reflash_motors import reflash_motors_if_needed

from .backend.mockup_sim import MockupSimBackend
from .backend.mujoco import MujocoBackend
from .backend.robot import RobotBackend

# Central signaling relay for WebRTC (optional)
_central_relay_task: Optional[asyncio.Task[Any]] = None


class Daemon:
    """Daemon for simulated or real Reachy Mini robot.

    Runs the server with the appropriate backend (Mujoco for simulation or RobotBackend for real hardware).
    """

    def __init__(
        self,
        log_level: str = "INFO",
        robot_name: str = "reachy_mini",
        wireless_version: bool = False,
        desktop_app_daemon: bool = False,
        no_media: bool = False,
        sim_mode: SimulationMode = SimulationMode.NONE,
    ) -> None:
        """Initialize the Reachy Mini daemon."""
        self.log_level = log_level
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.log_level)

        self.robot_name = robot_name

        self.wireless_version = wireless_version
        self.desktop_app_daemon = desktop_app_daemon
        self.no_media = no_media

        self.backend: "RobotBackend | MujocoBackend | MockupSimBackend | None" = None
        # Get package version
        try:
            package_version = version("reachy_mini")
            self.logger.info(f"Daemon version: {package_version}")
        except PackageNotFoundError:
            package_version = None
            self.logger.warning("Could not determine daemon version")

        from reachy_mini.utils.hardware_id import get_hardware_id

        self._status = DaemonStatus(
            robot_name=robot_name,
            state=DaemonState.NOT_INITIALIZED,
            wireless_version=wireless_version,
            desktop_app_daemon=desktop_app_daemon,
            simulation_enabled=None,
            mockup_sim_enabled=None,
            no_media=no_media,
            backend_status=None,
            error=None,
            wlan_ip=None,
            version=package_version,
            hardware_id=get_hardware_id(),
        )
        self.ws_server: "WSServer | None" = None
        self.backend_run_thread: "Thread | None" = None
        self._thread_event_publish_status = Event()

        # Single source of truth for which managed app (local Python app or
        # remote WebRTC client) currently holds the robot's app slot. Shared
        # with AppManager and the central signaling relay. SDK clients that
        # talk to the daemon directly bypass this lock — it only coordinates
        # the two managed app entry points.
        # See reachy_mini/daemon/robot_app_lock.py.
        self.robot_app_lock = RobotAppLock()

        self._media_server: Optional["GstMediaServer"] = (
            None  # GstMediaServer when media is enabled
        )
        self._media_released = False
        if not no_media:
            from reachy_mini.media.media_server import GstMediaServer

            try:
                self._media_server = GstMediaServer(log_level, sim_mode=sim_mode)
                self._status.camera_specs_name = self._media_server.camera_specs.name
            except Exception as e:
                self.logger.error(f"Failed to initialize media server: {e}")
                self._media_server = None
        else:
            self.logger.info(
                "Media disabled (--no-media). No camera, audio, or media server."
            )

    def __del__(self) -> None:
        """Destructor to ensure proper cleanup."""
        self.logger.debug("Cleaning up Daemon resources...")
        if self._media_server is not None:
            self._media_server.stop()
            self._media_server.close()
            self._media_server = None

    @property
    def media_released(self) -> bool:
        """Whether media hardware has been released for direct access."""
        return self._media_released

    async def release_media(self) -> None:
        """Release camera and audio hardware so clients can access them directly.

        Stops the GstMediaServer pipeline and central signalling relay.
        Idempotent: no-op if already released or no media server.
        """
        if self._media_released or self._media_server is None:
            return

        self.logger.info("Releasing media hardware for direct access...")
        self._media_server.stop()
        await self._stop_central_signaling_relay()
        self._media_released = True
        self._status.media_released = True
        self.logger.info("Media hardware released.")

    async def acquire_media(self) -> None:
        """Re-acquire camera and audio hardware after a release.

        Restarts the GstMediaServer pipeline and central signalling relay.
        Idempotent: no-op if not currently released or no media server.
        """
        if not self._media_released or self._media_server is None:
            return

        self.logger.info("Re-acquiring media hardware...")
        self._media_server.start()
        await self._start_central_signaling_relay()
        self._media_released = False
        self._status.media_released = False
        self.logger.info("Media hardware re-acquired.")

    def restart_media_pipeline(self) -> None:
        """Rebuild the media-server pipeline so an input-device change applies.

        The mic source is baked in at ``start()``, so the pipeline must be
        rebuilt for a new selection to take effect. Runs on a background thread
        (blocking GStreamer work) and briefly interrupts any WebRTC stream.
        No-op when there is no media server or media has been released to an app.
        """
        media_server = self._media_server
        if media_server is None or self._media_released:
            self.logger.info(
                "Media pipeline restart skipped: no media server, or an app holds "
                "the audio (media released). The device change is saved and applies "
                "on the next app launch / when the daemon re-acquires media."
            )
            return

        def _run() -> None:
            try:
                self.logger.info(
                    "Restarting media pipeline to apply audio device change..."
                )
                media_server.stop()
                media_server.start()
                self.logger.info("Media pipeline restarted.")
            except Exception as e:
                self.logger.error(f"Media pipeline restart failed: {e}")

        Thread(target=_run, daemon=True, name="media-device-restart").start()

    async def _start_central_signaling_relay(self) -> None:
        """Start the central signaling relay for remote WebRTC access."""
        global _central_relay_task

        if not self._media_server:
            return

        try:
            from huggingface_hub import get_token

            hf_token = get_token()
        except Exception as e:
            self.logger.debug(f"No HF token available, central signaling disabled: {e}")
            return

        if not hf_token:
            self.logger.info("No HF token found, central signaling relay disabled")
            return

        try:
            from reachy_mini.media.central_signaling_relay import start_central_relay

            # ``transport`` reflects how a remote client physically
            # reaches this daemon: the Wireless variant runs autonomously
            # on its onboard CM4 and is reached over Wi-Fi (or LAN); the
            # Lite variant is plugged into the user's desktop and the
            # daemon runs there, reached over USB. Keyed off
            # ``wireless_version`` so the badging follows the robot SKU.
            transport = "wifi" if self.wireless_version else "usb"

            self.logger.info("Starting central signaling relay...")
            relay = await start_central_relay(
                hf_token=hf_token,
                robot_name=self.robot_name,
                transport=transport,
                robot_app_lock=self.robot_app_lock,
            )
            self.logger.info("Central signaling relay started")

            # Wire the negotiation watchdog: when GstMediaServer
            # detects a stuck WebRTC negotiation (typically libnice
            # frozen mid-CHECKING), it forwards the failure to central
            # via the relay so the JS client gets a typed
            # ``endSession`` instead of a spinner-on-blank-page UX.
            # See `media_server.set_session_failed_handler`.
            if self._media_server is not None:
                self._media_server.set_session_failed_handler(
                    relay.notify_peer_session_failed
                )
        except Exception as e:
            self.logger.warning(f"Failed to start central signaling relay: {e}")

    async def _stop_central_signaling_relay(self) -> None:
        """Stop the central signaling relay."""
        try:
            from reachy_mini.media.central_signaling_relay import stop_central_relay

            await stop_central_relay()
            self.logger.info("Central signaling relay stopped")
        except Exception as e:
            self.logger.debug(f"Error stopping central signaling relay: {e}")

    async def start(
        self,
        sim: bool = False,
        mockup_sim: bool = False,
        serialport: str = "auto",
        scene: str = "empty",
        wake_up_on_start: bool = True,
        check_collision: bool = False,
        kinematics_engine: str = "AnalyticalKinematics",
        headless: bool = False,
        use_audio: bool = True,  # kept for backward compat, overridden by no_media
        hardware_config_filepath: str | None = None,
        on_wake_up_callback: Callable[[], None] | None = None,
    ) -> "DaemonState":
        """Start the Reachy Mini daemon.

        Args:
            sim (bool): If True, run in simulation mode using Mujoco. Defaults to False.
            mockup_sim (bool): If True, run in lightweight simulation mode (no MuJoCo). Defaults to False.
            serialport (str): Serial port for real motors. Defaults to "auto", which will try to find the port automatically.
            scene (str): Name of the scene to load in simulation mode ("empty" or "minimal"). Defaults to "empty".
            wake_up_on_start (bool): If True, wake up Reachy Mini on start. Defaults to True.
            check_collision (bool): If True, enable collision checking. Defaults to False.
            kinematics_engine (str): Kinematics engine to use. Defaults to "AnalyticalKinematics".
            headless (bool): If True, run Mujoco in headless mode (no GUI). Defaults to False.
            use_audio (bool): If True, enable audio. Defaults to True.
            hardware_config_filepath (str | None): Path to the hardware configuration YAML file. Defaults to None.
            on_wake_up_callback (Callable[[], None] | None): Fired once each time the robot finishes waking up. Defaults to None.

        Returns:
            DaemonState: The current state of the daemon after attempting to start it.

        """
        if self._status.state == DaemonState.RUNNING:
            self.logger.warning("Daemon is already running.")
            return self._status.state

        self.logger.info(
            f"Daemon start parameters: sim={sim}, mockup_sim={mockup_sim}, serialport={serialport}, scene={scene}, wake_up_on_start={wake_up_on_start}, check_collision={check_collision}, kinematics_engine={kinematics_engine}, headless={headless}, hardware_config_filepath={hardware_config_filepath}"
        )

        # mockup-sim behaves exactly like a real robot for apps (they open webcam directly)
        # Only MuJoCo (--sim) sets simulation_enabled=True (streams video via UDP)
        self._status.simulation_enabled = sim
        self._status.mockup_sim_enabled = mockup_sim

        # The wireless version binds all interfaces and advertises its LAN
        # address; loopback-only configurations have no meaningful wlan_ip.
        if self.wireless_version:
            self._status.wlan_ip = get_ip_address()

        # When no_media is set, override use_audio to False
        effective_use_audio = use_audio and not self.no_media

        self._start_params = {
            "sim": sim,
            "mockup_sim": mockup_sim,
            "serialport": serialport,
            "headless": headless,
            "use_audio": effective_use_audio,
            "scene": scene,
        }

        self.logger.info("Starting Reachy Mini daemon...")
        self._status.state = DaemonState.STARTING

        try:
            self.backend = self._setup_backend(
                wireless_version=self.wireless_version,
                sim=sim,
                mockup_sim=mockup_sim,
                serialport=serialport,
                scene=scene,
                check_collision=check_collision,
                kinematics_engine=kinematics_engine,
                headless=headless,
                use_audio=effective_use_audio,
                hardware_config_filepath=hardware_config_filepath,
            )

            self.ws_server = WSServer(backend=self.backend)
            self.ws_server.start()
            self._thread_publish_status = Thread(
                target=self._publish_status, daemon=True
            )
            self._thread_publish_status.start()

            def backend_wrapped_run() -> None:
                assert self.backend is not None, (
                    "Backend should be initialized before running."
                )

                try:
                    self.backend.wrapped_run()
                except Exception as e:
                    self.logger.error(f"Backend encountered an error: {e}")
                    self._status.state = DaemonState.ERROR
                    self._status.error = str(e)
                    if self.ws_server is not None:
                        self.ws_server.stop()
                    self.backend = None

            self.backend_run_thread = Thread(target=backend_wrapped_run)
            self.backend_run_thread.start()

            if not self.backend.ready.wait(timeout=2.0):
                self.logger.error(
                    "Backend is not ready after 2 seconds. Some error occurred."
                )
                self._status.state = DaemonState.ERROR
                self._status.error = self.backend.error
                return self._status.state

            if self._media_server and not self._media_released:
                if self.backend is not None:
                    self.backend.setup_media_server(self._media_server)
                    self.backend.set_restart_daemon_callback(self._spawn_webrtc_restart)
                    self.backend.set_start_update_callback(self._spawn_webrtc_update)
                self._media_server.start()

                # Start central signaling relay for remote WebRTC access
                await self._start_central_signaling_relay()

            # Wire the wake-up hook before any wake can fire (on-start below, or
            # later via button/REST on the wireless unit, which boots asleep).
            if on_wake_up_callback is not None:
                self.backend.set_on_wake_up_callback(on_wake_up_callback)

            if wake_up_on_start:
                try:
                    self.logger.info("Waking up Reachy Mini...")
                    self.backend.set_motor_control_mode(MotorControlMode.Enabled)
                    await self.backend.wake_up()
                except Exception as e:
                    self.logger.error(f"Error while waking up Reachy Mini: {e}")
                    self._status.state = DaemonState.ERROR
                    self._status.error = str(e)
                    return self._status.state
                except KeyboardInterrupt:
                    self.logger.warning("Wake up interrupted by user.")
                    self._status.state = DaemonState.STOPPING
                    return self._status.state

            if self._status.state != DaemonState.ERROR:
                self.logger.info("Daemon started successfully.")
                self._status.state = DaemonState.RUNNING
            else:
                self.logger.error("Daemon started with errors.")

        except Exception as e:
            self._status.state = DaemonState.ERROR
            self._status.error = str(e)
            self.logger.error(f"Failed to start daemon: {e}")

        return self._status.state

    async def stop(self, goto_sleep_on_stop: bool = True) -> "DaemonState":
        """Stop the Reachy Mini daemon.

        Args:
            goto_sleep_on_stop (bool): If True, put Reachy Mini to sleep on stop. Defaults to True.

        Returns:
            DaemonState: The current state of the daemon after attempting to stop it.

        Note:
            The relay releases its remote hold on ``self.robot_app_lock`` via
            ``relay.stop()``. A local-app hold is *not* force-released here
            because the daemon is going down; the lock object dies with the
            process. If restart-in-place is ever added, force-release the
            ``LOCAL_APP`` state here before restart.

        """
        if self._status.state == DaemonState.STOPPED:
            self.logger.warning("Daemon is already stopped.")
            return self._status.state

        if self.backend is None:
            self.logger.info("Daemon backend is not initialized.")
            if self.ws_server is not None:
                self.ws_server.stop()
            self._status.state = DaemonState.STOPPED
            return self._status.state

        try:
            if self._status.state in (DaemonState.STOPPING, DaemonState.ERROR):
                goto_sleep_on_stop = False

            self.logger.info("Stopping Reachy Mini daemon...")
            self._status.state = DaemonState.STOPPING
            self.backend.is_shutting_down = True
            self._thread_event_publish_status.set()

            if self._media_server and not self._media_released:
                # Stop pipeline (NULL) to release camera/audio hardware so
                # external tools (rpicam-still, etc.) can access them.
                # start() will rebuild the pipeline from scratch.
                self._media_server.stop()
                # Stop the central signaling relay
                await self._stop_central_signaling_relay()

            if goto_sleep_on_stop:
                try:
                    self.logger.info("Putting Reachy Mini to sleep...")
                    self.backend.set_motor_control_mode(MotorControlMode.Enabled)
                    await self.backend.goto_sleep()
                    self.backend.set_motor_control_mode(MotorControlMode.Disabled)
                except Exception as e:
                    self.logger.error(f"Error while putting Reachy Mini to sleep: {e}")
                    self._status.state = DaemonState.ERROR
                    self._status.error = str(e)
                except KeyboardInterrupt:
                    self.logger.warning("Sleep interrupted by user.")
                    self._status.state = DaemonState.STOPPING

            self.backend.should_stop.set()
            if self.backend_run_thread is not None:
                self.backend_run_thread.join(timeout=5.0)
                if self.backend_run_thread.is_alive():
                    self.logger.warning(
                        "Backend did not stop in time, forcing shutdown."
                    )
                    self._status.state = DaemonState.ERROR

            self.backend.close()
            self.backend.ready.clear()

            # WS server must be closed after backend finishes to publish all data
            if self.ws_server is not None:
                self.ws_server.stop()

            if self._status.state != DaemonState.ERROR:
                self.logger.info("Daemon stopped successfully.")
                self._status.state = DaemonState.STOPPED
        except Exception as e:
            self.logger.error(f"Error while stopping the daemon: {e}")
            self._status.state = DaemonState.ERROR
            self._status.error = str(e)
        except KeyboardInterrupt:
            self.logger.warning("Daemon already stopping...")

        if self.backend is not None:
            backend_status = self.backend.get_status()
            if backend_status.error:
                self._status.state = DaemonState.ERROR

            self.backend = None

        return self._status.state

    async def restart(
        self,
        sim: Optional[bool] = None,
        mockup_sim: Optional[bool] = None,
        serialport: Optional[str] = None,
        scene: Optional[str] = None,
        headless: Optional[bool] = None,
        use_audio: Optional[bool] = None,
        wake_up_on_start: Optional[bool] = None,
        goto_sleep_on_stop: Optional[bool] = None,
    ) -> "DaemonState":
        """Restart the Reachy Mini daemon.

        Args:
            sim (bool): If True, run in simulation mode using Mujoco. Defaults to None (uses the previous value).
            mockup_sim (bool): If True, run in lightweight simulation mode (no MuJoCo). Defaults to None (uses the previous value).
            serialport (str): Serial port for real motors. Defaults to None (uses the previous value).
            scene (str): Name of the scene to load in simulation mode ("empty" or "minimal"). Defaults to None (uses the previous value).
            headless (bool): If True, run Mujoco in headless mode (no GUI). Defaults to None (uses the previous value).
            use_audio (bool): If True, enable audio. Defaults to None (uses the previous value).
            wake_up_on_start (bool): If True, wake up Reachy Mini on start. Defaults to None (don't wake up).
            goto_sleep_on_stop (bool): If True, put Reachy Mini to sleep on stop. Defaults to None (don't go to sleep).

        Returns:
            DaemonState: The current state of the daemon after attempting to restart it.

        """
        if self._status.state == DaemonState.STOPPED:
            self.logger.warning("Daemon is not running.")
            return self._status.state

        if self._status.state in (DaemonState.RUNNING, DaemonState.ERROR):
            self.logger.info("Restarting Reachy Mini daemon...")

            await self.stop(
                goto_sleep_on_stop=goto_sleep_on_stop
                if goto_sleep_on_stop is not None
                else False
            )
            params: dict[str, Any] = {
                "sim": sim if sim is not None else self._start_params["sim"],
                "mockup_sim": mockup_sim
                if mockup_sim is not None
                else self._start_params["mockup_sim"],
                "serialport": serialport
                if serialport is not None
                else self._start_params["serialport"],
                "scene": scene if scene is not None else self._start_params["scene"],
                "headless": headless
                if headless is not None
                else self._start_params["headless"],
                "use_audio": use_audio
                if use_audio is not None
                else self._start_params["use_audio"],
                "wake_up_on_start": wake_up_on_start
                if wake_up_on_start is not None
                else False,
            }

            return await self.start(**params)

        raise NotImplementedError(
            "Restarting is only supported when the daemon is in RUNNING or ERROR state."
        )

    def _spawn_webrtc_restart(self) -> None:
        """Run ``self.restart()`` on a fresh thread.

        Called from the backend's WebRTC ``restart_daemon`` handler to
        recover from a broken backend state (e.g. a dead motor
        controller) without forcing a ``systemctl restart`` or a REST
        round-trip. Mirrors ``bg_job_register.run_command``: a daemon
        thread starts its own asyncio loop, runs ``restart()``, then
        exits. We can't reuse the FastAPI background-job registry from
        here because the backend has no FastAPI ``Request`` context.

        Returns immediately so the caller can flush its DataChannel
        ack before the WebRTC stack is torn down.
        """
        if self._status.state == DaemonState.STOPPED:
            self.logger.warning(
                "Ignoring WebRTC restart_daemon: daemon already stopped."
            )
            return

        def _run() -> None:
            try:
                asyncio.run(self.restart())
            except Exception as e:
                self.logger.error(f"WebRTC restart_daemon failed: {e}")

        Thread(target=_run, daemon=True, name="webrtc-restart-daemon").start()

    def _spawn_webrtc_update(self, pre_release: bool = False) -> Optional[str]:
        """Run a PyPI update of the daemon on a fresh thread.

        Called from the backend's WebRTC ``start_update`` handler. Mirrors
        ``_spawn_webrtc_restart`` and the REST ``/update/start`` endpoint:
        ``update_reachy_mini`` upgrades the package and ends with a
        ``systemctl restart``. We run it on a daemon thread with its own
        asyncio loop so the caller can flush its DataChannel ack before the
        WebRTC stack is torn down by the restart.

        Returns a refusal reason string when the update is declined so the
        caller can ack an error instead of ``ok`` (and never spawns the
        thread in that case); returns ``None`` once the job is accepted.
        The same guards as the REST endpoint are enforced:

        * wireless-only -- the self-update path lives in
          ``utils.wireless_version`` and has no meaning on a Lite, which is
          updated through its host machine;
        * an update must actually be available, otherwise we would force a
          needless reinstall + restart of the running version;
        * the shared ``busy_lock`` from ``routers/update.py`` is acquired so
          a WebRTC update and a concurrent REST update can never run at the
          same time and corrupt the venv.

        Progress is fanned out to every connected client as
        ``update_progress`` broadcasts (one per log line), mirroring the
        REST ``WS /update/ws/logs`` stream. A successful update restarts
        the daemon before a ``done`` event can be sent, so consumers infer
        success from the transport teardown + reconnect.
        """
        if self._status.state == DaemonState.STOPPED:
            self.logger.warning("Ignoring WebRTC start_update: daemon already stopped.")
            return "Daemon is stopped"

        # The PyPI self-update only exists on the wireless robot; a Lite is
        # updated via its host machine, so reject it here rather than failing
        # deep inside `update_reachy_mini`.
        if not self.wireless_version:
            return "start_update is only supported on Reachy Mini Wireless"

        # Local import: keeps the wireless-update dependency chain out of
        # the daemon module's import-time graph (matches routers/update.py).
        from reachy_mini.daemon.app.routers.update import busy_lock
        from reachy_mini.utils.wireless_version.update import update_reachy_mini
        from reachy_mini.utils.wireless_version.update_available import (
            is_update_available,
        )

        # Share the REST router's lock so a WebRTC-triggered update and an
        # HTTP-triggered one are mutually exclusive (a concurrent pip install
        # into the same venv would corrupt it). Acquire non-blocking up front
        # so we can reject immediately; on the accepted path the worker thread
        # owns the lock and releases it, on every rejection we release here.
        if not busy_lock.acquire(blocking=False):
            return "Update already in progress"

        try:
            if not is_update_available("reachy_mini", pre_release):
                busy_lock.release()
                return "No update available"
        except Exception as e:
            busy_lock.release()
            self.logger.error(f"WebRTC start_update availability check failed: {e}")
            return f"Update availability check failed: {e}"

        backend = self.backend

        def _broadcast(
            status: str, *, line: str | None = None, error: str | None = None
        ) -> None:
            if backend is None:
                return
            payload: dict[str, Any] = {"type": "update_progress", "status": status}
            if line is not None:
                payload["line"] = line
            if error is not None:
                payload["error"] = error
            try:
                backend.broadcast_to_all_clients(json.dumps(payload))
            except Exception as e:
                self.logger.warning(f"update_progress broadcast failed: {e}")

        class _ProgressHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                _broadcast("in_progress", line=self.format(record))

        # Dedicated logger: streams update output to clients while still
        # propagating to the daemon's journalctl logs (propagate=True).
        update_logger = logging.getLogger("reachy_mini.webrtc_update")
        update_logger.setLevel(logging.INFO)
        update_logger.handlers.clear()
        update_logger.addHandler(_ProgressHandler())

        def _run() -> None:
            try:
                asyncio.run(update_reachy_mini(update_logger, pre_release=pre_release))
                _broadcast("done")
            except Exception as e:
                self.logger.error(f"WebRTC start_update failed: {e}")
                _broadcast("failed", error=str(e))
            finally:
                # On success `update_reachy_mini` triggers a `systemctl
                # restart` that kills this process before we get here; the
                # release only matters when the update failed without
                # restarting, freeing the lock for a retry.
                busy_lock.release()

        Thread(target=_run, daemon=True, name="webrtc-start-update").start()
        return None

    def status(self) -> "DaemonStatus":
        """Get the current status of the Reachy Mini daemon."""
        if self.backend is not None:
            self._status.backend_status = self.backend.get_status()

            assert self._status.backend_status is not None, (
                "Backend status should not be None after backend initialization."
            )

            if self._status.backend_status.error:
                self._status.state = DaemonState.ERROR
            self._status.error = self._status.backend_status.error
        else:
            self._status.backend_status = None

        return self._status

    def _publish_status(self) -> None:
        self._thread_event_publish_status.clear()
        while self._thread_event_publish_status.is_set() is False:
            json_str = self.status().model_dump_json()
            if self.ws_server is None:
                self.logger.warning(
                    f"WS server not initialized, cannot publish status: {json_str}"
                )
            else:
                self.ws_server.publish_status(json_str)
            time.sleep(1)

    def _setup_backend(
        self,
        wireless_version: bool,
        sim: bool,
        mockup_sim: bool,
        serialport: str,
        scene: str,
        check_collision: bool,
        kinematics_engine: str,
        headless: bool,
        use_audio: bool,
        hardware_config_filepath: str | None = None,
        reflash_motors_on_start: bool = True,
    ) -> "RobotBackend | MujocoBackend | MockupSimBackend":
        if mockup_sim:
            return MockupSimBackend(
                check_collision=check_collision,
                kinematics_engine=kinematics_engine,
                use_audio=use_audio,
            )
        elif sim:
            return MujocoBackend(
                scene=scene,
                check_collision=check_collision,
                kinematics_engine=kinematics_engine,
                headless=headless,
                use_audio=use_audio,
            )
        else:
            if serialport == "auto":
                ports = find_serial_port(wireless_version=wireless_version)

                if len(ports) == 0:
                    raise RuntimeError(
                        "No Reachy Mini serial port found. "
                        "Check USB connection and permissions. "
                        "Or directly specify the serial port using --serialport."
                    )
                elif len(ports) > 1:
                    raise RuntimeError(
                        f"Multiple Reachy Mini serial ports found {ports}. "
                        "Please specify the serial port using --serialport."
                    )

                serialport = ports[0]
                self.logger.info(f"Found Reachy Mini serial port: {serialport}")

            self.logger.info(
                f"Creating RobotBackend with parameters: serialport={serialport}, check_collision={check_collision}, kinematics_engine={kinematics_engine}"
            )

            if reflash_motors_on_start:
                reflash_motors_if_needed(serialport, dont_light_up=True)

            return RobotBackend(
                serialport=serialport,
                log_level=self.log_level,
                check_collision=check_collision,
                kinematics_engine=kinematics_engine,
                use_audio=use_audio,
                wireless_version=wireless_version,
                hardware_config_filepath=hardware_config_filepath,
            )
