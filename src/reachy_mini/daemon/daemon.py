"""Daemon for Reachy Mini robot.

This module provides a daemon that runs a backend for either a simulated Reachy Mini using Mujoco or a real Reachy Mini robot using a serial connection.
It includes methods to start, stop, and restart the daemon, as well as to check its status.
It also provides a command-line interface for easy interaction.
"""

import asyncio
import logging
import time
from importlib.metadata import PackageNotFoundError, version
from threading import Event, Thread
from typing import Any, Optional

from reachy_mini.daemon.utils import (
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
        use_sim: bool = False,
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
        )
        self.ws_server: "WSServer | None" = None
        self.backend_run_thread: "Thread | None" = None
        self._thread_event_publish_status = Event()

        self._media_server: Optional["GstMediaServer"] = (
            None  # GstMediaServer when media is enabled
        )
        self._media_released = False
        if not no_media:
            from reachy_mini.media.media_server import GstMediaServer

            try:
                self._media_server = GstMediaServer(log_level, use_sim=use_sim)
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
            self._media_server.__del__()
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

            self.logger.info("Starting central signaling relay...")
            await start_central_relay(
                hf_token=hf_token,
                robot_name=self.robot_name,
            )
            self.logger.info("Central signaling relay started")
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
        localhost_only: bool = True,
        wake_up_on_start: bool = True,
        check_collision: bool = False,
        kinematics_engine: str = "AnalyticalKinematics",
        headless: bool = False,
        use_audio: bool = True,  # kept for backward compat, overridden by no_media
        hardware_config_filepath: str | None = None,
    ) -> "DaemonState":
        """Start the Reachy Mini daemon.

        Args:
            sim (bool): If True, run in simulation mode using Mujoco. Defaults to False.
            mockup_sim (bool): If True, run in lightweight simulation mode (no MuJoCo). Defaults to False.
            serialport (str): Serial port for real motors. Defaults to "auto", which will try to find the port automatically.
            scene (str): Name of the scene to load in simulation mode ("empty" or "minimal"). Defaults to "empty".
            localhost_only (bool): If True, restrict the server to localhost only clients. Defaults to True.
            wake_up_on_start (bool): If True, wake up Reachy Mini on start. Defaults to True.
            check_collision (bool): If True, enable collision checking. Defaults to False.
            kinematics_engine (str): Kinematics engine to use. Defaults to "AnalyticalKinematics".
            headless (bool): If True, run Mujoco in headless mode (no GUI). Defaults to False.
            use_audio (bool): If True, enable audio. Defaults to True.
            hardware_config_filepath (str | None): Path to the hardware configuration YAML file. Defaults to None.

        Returns:
            DaemonState: The current state of the daemon after attempting to start it.

        """
        if self._status.state == DaemonState.RUNNING:
            self.logger.warning("Daemon is already running.")
            return self._status.state

        self.logger.info(
            f"Daemon start parameters: sim={sim}, mockup_sim={mockup_sim}, serialport={serialport}, scene={scene}, localhost_only={localhost_only}, wake_up_on_start={wake_up_on_start}, check_collision={check_collision}, kinematics_engine={kinematics_engine}, headless={headless}, hardware_config_filepath={hardware_config_filepath}"
        )

        # mockup-sim behaves exactly like a real robot for apps (they open webcam directly)
        # Only MuJoCo (--sim) sets simulation_enabled=True (streams video via UDP)
        self._status.simulation_enabled = sim
        self._status.mockup_sim_enabled = mockup_sim

        if not localhost_only:
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
            "localhost_only": localhost_only,
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
                self._media_server.start()

                # Start central signaling relay for remote WebRTC access
                await self._start_central_signaling_relay()

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
        localhost_only: Optional[bool] = None,
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
            localhost_only (bool): If True, restrict the server to localhost only clients. Defaults to None (uses the previous value).
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
                "localhost_only": localhost_only
                if localhost_only is not None
                else self._start_params["localhost_only"],
                "wake_up_on_start": wake_up_on_start
                if wake_up_on_start is not None
                else False,
            }

            return await self.start(**params)

        raise NotImplementedError(
            "Restarting is only supported when the daemon is in RUNNING or ERROR state."
        )

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
