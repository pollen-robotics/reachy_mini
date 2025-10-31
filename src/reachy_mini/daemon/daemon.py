"""Daemon for Reachy Mini robot.

This module provides a daemon that runs a backend for either a simulated Reachy Mini using Mujoco or a real Reachy Mini robot using a serial connection.
It includes methods to start, stop, and restart the daemon, as well as to check its status.
It also provides a command-line interface for easy interaction.
"""

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass
from enum import Enum
from threading import Event, Thread
from typing import Any, Optional

from reachy_mini.daemon.backend.abstract import MotorControlMode
from reachy_mini.daemon.utils import (
    convert_enum_to_dict,
    find_serial_port,
    get_ip_address,
)

from ..io.zenoh_server import ZenohServer
from .backend.mujoco import MujocoBackend, MujocoBackendStatus
from .backend.robot import RobotBackend, RobotBackendStatus


class Daemon:
    """Daemon for simulated or real Reachy Mini robot.

    Runs the server with the appropriate backend (Mujoco for simulation or RobotBackend for real hardware).
    """

    def __init__(self, log_level: str = "INFO", wireless_version: bool = False) -> None:
        """Initialize the Reachy Mini daemon."""
        self.log_level = log_level
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.log_level)

        self.wireless_version = wireless_version

        self.backend: "RobotBackend | MujocoBackend | None" = None
        self._status = DaemonStatus(
            state=DaemonState.NOT_INITIALIZED,
            wireless_version=wireless_version,
            simulation_enabled=None,
            backend_status=None,
            error=None,
            wlan_ip=None,
        )
        self._thread_event_publish_status = Event()

        self._webrtc: Optional[Any] = (
            None  # type GstWebRTC imported for wireless version only
        )
        if wireless_version:
            from reachy_mini.media.webrtc_daemon import GstWebRTC

            self._webrtc = GstWebRTC(log_level)

    async def start(
        self,
        sim: bool = False,
        serialport: str = "auto",
        scene: str = "empty",
        localhost_only: bool = True,
        wake_up_on_start: bool = True,
        check_collision: bool = False,
        kinematics_engine: str = "AnalyticalKinematics",
        headless: bool = False,
    ) -> "DaemonState":
        """Start the Reachy Mini daemon.

        Args:
            sim (bool): If True, run in simulation mode using Mujoco. Defaults to False.
            serialport (str): Serial port for real motors. Defaults to "auto", which will try to find the port automatically.
            scene (str): Name of the scene to load in simulation mode ("empty" or "minimal"). Defaults to "empty".
            localhost_only (bool): If True, restrict the server to localhost only clients. Defaults to True.
            wake_up_on_start (bool): If True, wake up Reachy Mini on start. Defaults to True.
            check_collision (bool): If True, enable collision checking. Defaults to False.
            kinematics_engine (str): Kinematics engine to use. Defaults to "AnalyticalKinematics".
            headless (bool): If True, run Mujoco in headless mode (no GUI). Defaults to False.

        Returns:
            DaemonState: The current state of the daemon after attempting to start it.

        """
        if self._status.state == DaemonState.RUNNING:
            self.logger.warning("Daemon is already running.")
            return self._status.state

        self._status.simulation_enabled = sim

        if not localhost_only:
            self._status.wlan_ip = get_ip_address()

        self._start_params = {
            "sim": sim,
            "serialport": serialport,
            "headless": headless,
            "scene": scene,
            "localhost_only": localhost_only,
        }

        self.logger.info("Starting Reachy Mini daemon...")
        self._status.state = DaemonState.STARTING

        try:
            self.backend = self._setup_backend(
                wireless_version=self.wireless_version,
                sim=sim,
                serialport=serialport,
                scene=scene,
                check_collision=check_collision,
                kinematics_engine=kinematics_engine,
                headless=headless,
            )
        except Exception as e:
            self._status.state = DaemonState.ERROR
            self._status.error = str(e)
            raise e

        self.server = ZenohServer(self.backend, localhost_only=localhost_only)
        self.server.start()

        self._thread_publish_status = Thread(target=self._publish_status, daemon=True)
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
                self.server.stop()
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

        if self._webrtc:
            await asyncio.sleep(
                0.2
            )  # Give some time for the backend to release the audio device
            self._webrtc.start()

        self.logger.info("Daemon started successfully.")
        self._status.state = DaemonState.RUNNING
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
            self._status.state = DaemonState.STOPPED
            return self._status.state

        try:
            if self._status.state in (DaemonState.STOPPING, DaemonState.ERROR):
                goto_sleep_on_stop = False

            self.logger.info("Stopping Reachy Mini daemon...")
            self._status.state = DaemonState.STOPPING
            self.backend.is_shutting_down = True
            self._thread_event_publish_status.set()
            self.server.stop()

            if self._webrtc:
                self._webrtc.stop()

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
            self.backend_run_thread.join(timeout=5.0)
            if self.backend_run_thread.is_alive():
                self.logger.warning("Backend did not stop in time, forcing shutdown.")
                self._status.state = DaemonState.ERROR

            self.backend.close()
            self.backend.ready.clear()

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
        serialport: Optional[str] = None,
        scene: Optional[str] = None,
        headless: Optional[bool] = None,
        localhost_only: Optional[bool] = None,
        wake_up_on_start: Optional[bool] = None,
        goto_sleep_on_stop: Optional[bool] = None,
    ) -> "DaemonState":
        """Restart the Reachy Mini daemon.

        Args:
            sim (bool): If True, run in simulation mode using Mujoco. Defaults to None (uses the previous value).
            serialport (str): Serial port for real motors. Defaults to None (uses the previous value).
            scene (str): Name of the scene to load in simulation mode ("empty" or "minimal"). Defaults to None (uses the previous value).
            headless (bool): If True, run Mujoco in headless mode (no GUI). Defaults to None (uses the previous value).
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
                "serialport": serialport
                if serialport is not None
                else self._start_params["serialport"],
                "scene": scene if scene is not None else self._start_params["scene"],
                "headless": headless
                if headless is not None
                else self._start_params["headless"],
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
            json_str = json.dumps(
                asdict(self.status(), dict_factory=convert_enum_to_dict)
            )
            self.server.pub_status.put(json_str)
            time.sleep(1)

    async def run4ever(
        self,
        sim: bool = False,
        serialport: str = "auto",
        scene: str = "empty",
        localhost_only: bool = True,
        wake_up_on_start: bool = True,
        goto_sleep_on_stop: bool = True,
        check_collision: bool = False,
        kinematics_engine: str = "AnalyticalKinematics",
        headless: bool = False,
    ) -> None:
        """Run the Reachy Mini daemon indefinitely.

        First, it starts the daemon, then it keeps checking the status and allows for graceful shutdown on user interrupt (Ctrl+C).

        Args:
            sim (bool): If True, run in simulation mode using Mujoco. Defaults to False.
            serialport (str): Serial port for real motors. Defaults to "auto", which will try to find the port automatically.
            scene (str): Name of the scene to load in simulation mode ("empty" or "minimal"). Defaults to "empty".
            localhost_only (bool): If True, restrict the server to localhost only clients. Defaults to True.
            wake_up_on_start (bool): If True, wake up Reachy Mini on start. Defaults to True.
            goto_sleep_on_stop (bool): If True, put Reachy Mini to sleep on stop. Defaults to True
            check_collision (bool): If True, enable collision checking. Defaults to False.
            kinematics_engine (str): Kinematics engine to use. Defaults to "AnalyticalKinematics".
            headless (bool): If True, run Mujoco in headless mode (no GUI). Defaults to False.

        """
        await self.start(
            sim=sim,
            serialport=serialport,
            scene=scene,
            localhost_only=localhost_only,
            wake_up_on_start=wake_up_on_start,
            check_collision=check_collision,
            kinematics_engine=kinematics_engine,
            headless=headless,
        )

        if self._status.state == DaemonState.RUNNING:
            try:
                self.logger.info("Daemon is running. Press Ctrl+C to stop.")
                while self.backend_run_thread.is_alive():
                    self.logger.info(f"Daemon status: {self.status()}")
                    for _ in range(10):
                        self.backend_run_thread.join(timeout=1.0)
                else:
                    self.logger.error("Backend thread has stopped unexpectedly.")
                    self._status.state = DaemonState.ERROR
            except KeyboardInterrupt:
                self.logger.warning("Daemon interrupted by user.")
            except Exception as e:
                self.logger.error(f"An error occurred: {e}")
                self._status.state = DaemonState.ERROR
                self._status.error = str(e)

        await self.stop(goto_sleep_on_stop)

    def _setup_backend(
        self,
        wireless_version: bool,
        sim: bool,
        serialport: str,
        scene: str,
        check_collision: bool,
        kinematics_engine: str,
        headless: bool,
    ) -> "RobotBackend | MujocoBackend":
        if sim:
            return MujocoBackend(
                scene=scene,
                check_collision=check_collision,
                kinematics_engine=kinematics_engine,
                headless=headless,
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
                        f"Multiple Reachy Mini serial ports found {ports}."
                        "Please specify the serial port using --serialport."
                    )

                serialport = ports[0]
                self.logger.info(f"Found Reachy Mini serial port: {serialport}")

            return RobotBackend(
                serialport=serialport,
                log_level=self.log_level,
                check_collision=check_collision,
                kinematics_engine=kinematics_engine,
            )


class DaemonState(Enum):
    """Enum representing the state of the Reachy Mini daemon."""

    NOT_INITIALIZED = "not_initialized"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class DaemonStatus:
    """Dataclass representing the status of the Reachy Mini daemon."""

    state: DaemonState
    wireless_version: bool
    simulation_enabled: Optional[bool]
    backend_status: Optional[RobotBackendStatus | MujocoBackendStatus]
    error: Optional[str] = None
    wlan_ip: Optional[str] = None
