import logging
from dataclasses import dataclass
from enum import Enum
from threading import Thread
from typing import Optional

import serial.tools.list_ports

from reachy_mini import ReachyMini
from reachy_mini.io import Server
from reachy_mini.mujoco_backend import MujocoBackend, MujocoBackendStatus
from reachy_mini.robot_backend import RobotBackend, RobotBackendStatus


class Daemon:
    def __init__(self, log_level: str = "INFO"):
        self.log_level = log_level
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.log_level)

        self._status = DaemonStatus(
            state=DaemonState.STOPPED,
            backend_status=None,
            error=None,
        )

    def start(
        self,
        sim: bool = False,
        serialport: str = "auto",
        scene: str = "empty",
        localhost_only: bool = True,
        wake_up_on_start: bool = True,
    ) -> "DaemonState":
        if self._status.state == DaemonState.RUNNING:
            self.logger.warning("Daemon is already running.")
            return self._status.state

        self.logger.info("Starting Reachy Mini daemon...")

        try:
            self.backend = self.setup_backend(
                sim=sim,
                serialport=serialport,
                scene=scene,
            )
        except Exception as e:
            self._status.state = DaemonState.ERROR
            self._status.error = str(e)
            raise e

        self.server = Server(self.backend, localhost_only=localhost_only)
        self.server.start()

        self.backend_run_thread = Thread(target=self.backend.run)
        self.backend_run_thread.start()

        if not self.backend.ready.wait(timeout=2.0):
            self.logger.error(
                "Backend is not ready after 2 seconds. Some error occurred."
            )
            self._status.state = DaemonState.ERROR
            return self._status.state

        if wake_up_on_start:
            try:
                self.logger.info("Waking up Reachy Mini...")
                with ReachyMini() as mini:
                    mini.set_torque(on=True)
                    mini.wake_up()
            except Exception as e:
                self.logger.error(f"Error while waking up Reachy Mini: {e}")
                self._status.state = DaemonState.ERROR
                self._status.error = str(e)
                return self._status.state
            except KeyboardInterrupt:
                self.logger.warning("Wake up interrupted by user.")
                self._status.state = DaemonState.STOPPING
                return self._status.state

        self.logger.info("Daemon started successfully.")
        self._status.state = DaemonState.RUNNING
        return self._status.state

    def stop(self, goto_sleep_on_stop: bool = True):
        if self._status.state == DaemonState.STOPPED:
            self.logger.warning("Daemon is already stopped.")
            return

        try:
            if self._status.state in (DaemonState.STOPPING, DaemonState.ERROR):
                goto_sleep_on_stop = False

            self.logger.info("Stopping Reachy Mini daemon...")
            self._status.state = DaemonState.STOPPING

            if goto_sleep_on_stop:
                try:
                    self.logger.info("Putting Reachy Mini to sleep...")
                    with ReachyMini() as mini:
                        mini.goto_sleep()
                        mini.set_torque(on=False)
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
            self.server.stop()

            if self._status.state != DaemonState.ERROR:
                self.logger.info("Daemon stopped successfully.")
                self._status.state = DaemonState.STOPPED
        except Exception as e:
            self.logger.error(f"Error while stopping the daemon: {e}")
            self._status.state = DaemonState.ERROR
            self._status.error = str(e)
        except KeyboardInterrupt:
            self.logger.warning("Daemon already stopping...")
            pass

    def restart(self):
        if self._status.state == DaemonState.STOPPED:
            self.logger.warning("Daemon is not running.")
            return self._status.state

        if self._status.state in (DaemonState.RUNNING, DaemonState.ERROR):
            self.logger.info("Restarting Reachy Mini daemon...")
            self.stop(goto_sleep_on_stop=False)
            # TODO: Re-use the existing parameters for start
            self.start(wake_up_on_start=False)

    def status(self) -> "DaemonStatus":
        if hasattr(self, "backend"):
            self._status.backend_status = self.backend.get_status()

        return self._status

    def run4ever(
        self,
        sim: bool = False,
        serialport: str = "auto",
        scene: str = "empty",
        localhost_only: bool = True,
        wake_up_on_start: bool = True,
        goto_sleep_on_stop: bool = True,
    ):
        self.start(
            sim=sim,
            serialport=serialport,
            scene=scene,
            localhost_only=localhost_only,
            wake_up_on_start=wake_up_on_start,
        )

        if self._status.state == DaemonState.RUNNING:
            try:
                self.logger.info("Daemon is running. Press Ctrl+C to stop.")
                while self.backend_run_thread.is_alive():
                    self.logger.debug(f"Daemon status: {self.status()}")
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

        self.stop(goto_sleep_on_stop)

    def setup_backend(self, sim, serialport, scene) -> "RobotBackend | MujocoBackend":
        if sim:
            return MujocoBackend(scene=scene)
        else:
            if serialport == "auto":
                ports = find_serial_port()

                if len(ports) == 0:
                    raise RuntimeError(
                        "No Reachy Mini serial port found. "
                        "Check USB connection and permissions."
                        "Or directly specify the serial port using --serialport."
                    )
                elif len(ports) > 1:
                    raise RuntimeError(
                        f"Multiple Reachy Mini serial ports found {ports}."
                        "Please specify the serial port using --serialport."
                    )

                serialport = ports[0]
                self.logger.info(f"Found Reachy Mini serial port: {serialport}")

            return RobotBackend(serialport=serialport, log_level=self.log_level)


class DaemonState(Enum):
    """
    Enum representing the state of the Reachy Mini daemon.
    """

    INITIALIZING = "initializing"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class DaemonStatus:
    state: DaemonState
    backend_status: Optional[RobotBackendStatus | MujocoBackendStatus]
    error: Optional[str] = None


def find_serial_port(vid: str = "1a86", pid: str = "55d3") -> list[str]:
    ports = serial.tools.list_ports.comports()

    vid = vid.upper()
    pid = pid.upper()

    return [p.device for p in ports if f"USB VID:PID={vid}:{pid}" in p.hwid]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run the Reachy Mini daemon.")
    parser.add_argument(
        "--sim",
        action="store_true",
        help="Run in simulation mode using Mujoco.",
    )
    parser.add_argument(
        "-p",
        "--serialport",
        type=str,
        default="auto",
        help="Serial port for real motors (default: will try to automatically find the port).",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="empty",
        help="Name of the scene to load (default: empty)",
    )
    parser.add_argument(
        "--localhost-only",
        action="store_true",
        default=True,
        help="Restrict the server to localhost only (default: True).",
    )
    parser.add_argument(
        "--no-localhost-only",
        action="store_false",
        dest="localhost_only",
        help="Allow the server to listen on all interfaces (default: False).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    Daemon(log_level=args.log_level).run4ever(
        sim=args.sim,
        serialport=args.serialport,
        scene=args.scene,
        localhost_only=args.localhost_only,
    )


if __name__ == "__main__":
    main()
