from dataclasses import dataclass
from enum import Enum
from threading import Thread

import serial.tools.list_ports

from reachy_mini import ReachyMini
from reachy_mini.io import Server
from reachy_mini.mujoco_backend import MujocoBackend, MujocoBackendStatus
from reachy_mini.robot_backend import RobotBackend, RobotBackendStatus


class Daemon:
    def __init__(self):
        self.state = DaemonState.STOPPED

    def start(
        self,
        sim: bool = False,
        serialport: str = "auto",
        scene: str = "empty",
        localhost_only: bool = True,
        wake_up_on_start: bool = True,
    ) -> "DaemonState":
        if self.state == DaemonState.RUNNING:
            print("Daemon is already running.")
            return self.state

        self.backend = setup_backend(
            sim=sim,
            serialport=serialport,
            scene=scene,
        )

        print("Starting Reachy Mini daemon...")

        self.server = Server(self.backend, localhost_only=localhost_only)
        self.server.start()

        self.backend_run_thread = Thread(target=self.backend.run)
        self.backend_run_thread.start()

        if not self.backend.ready.wait(timeout=2.0):
            print("Backend is not ready after 2 seconds. Some error occurred.")
            self.state = DaemonState.ERROR
            return self.state

        if wake_up_on_start:
            try:
                print("Waking up Reachy Mini...")
                with ReachyMini() as mini:
                    mini.set_torque(on=True)
                    mini.wake_up()
            except Exception as e:
                print(f"Error while waking up Reachy Mini: {e}")
                self.state = DaemonState.ERROR
                return self.state
            except KeyboardInterrupt:
                print("Wake up interrupted by user.")
                self.state = DaemonState.STOPPING
                return self.state

        print("Daemon started successfully.")
        self.state = DaemonState.RUNNING
        return self.state

    def stop(self, goto_sleep_on_stop: bool = True):
        if self.state == DaemonState.STOPPED:
            print("Daemon is already stopped.")
            return

        try:
            if self.state in (DaemonState.STOPPING, DaemonState.ERROR):
                goto_sleep_on_stop = False

            print("Stopping Reachy Mini daemon...")
            self.state = DaemonState.STOPPING

            if goto_sleep_on_stop:
                try:
                    print("Putting Reachy Mini to sleep...")
                    with ReachyMini() as mini:
                        mini.goto_sleep()
                        mini.set_torque(on=False)
                except Exception as e:
                    print(f"Error while putting Reachy Mini to sleep: {e}")
                    self.state = DaemonState.ERROR
                except KeyboardInterrupt:
                    print("Sleep interrupted by user.")
                    self.state = DaemonState.STOPPING

            self.backend.should_stop.set()
            self.backend_run_thread.join(timeout=5.0)
            if self.backend_run_thread.is_alive():
                print("Backend did not stop in time, forcing shutdown.")
                self.state = DaemonState.ERROR

            self.backend.close()
            self.server.stop()

            if self.state != DaemonState.ERROR:
                print("Daemon stopped successfully.")
                self.state = DaemonState.STOPPED
        except Exception as e:
            print(f"Error while stopping the daemon: {e}")
            self.state = DaemonState.ERROR
        except KeyboardInterrupt:
            print("Daemon already stopping...")
            pass

    def restart(self):
        if self.state == DaemonState.STOPPED:
            print("Daemon is not running.")
            return self.state

        if self.state in (DaemonState.RUNNING, DaemonState.ERROR):
            print("Restarting Reachy Mini daemon...")
            self.stop(goto_sleep_on_stop=False)
            # TODO: Re-use the existing parameters for start
            self.start(wake_up_on_start=False)

    def status(self) -> "DaemonStatus":
        return DaemonStatus(
            state=self.state,
            backend_stats=self.backend.get_stats() if hasattr(self, "backend") else {},
        )

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

        if self.state == DaemonState.RUNNING:
            try:
                print("Daemon is running. Press Ctrl+C to stop.")
                while self.backend_run_thread.is_alive():
                    self.backend_run_thread.join(timeout=1.0)
                else:
                    print("Backend thread has stopped unexpectedly.")
                    self.state = DaemonState.ERROR
            except KeyboardInterrupt:
                print("Daemon interrupted by user.")
                # signal.signal(signal.SIGINT, signal_handler)
            except Exception as e:
                print(f"An error occurred: {e}")
                self.state = DaemonState.ERROR

        self.stop(goto_sleep_on_stop)


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
    backend_stats: dict


def setup_backend(sim, serialport, scene) -> "Backend":
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
                    "Multiple Reachy Mini serial ports found. "
                    "Please specify the serial port using --serialport."
                )

            serialport = ports[0]
            print(f"Found Reachy Mini serial port: {serialport}")

        return RobotBackend(serialport=serialport)


def find_serial_port(vid: str = "1a86", pid: str = "55d3") -> list[str]:
    ports = serial.tools.list_ports.comports()

    vid = vid.upper()
    pid = pid.upper()

    return [p.device for p in ports if f"USB VID:PID={vid}:{pid}" in p.hwid]


def signal_handler(signum, frame):
    """Handle termination signals"""
    print("Daemon already stopping...")


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
    args = parser.parse_args()

    Daemon().run4ever(
        sim=args.sim,
        serialport=args.serialport,
        scene=args.scene,
        localhost_only=args.localhost_only,
    )


if __name__ == "__main__":
    main()
