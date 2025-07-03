"""FastAPI application for managing the Reachy Mini daemon.

This module provides endpoints to start, stop, restart, and check the status of the Reachy Mini daemon.
It allows configuration of the daemon's parameters such as simulation mode, serial port, scene, and localhost-only connections.
"""

from fastapi import FastAPI

from reachy_mini.io.daemon import Daemon, DaemonStatus

app = FastAPI()
daemon = Daemon()


@app.post("/start")
def start_daemon(
    sim: bool = False,
    serialport: str = "auto",
    scene: str = "empty",
    localhost_only: bool = True,
    wake_up_on_start: bool = True,
) -> dict:
    """Start the Reachy Mini daemon.

    Args:
        sim (bool): Whether to run in simulation mode.
        serialport (str): Serial port to use, "auto" for automatic detection.
        scene (str): Scene configuration to load.
        localhost_only (bool): Whether to restrict connections to localhost only.
        wake_up_on_start (bool): Whether to wake up the robot on start.

    Returns:
        dict: A dictionary containing the current state of the daemon.

    """
    daemon.start(
        sim=sim,
        serialport=serialport,
        scene=scene,
        localhost_only=localhost_only,
        wake_up_on_start=wake_up_on_start,
    )
    return {"state": daemon._status.state}


@app.post("/stop")
def stop_daemon(goto_sleep_on_stop: bool = True) -> dict:
    """Stop the Reachy Mini daemon.

    Args:
        goto_sleep_on_stop (bool): Whether to put the robot to sleep on stop.

    Returns:
        dict: A dictionary containing the current state of the daemon.

    """
    daemon.stop(goto_sleep_on_stop=goto_sleep_on_stop)
    return {"state": daemon._status.state}


@app.post("/restart")
def restart_daemon() -> dict:
    """Restart the Reachy Mini daemon.

    Returns:
        dict: A dictionary containing the current state of the daemon.

    """
    daemon.restart()
    return {"state": daemon._status.state}


@app.get("/status")
def status_daemon() -> "DaemonStatus":
    """Get the current status of the Reachy Mini daemon.

    Returns:
        DaemonStatus: The current status of the daemon.

    """
    return daemon.status()
