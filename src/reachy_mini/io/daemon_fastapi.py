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
    daemon.start(
        sim=sim,
        serialport=serialport,
        scene=scene,
        localhost_only=localhost_only,
        wake_up_on_start=wake_up_on_start,
    )
    return {"state": daemon.state}


@app.post("/stop")
def stop_daemon(goto_sleep_on_stop: bool = True) -> dict:
    daemon.stop(goto_sleep_on_stop=goto_sleep_on_stop)
    return {"state": daemon.state}


@app.post("/restart")
def restart_daemon() -> dict:
    daemon.restart()
    return {"state": daemon.state}


@app.get("/status")
def status_daemon() -> "DaemonStatus":
    return daemon.status()
