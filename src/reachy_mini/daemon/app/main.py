"""Daemon entry point for the Reachy Mini robot.

This script serves as the command-line interface (CLI) entry point for the Reachy Mini daemon.
It initializes the daemon with specified parameters such as simulation mode, serial port,
scene to load, and logging level. The daemon runs indefinitely, handling requests and
managing the robot's state.

"""

import argparse
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException, Request, responses
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates

from reachy_mini.apps.manager import AppManager
from reachy_mini.daemon.app.routers import apps, daemon, kinematics, motors, move, state
from reachy_mini.daemon.daemon import Daemon

EXAMPLES_WEB_DIR = Path(__file__).parent / "examples"
TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for the FastAPI application."""
    await app.state.daemon.start(
        sim=app.state.args.sim,
        scene=app.state.args.scene,
        wake_up_on_start=app.state.args.wake_up_on_start,
    )
    yield
    await app.state.app_manager.close()
    await app.state.daemon.stop(
        goto_sleep_on_stop=app.state.args.goto_sleep_on_stop,
    )


logging.basicConfig(level=logging.INFO)

app = FastAPI(
    lifespan=lifespan,
)
app.state.daemon = Daemon()
app.state.app_manager = AppManager()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to your HF domain
    allow_methods=["*"],
    allow_headers=["*"],
)


router = APIRouter(prefix="/api")
router.include_router(apps.router)
router.include_router(daemon.router)
router.include_router(kinematics.router)
router.include_router(motors.router)
router.include_router(move.router)
router.include_router(state.router)

app.include_router(router)


# Route to list available HTML/JS/CSS examples with links using Jinja2 template
@app.get("/examples")
async def list_examples(request: Request):
    """Render the examples list using a Jinja2 template."""
    files = [f for f in os.listdir(EXAMPLES_WEB_DIR) if f.endswith(".html")]
    return templates.TemplateResponse(
        "examples_list.html", {"request": request, "files": files}
    )


# Route to serve a specific example file from examples_web
@app.get("/examples/{filename}")
async def serve_example(filename: str):
    """Serve a specific example file from examples_web directory."""
    file_path = os.path.join(EXAMPLES_WEB_DIR, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Example not found")
    return responses.FileResponse(file_path)


def main():
    """Run the FastAPI app with Uvicorn."""
    parser = argparse.ArgumentParser(description="Run the Reachy Mini daemon.")
    # Real robot mode
    parser.add_argument(
        "-p",
        "--serialport",
        type=str,
        default="auto",
        help="Serial port for real motors (default: will try to automatically find the port).",
    )
    # Simulation mode
    parser.add_argument(
        "--sim",
        action="store_true",
        help="Run in simulation mode using Mujoco.",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="empty",
        help="Name of the scene to load (default: empty)",
    )
    # Daemon options
    parser.add_argument(
        "--wake-up-on-start",
        action="store_true",
        default=True,
        help="Wake up the robot on daemon start (default: True).",
    )
    parser.add_argument(
        "--goto-sleep-on-stop",
        action="store_true",
        default=True,
        help="Put the robot to sleep on daemon stop (default: True).",
    )
    # Zenoh server options
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
    # Kinematics options
    parser.add_argument(
        "--check-collision",
        action="store_true",
        default=False,
        help="Enable collision checking (default: False).",
    )

    parser.add_argument(
        "--kinematics-engine",
        type=str,
        default="Placo",
        choices=["Placo", "NN", "Analytical"],
        help="Set the kinematics engine (default: Placo).",
    )
    # FastAPI server options
    parser.add_argument(
        "--fastapi-host",
        type=str,
        default="0.0.0.0",
    )
    parser.add_argument(
        "--fastapi-port",
        type=int,
        default=8000,
    )
    # Logging options
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO).",
    )

    args = parser.parse_args()

    app.state.args = args
    uvicorn.run(app, host=args.fastapi_host, port=args.fastapi_port)


if __name__ == "__main__":
    main()
