"""Daemon entry point for the Reachy Mini robot.

This script serves as the command-line interface (CLI) entry point for the Reachy Mini daemon.
It initializes the daemon with specified parameters such as simulation mode, serial port,
scene to load, and logging level. The daemon runs indefinitely, handling requests and
managing the robot's state.

"""

import argparse
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator

import uvicorn
from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from reachy_mini.apps.manager import AppManager
from reachy_mini.daemon.app.routers import apps, daemon, kinematics, motors, move, state
from reachy_mini.daemon.daemon import Daemon


@dataclass
class Args:
    """Arguments for configuring the Reachy Mini daemon."""

    log_level: str = "INFO"

    wireless_version: bool = False

    serialport: str = "auto"

    sim: bool = False
    scene: str = "empty"
    headless: bool = False

    kinematics_engine: str = "AnalyticalKinematics"
    check_collision: bool = False

    autostart: bool = True

    wake_up_on_start: bool = True
    goto_sleep_on_stop: bool = True

    fastapi_host: str = "0.0.0.0"
    fastapi_port: int = 8000

    localhost_only: bool | None = None


def create_app(args: Args) -> FastAPI:
    """Create and configure the FastAPI application."""
    localhost_only = (
        args.localhost_only
        if args.localhost_only is not None
        else (False if args.wireless_version else True)
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        """Lifespan context manager for the FastAPI application."""
        args = app.state.args  # type: Args

        if args.autostart:
            await app.state.daemon.start(
                serialport=args.serialport,
                sim=args.sim,
                scene=args.scene,
                headless=args.headless,
                kinematics_engine=args.kinematics_engine,
                check_collision=args.check_collision,
                wake_up_on_start=args.wake_up_on_start,
                localhost_only=localhost_only,
            )
        yield
        await app.state.app_manager.close()
        await app.state.daemon.stop(
            goto_sleep_on_stop=args.goto_sleep_on_stop,
        )

    app = FastAPI(
        lifespan=lifespan,
    )

    app.state.args = args
    app.state.daemon = Daemon(wireless_version=args.wireless_version)
    app.state.app_manager = AppManager()

    router = APIRouter(prefix="/api")
    router.include_router(apps.router)
    router.include_router(daemon.router)
    router.include_router(kinematics.router)
    router.include_router(motors.router)
    router.include_router(move.router)
    router.include_router(state.router)
    app.include_router(router)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # or restrict to your HF domain
        allow_methods=["*"],
        allow_headers=["*"],
    )

    STATIC_DIR = Path(__file__).parent / "dashboard" / "static"
    TEMPLATES_DIR = Path(__file__).parent / "dashboard" / "templates"

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    @app.get("/")
    async def dashboard(request: Request) -> HTMLResponse:
        """Render the dashboard."""
        return templates.TemplateResponse("index.html", {"request": request})

    return app


def run_app(args: Args) -> None:
    """Run the FastAPI app with Uvicorn."""
    logging.basicConfig(level=logging.INFO)

    app = create_app(args)
    uvicorn.run(app, host=args.fastapi_host, port=args.fastapi_port)


def main() -> None:
    """Run the FastAPI app with Uvicorn."""
    default_args = Args()

    parser = argparse.ArgumentParser(description="Run the Reachy Mini daemon.")
    parser.add_argument(
        "--wireless-version",
        action="store_true",
        default=default_args.wireless_version,
        help="Use the wireless version of Reachy Mini (default: False).",
    )

    # Real robot mode
    parser.add_argument(
        "-p",
        "--serialport",
        type=str,
        default=default_args.serialport,
        help="Serial port for real motors (default: will try to automatically find the port).",
    )
    # Simulation mode
    parser.add_argument(
        "--sim",
        action="store_true",
        default=default_args.sim,
        help="Run in simulation mode using Mujoco.",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=default_args.scene,
        help="Name of the scene to load (default: empty)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=default_args.headless,
        help="Run the daemon in headless mode (default: False).",
    )
    # Daemon options
    parser.add_argument(
        "--autostart",
        action="store_true",
        default=default_args.autostart,
        help="Automatically start the daemon on launch (default: True).",
    )
    parser.add_argument(
        "--no-autostart",
        action="store_false",
        dest="autostart",
        help="Do not automatically start the daemon on launch (default: False).",
    )
    parser.add_argument(
        "--wake-up-on-start",
        action="store_true",
        default=default_args.wake_up_on_start,
        help="Wake up the robot on daemon start (default: True).",
    )
    parser.add_argument(
        "--no-wake-up-on-start",
        action="store_false",
        dest="wake_up_on_start",
        help="Do not wake up the robot on daemon start (default: False).",
    )
    parser.add_argument(
        "--goto-sleep-on-stop",
        action="store_true",
        default=default_args.goto_sleep_on_stop,
        help="Put the robot to sleep on daemon stop (default: True).",
    )
    parser.add_argument(
        "--no-goto-sleep-on-stop",
        action="store_false",
        dest="goto_sleep_on_stop",
        help="Do not put the robot to sleep on daemon stop (default: False).",
    )
    # Zenoh server options
    parser.add_argument(
        "--localhost-only",
        action="store_true",
        default=default_args.localhost_only,
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
        default=default_args.check_collision,
        help="Enable collision checking (default: False).",
    )

    parser.add_argument(
        "--kinematics-engine",
        type=str,
        default=default_args.kinematics_engine,
        choices=["Placo", "NN", "AnalyticalKinematics"],
        help="Set the kinematics engine (default: AnalyticalKinematics).",
    )
    # FastAPI server options
    parser.add_argument(
        "--fastapi-host",
        type=str,
        default=default_args.fastapi_host,
    )
    parser.add_argument(
        "--fastapi-port",
        type=int,
        default=default_args.fastapi_port,
    )
    # Logging options
    parser.add_argument(
        "--log-level",
        type=str,
        default=default_args.log_level,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO).",
    )

    args = parser.parse_args()
    run_app(Args(**vars(args)))


if __name__ == "__main__":
    main()
