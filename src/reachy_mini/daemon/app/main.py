"""Daemon entry point for the Reachy Mini robot.

This script serves as the command-line interface (CLI) entry point for the Reachy Mini daemon.
It initializes the daemon with specified parameters such as simulation mode, serial port,
scene to load, and logging level. The daemon runs indefinitely, handling requests and
managing the robot's state.

"""

import argparse
import asyncio
import logging
import types
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator

import uvicorn
from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from reachy_mini.apps.manager import AppManager
from reachy_mini.daemon.app.routers import (
    apps,
    daemon,
    hf_auth,
    kinematics,
    logs,
    motors,
    move,
    state,
    volume,
)
from reachy_mini.daemon.daemon import Daemon
from reachy_mini.media.audio_utils import (
    check_reachymini_asoundrc,
    write_asoundrc_to_home,
)
from reachy_mini.utils.wireless_version.startup_check import (
    check_and_fix_venvs_ownership,
    check_and_sync_apps_venv_sdk,
    check_and_update_bluetooth_service,
    check_and_update_wireless_launcher,
)


@dataclass
class Args:
    """Arguments for configuring the Reachy Mini daemon."""

    log_level: str = "INFO"
    log_file: str | None = None

    wireless_version: bool = False
    desktop_app_daemon: bool = False

    serialport: str = "auto"
    hardware_config_filepath: str | None = None

    sim: bool = False
    scene: str = "empty"
    headless: bool = False
    websocket_uri: str | None = None
    stream_media: bool = False
    use_audio: bool = True

    kinematics_engine: str = "AnalyticalKinematics"
    check_collision: bool = False

    autostart: bool = True
    timeout_health_check: float | None = None

    wake_up_on_start: bool = True
    goto_sleep_on_stop: bool = True

    robot_name: str = "reachy_mini"

    fastapi_host: str = "0.0.0.0"
    fastapi_port: int = 8000

    localhost_only: bool | None = None


def create_app(args: Args, health_check_event: asyncio.Event | None = None) -> FastAPI:
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

        try:
            if args.autostart:
                await app.state.daemon.start(
                    serialport=args.serialport,
                    sim=args.sim,
                    scene=args.scene,
                    headless=args.headless,
                    websocket_uri=args.websocket_uri,
                    stream_media=args.stream_media,
                    use_audio=args.use_audio,
                    kinematics_engine=args.kinematics_engine,
                    check_collision=args.check_collision,
                    wake_up_on_start=args.wake_up_on_start,
                    localhost_only=localhost_only,
                    hardware_config_filepath=args.hardware_config_filepath,
                )
            yield
        finally:
            # Ensure cleanup happens even if there's an exception
            try:
                logging.info("Shutting down app manager...")
                await app.state.app_manager.close()
            except Exception as e:
                logging.exception(f"Error closing app manager: {e}")

            try:
                logging.info("Shutting down daemon...")
                await app.state.daemon.stop(
                    goto_sleep_on_stop=args.goto_sleep_on_stop,
                )
            except Exception as e:
                logging.exception(f"Error stopping daemon: {e}")

    app = FastAPI(
        lifespan=lifespan,
    )

    app.state.args = args
    app.state.daemon = Daemon(
        robot_name=args.robot_name,
        wireless_version=args.wireless_version,
        desktop_app_daemon=args.desktop_app_daemon,
    )
    app.state.app_manager = AppManager(
        wireless_version=args.wireless_version,
        desktop_app_daemon=args.desktop_app_daemon,
        daemon=app.state.daemon,
    )

    router = APIRouter(prefix="/api")
    router.include_router(apps.router)
    router.include_router(daemon.router)
    router.include_router(hf_auth.router)
    router.include_router(kinematics.router)
    router.include_router(motors.router)
    router.include_router(move.router)
    router.include_router(state.router)
    router.include_router(volume.router)

    if args.wireless_version:
        from .routers import cache, update, wifi_config

        app.include_router(cache.router)
        app.include_router(logs.router)
        app.include_router(update.router)
        app.include_router(wifi_config.router)

    app.include_router(router)

    if health_check_event is not None:

        @app.post("/health-check")
        async def health_check() -> dict[str, str]:
            """Health check endpoint to reset the health check timer."""
            health_check_event.set()
            return {"status": "ok"}

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
        return templates.TemplateResponse(
            "index.html", {"request": request, "args": args}
        )

    if args.wireless_version:

        @app.get("/settings")
        async def settings(request: Request) -> HTMLResponse:
            """Render the settings page."""
            return templates.TemplateResponse("settings.html", {"request": request})

        @app.get("/logs")
        async def logs_page(request: Request) -> HTMLResponse:
            """Render the logs page."""
            return templates.TemplateResponse("logs.html", {"request": request})

    return app


def run_app(args: Args) -> None:
    """Run the FastAPI app with Uvicorn."""
    # Configure logging to ensure all logs go to stderr (captured by systemd)
    import sys

    root_logger = logging.getLogger()
    root_logger.setLevel(args.log_level)

    # Create handler that writes to stderr with immediate flush
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(args.log_level)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    root_logger.addHandler(handler)

    # Explicitly configure the apps.manager logger to ensure propagation
    apps_logger = logging.getLogger("reachy_mini.apps.manager")
    apps_logger.setLevel(args.log_level)
    apps_logger.propagate = True  # Ensure it propagates to root logger

    # Install exception hook to catch uncaught exceptions
    def exception_hook(exc_type: type[BaseException], exc_value: BaseException, exc_traceback: types.TracebackType | None) -> None:
        """Log uncaught exceptions with full traceback."""
        if issubclass(exc_type, KeyboardInterrupt):
            # Allow KeyboardInterrupt to exit normally
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        root_logger.critical(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
        sys.stderr.flush()

    sys.excepthook = exception_hook

    async def run_server() -> None:
        # Set up asyncio exception handler to catch unhandled task exceptions
        loop = asyncio.get_running_loop()

        def asyncio_exception_handler(loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
            """Handle exceptions in asyncio tasks."""
            exception = context.get("exception")
            if exception:
                root_logger.error(
                    f"Unhandled exception in asyncio task: {context.get('message', 'No message')}",
                    exc_info=(type(exception), exception, exception.__traceback__)
                )
            else:
                root_logger.error(f"Asyncio error: {context}")
            sys.stderr.flush()

        loop.set_exception_handler(asyncio_exception_handler)

        health_check_event = asyncio.Event()
        app = create_app(args, health_check_event)

        config = uvicorn.Config(
            app,
            host=args.fastapi_host,
            port=args.fastapi_port,
            log_config=None,  # Don't override Python logging configuration
        )
        server = uvicorn.Server(config)

        health_check_task = None

        async def health_check_timeout(timeout_seconds: float) -> None:
            while True:
                try:
                    await asyncio.wait_for(
                        health_check_event.wait(),
                        timeout=timeout_seconds,
                    )
                    health_check_event.clear()
                except asyncio.TimeoutError:
                    logging.warning("Health check timeout reached, stopping app.")
                    server.should_exit = True
                    break
                except asyncio.CancelledError:
                    logging.info("Health check task cancelled.")
                    break

        try:
            if args.timeout_health_check is not None:
                health_check_task = asyncio.create_task(
                    health_check_timeout(args.timeout_health_check)
                )
            await server.serve()
        except KeyboardInterrupt:
            logging.info("Received Ctrl-C, shutting down gracefully.")
        except Exception as e:
            logging.exception(f"Error during server operation: {e}")
            raise
        finally:
            # Cancel health check task if it exists
            if health_check_task and not health_check_task.done():
                health_check_task.cancel()
                try:
                    await health_check_task
                except asyncio.CancelledError:
                    pass

    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        logging.info("Shutdown complete.")
    except Exception as e:
        logging.exception(f"Error during shutdown: {e}")
        sys.stderr.flush()
        raise


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
    parser.add_argument(
        "--desktop-app-daemon",
        action="store_true",
        default=default_args.desktop_app_daemon,
        help="Use the desktop version of Reachy Mini (default: False).",
    )

    parser.add_argument(
        "--robot-name",
        type=str,
        default=default_args.robot_name,
        help="Name of the robot (default: reachy_mini).",
    )

    # Real robot mode
    parser.add_argument(
        "-p",
        "--serialport",
        type=str,
        default=default_args.serialport,
        help="Serial port for real motors (default: will try to automatically find the port).",
    )
    default_hw_config_path = str(
        (
            Path(__file__).parent.parent.parent
            / "assets"
            / "config"
            / "hardware_config.yaml"
        ).resolve()
    )
    parser.add_argument(
        "--hardware-config-filepath",
        type=str,
        default=default_hw_config_path,
        help=f"Path to the hardware configuration YAML file (default: {default_hw_config_path}).",
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
    parser.add_argument(
        "--websocket-uri",
        type=str,
        default=default_args.websocket_uri,
        help="WebSocket URI for remote control and streaming of the robot (default: None). Example: ws://localhost:8000",
    )
    parser.add_argument(
        "--stream-media",
        action="store_true",
        default=default_args.stream_media,
        help="Stream media to the WebSocket. Requires a WebSocket URI to be set. (default: False).",
    )
    parser.add_argument(
        "--deactivate-audio",
        action="store_false",
        dest="use_audio",
        default=default_args.use_audio,
        help="Deactivate audio (default: True).",
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
        "--timeout-health-check",
        type=float,
        default=None,
        help="Set the health check timeout in seconds (default: None).",
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
    parser.add_argument(
        "--log-file",
        type=str,
        default=default_args.log_file,
        help="Path to a file to write logs to.",
    )

    args = parser.parse_args()

    if args.log_file:
        file_handler = logging.FileHandler(args.log_file, mode="a")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logging.getLogger().addHandler(file_handler)
        logging.getLogger().setLevel(args.log_level)

    if args.wireless_version:
        # Check and fix ownership of /venvs directory
        check_and_fix_venvs_ownership(custom_logger=logging.getLogger())

        # Check and update bluetooth service if needed
        check_and_update_bluetooth_service()

        # Check and update wireless launcher if needed
        check_and_update_wireless_launcher()

        # Check and sync apps_venv SDK version with daemon
        check_and_sync_apps_venv_sdk()

        if check_reachymini_asoundrc():
            logging.getLogger().info(
                "~/.asoundrc correctly configured for Reachy Mini Audio."
            )
        else:
            logging.getLogger().warning(
                "~/.asoundrc not found or not correctly configured for Reachy Mini Audio. "
                "Creating a new one."
            )
            write_asoundrc_to_home()

    run_app(Args(**vars(args)))


if __name__ == "__main__":
    main()
