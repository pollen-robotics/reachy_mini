"""Daemon entry point for the Reachy Mini robot.

This script serves as the command-line interface (CLI) entry point for the Reachy Mini daemon.
It initializes the daemon with specified parameters such as simulation mode, serial port,
scene to load, and logging level. The daemon runs indefinitely, handling requests and
managing the robot's state.

"""

import argparse
import logging
import mimetypes
import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware

from reachy_mini.apps.manager import AppManager
from reachy_mini.daemon.app.routers import apps, daemon, kinematics, motors, move, simulation, state
from reachy_mini.daemon.daemon import Daemon

DASHBOARD_PAGES = Path(__file__).parent / "dashboard"
TEMPLATES_DIR = Path(__file__).parent / "templates"
WASM_DIR = Path(__file__).parent / "wasm" / "dist"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


class COOPStaticFiles(StaticFiles):
    """StaticFiles with COOP/COEP headers and proper MIME types for WASM support."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure proper MIME types are registered
        mimetypes.add_type('application/javascript', '.js')
        mimetypes.add_type('application/wasm', '.wasm')
        mimetypes.add_type('text/css', '.css')
        mimetypes.add_type('text/html', '.html')

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)

        # Debug logging
        logger.info(f"Serving file: {path}")

        if hasattr(response, 'headers'):
            # Add COOP/COEP headers required for SharedArrayBuffer
            response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
            response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
            response.headers['Cross-Origin-Resource-Policy'] = 'cross-origin'

            # Extract just the file extension for MIME type detection
            file_path = path.lower()
            original_content_type = response.headers.get('content-type', 'unknown')

            if file_path.endswith('.js'):
                response.headers['Content-Type'] = 'application/javascript; charset=utf-8'
                logger.info(f"Set JS MIME type for {path}: application/javascript")
            elif file_path.endswith('.wasm'):
                response.headers['Content-Type'] = 'application/wasm'
                logger.info(f"Set WASM MIME type for {path}: application/wasm")
            elif file_path.endswith('.css'):
                response.headers['Content-Type'] = 'text/css; charset=utf-8'
                logger.info(f"Set CSS MIME type for {path}: text/css")
            elif file_path.endswith('.html') or file_path.endswith('/') or file_path == '':
                response.headers['Content-Type'] = 'text/html; charset=utf-8'
                logger.info(f"Set HTML MIME type for {path}: text/html")
            else:
                logger.warning(f"No MIME type override for {path}, original: {original_content_type}")

        return response


class SimulationMimeTypeMiddleware(BaseHTTPMiddleware):
    """Middleware to fix MIME types for simulation assets."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Only process simulation and assets requests
        if request.url.path.startswith("/simulation/") or request.url.path.startswith("/assets/"):
            path = request.url.path.lower()

            # Override MIME types for specific extensions
            if path.endswith('.js'):
                response.headers['Content-Type'] = 'application/javascript; charset=utf-8'
                logger.info(f"Middleware: Fixed MIME type for JS file: {request.url.path}")
            elif path.endswith('.wasm'):
                response.headers['Content-Type'] = 'application/wasm'
                logger.info(f"Middleware: Fixed MIME type for WASM file: {request.url.path}")
            elif path.endswith('.css'):
                response.headers['Content-Type'] = 'text/css; charset=utf-8'
                logger.info(f"Middleware: Fixed MIME type for CSS file: {request.url.path}")

            # Always add COOP/COEP headers for simulation
            response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
            response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
            response.headers['Cross-Origin-Resource-Policy'] = 'cross-origin'

        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for the FastAPI application."""
    await app.state.daemon.start(
        serialport=app.state.args.serialport,
        sim=app.state.args.sim,
        scene=app.state.args.scene,
        headless=app.state.args.headless,
        kinematics_engine=app.state.args.kinematics_engine,
        check_collision=app.state.args.check_collision,
        wake_up_on_start=app.state.args.wake_up_on_start,
        localhost_only=app.state.args.localhost_only,
    )
    yield
    await app.state.app_manager.close()
    await app.state.daemon.stop(
        goto_sleep_on_stop=app.state.args.goto_sleep_on_stop,
    )


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Add simulation MIME type middleware
app.add_middleware(SimulationMimeTypeMiddleware)


router = APIRouter(prefix="/api")
router.include_router(apps.router)
router.include_router(daemon.router)
router.include_router(kinematics.router)
router.include_router(motors.router)
router.include_router(move.router)
router.include_router(simulation.router)
router.include_router(state.router)

app.include_router(router)


# Route to list available HTML/JS/CSS examples with links using Jinja2 template
@app.get("/")
async def list_examples(request: Request):
    """Render the dashboard."""
    files = [f for f in os.listdir(DASHBOARD_PAGES) if f.endswith(".html")]
    return templates.TemplateResponse(
        "dashboard.html", {"request": request, "files": files}
    )


@app.get("/simulation-debug")
async def simulation_debug():
    """Debug endpoint to check simulation files."""
    wasm_files = []
    if WASM_DIR.exists():
        for file_path in WASM_DIR.rglob("*"):
            if file_path.is_file():
                wasm_files.append(str(file_path.relative_to(WASM_DIR)))

    return {
        "wasm_dir_exists": WASM_DIR.exists(),
        "wasm_dir_path": str(WASM_DIR),
        "files": wasm_files[:20],  # Limit output
        "index_html_exists": (WASM_DIR / "index.html").exists(),
        "js_files": [f for f in wasm_files if f.endswith('.js')][:5],
        "test_urls": [
            "/simulation/",
            "/simulation/index.html",
            "/simulation/assets/index-BWx1u7ga.js"
        ]
    }


@app.get("/test-js-mime")
async def test_js_mime():
    """Test endpoint to check if JS MIME type is working."""
    js_file = WASM_DIR / "assets" / "index-BWx1u7ga.js"
    if js_file.exists():
        return {
            "file_exists": True,
            "file_size": js_file.stat().st_size,
            "test_url": "/simulation/assets/index-BWx1u7ga.js",
            "file_path": str(js_file)
        }
    else:
        return {"file_exists": False, "expected_path": str(js_file)}


@app.get("/assets/{asset_path:path}")
async def serve_assets_directly(asset_path: str, request: Request):
    """Serve assets directly from /assets/ path for compatibility."""
    logger.info(f"Serving asset directly: /assets/{asset_path}")

    # Construct the file path
    file_path = WASM_DIR / "assets" / asset_path

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Asset not found: {asset_path}")

    # Read the file
    with open(file_path, 'rb') as f:
        content = f.read()

    # Determine MIME type based on extension
    content_type = "application/octet-stream"  # default
    if asset_path.lower().endswith('.js'):
        content_type = "application/javascript; charset=utf-8"
    elif asset_path.lower().endswith('.wasm'):
        content_type = "application/wasm"
    elif asset_path.lower().endswith('.css'):
        content_type = "text/css; charset=utf-8"

    # Add required headers
    headers = {
        'Content-Type': content_type,
        'Cross-Origin-Opener-Policy': 'same-origin',
        'Cross-Origin-Embedder-Policy': 'require-corp',
        'Cross-Origin-Resource-Policy': 'cross-origin',
    }

    logger.info(f"Serving {asset_path} with MIME type: {content_type}")
    return Response(content=content, headers=headers)


app.mount(
    "/dashboard",
    StaticFiles(directory=str(DASHBOARD_PAGES), html=True),
    name="dashboard",
)

# Mount simulation files (middleware will handle MIME types and headers)
if WASM_DIR.exists():
    app.mount(
        "/simulation",
        StaticFiles(directory=str(WASM_DIR), html=True),
        name="simulation",
    )
    logger.info(f"Mounted simulation directory: {WASM_DIR}")
else:
    logger.warning(f"Simulation directory not found: {WASM_DIR}")


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
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the daemon in headless mode (default: False).",
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
        default="AnalyticalKinematics",
        choices=["Placo", "NN", "AnalyticalKinematics"],
        help="Set the kinematics engine (default: AnalyticalKinematics).",
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
