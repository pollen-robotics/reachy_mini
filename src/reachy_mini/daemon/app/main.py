"""Main API entry point server."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException, Request, responses
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates

from reachy_mini.apps.manager import AppManager
from reachy_mini.daemon.app.routers import apps, daemon, kinematics, motors, move, state
from reachy_mini.daemon.daemon import Daemon

# from ...apps.manager import AppManager
# from ..daemon import Daemon
# from .routers import apps, daemon, kinematics, motors, move, state

EXAMPLES_WEB_DIR = Path(__file__).parent / "examples"
TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for the FastAPI application."""
    await app.state.daemon.start(
        sim=app.state.params["sim"],
        scene=app.state.params["scene"],
        wake_up_on_start=app.state.params["wake_up_on_start"],
    )
    yield
    await app.state.app_manager.close()
    await app.state.daemon.stop(
        goto_sleep_on_stop=app.state.params["goto_sleep_on_stop"]
    )


logging.basicConfig(level=logging.INFO)

app = FastAPI(
    lifespan=lifespan,
)
app.state.daemon = Daemon()
app.state.app_manager = AppManager()
app.state.params = {
    "sim": False,
    "scene": "empty",
    "wake_up_on_start": False,
    "goto_sleep_on_stop": False,
}

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
