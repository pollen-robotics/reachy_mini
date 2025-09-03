"""Main API entry point server."""

from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..daemon import Daemon
from .routers import daemon, kinematics, motors, move, state


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for the FastAPI application."""
    app.state.daemon.start(sim=True, wake_up_on_start=False)
    yield
    app.state.daemon.stop(goto_sleep_on_stop=False)


app = FastAPI(
    lifespan=lifespan,
)
app.state.daemon = Daemon()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to your HF domain
    allow_methods=["*"],
    allow_headers=["*"],
)


router = APIRouter(prefix="/api")
router.include_router(daemon.router)
router.include_router(kinematics.router)
router.include_router(motors.router)
router.include_router(move.router)
router.include_router(state.router)


app.include_router(router)
