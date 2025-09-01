from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI

from ..daemon import Daemon
from .routers import daemon, kinematics, state


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.daemon.start(sim=True, wake_up_on_start=False)
    yield
    app.state.daemon.stop(goto_sleep_on_stop=False)


app = FastAPI(
    lifespan=lifespan,
)
app.state.daemon = Daemon()


router = APIRouter(prefix="/api")
router.include_router(daemon.router)
router.include_router(kinematics.router)
router.include_router(state.router)


app.include_router(router)
