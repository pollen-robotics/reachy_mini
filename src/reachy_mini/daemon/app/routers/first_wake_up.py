"""First wake-up API routes."""

import logging

from fastapi import APIRouter
from platformdirs import user_state_path

router = APIRouter(prefix="/first-wake-up")
logger = logging.getLogger(__name__)

_FIRST_WAKE_UP_FLAG = user_state_path("reachy_mini") / ".first_wake_up_done"


@router.get("/status")
async def get_first_wake_up_status() -> dict[str, bool]:
    """Check whether the robot has completed its first wake-up."""
    return {"is_completed": _FIRST_WAKE_UP_FLAG.exists()}


@router.post("/set")
async def set_first_wake_up_status(is_completed: bool) -> dict[str, bool]:
    """Mark (or unmark) the first wake-up as completed."""
    if is_completed:
        _FIRST_WAKE_UP_FLAG.parent.mkdir(parents=True, exist_ok=True)
        _FIRST_WAKE_UP_FLAG.touch()
    else:
        _FIRST_WAKE_UP_FLAG.unlink(missing_ok=True)
    logger.info("first_wake_up_done set to %s", is_completed)
    return {"is_completed": is_completed}
