"""Movement-related API routes.

This exposes:
- goto
- play (wake_up, goto_sleep)
- stop running moves
- set_target and streaming set_target
"""

import asyncio
import json
import logging
from enum import Enum
from typing import Any, Coroutine
from uuid import UUID, uuid4

import numpy as np
import requests
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from reachy_mini.motion.recorded_move import RecordedMoves

from ....daemon.backend.abstract import Backend
from ..dependencies import get_backend, ws_get_backend
from ..models import AnyPose, FullBodyTarget

logger = logging.getLogger(__name__)

# Conversation app breathing coordination
CONVERSATION_APP_URL = "http://localhost:7860"

router = APIRouter(
    prefix="/move",
)


class InterpolationMode(str, Enum):
    """Interpolation modes for movement."""

    # TODO: This should be the same as for the backend

    LINEAR = "linear"
    MINJERK = "minjerk"
    EASE = "ease"
    CARTOON = "cartoon"


class GotoModelRequest(BaseModel):
    """Request model for the goto endpoint."""

    head_pose: AnyPose | None = None
    antennas: tuple[float, float] | None = None
    body_yaw: float | None = None
    duration: float
    interpolation: InterpolationMode = InterpolationMode.MINJERK

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "head_pose": {
                        "x": 0.0,
                        "y": 0.0,
                        "z": 0.0,
                        "roll": 0.0,
                        "pitch": 0.0,
                        "yaw": 0.0,
                    },
                    "antennas": [0.0, 0.0],
                    "duration": 2.0,
                    "interpolation": "minjerk",
                },
                {
                    "antennas": [0.0, 0.0],
                    "duration": 1.0,
                    "interpolation": "linear",
                },
            ],
        }
    }


class MoveUUID(BaseModel):
    """Model representing a unique identifier for a move task."""

    uuid: UUID


move_tasks: dict[UUID, asyncio.Task[None]] = {}

# Cache for RecordedMoves datasets to avoid re-fetching from HuggingFace
_dataset_cache: dict[str, RecordedMoves] = {}


def scale_conversation_breathing_down() -> bool:
    """Scale conversation app breathing amplitude down before executing move."""
    try:
        response = requests.post(
            f"{CONVERSATION_APP_URL}/api/breathing/scale_down",
            json={"target_scale": 0.2, "duration": 0.5},
            timeout=1.0,
        )
        response.raise_for_status()
        logger.info("Conversation app breathing scaled down to 20%")
        return True
    except Exception as e:
        logger.warning(f"Failed to scale down conversation app breathing: {e}")
        return False


def scale_conversation_breathing_up() -> None:
    """Scale conversation app breathing amplitude back up after move completes."""
    try:
        requests.post(
            f"{CONVERSATION_APP_URL}/api/breathing/scale_up",
            json={"target_scale": 1.0, "duration": 1.0},
            timeout=1.0,
        )
        logger.info("Conversation app breathing scaled up to 100%")
    except Exception as e:
        logger.warning(f"Failed to scale up conversation app breathing: {e}")


async def coordinated_move(coro: Coroutine[Any, Any, None]) -> None:
    """Execute move with breathing coordination.

    Scales breathing amplitude down to 20% during external moves,
    then scales back up to 100% when complete. This keeps the robot
    looking alive during moves instead of completely frozen.
    """
    scale_conversation_breathing_down()
    try:
        await coro
    finally:
        scale_conversation_breathing_up()


def create_move_task(coro: Coroutine[Any, Any, None]) -> MoveUUID:
    """Create a new move task using async task coroutine."""
    uuid = uuid4()

    task = asyncio.create_task(coro)

    task.add_done_callback(lambda t: move_tasks.pop(uuid, None))
    move_tasks[uuid] = task

    return MoveUUID(uuid=uuid)


async def stop_move_task(uuid: UUID) -> dict[str, str]:
    """Stop a running move task by cancelling it."""
    if uuid not in move_tasks:
        raise KeyError(f"Running move with UUID {uuid} not found")

    task = move_tasks.pop(uuid, None)
    assert task is not None

    if task:
        if task.cancel():
            try:
                await task
            except asyncio.CancelledError:
                pass

    return {
        "message": f"Stopped move with UUID: {uuid}",
    }


@router.get("/running")
async def get_running_moves() -> list[MoveUUID]:
    """Get a list of currently running move tasks."""
    return [MoveUUID(uuid=uuid) for uuid in move_tasks.keys()]


@router.post("/goto")
async def goto(
    goto_req: GotoModelRequest, backend: Backend = Depends(get_backend)
) -> MoveUUID:
    """Request a movement to a specific target."""
    return create_move_task(
        coordinated_move(
            backend.goto_target(
                head=goto_req.head_pose.to_pose_array() if goto_req.head_pose else None,
                antennas=np.array(goto_req.antennas) if goto_req.antennas else None,
                body_yaw=goto_req.body_yaw,
                duration=goto_req.duration,
            )
        )
    )


@router.post("/play/wake_up")
async def play_wake_up(backend: Backend = Depends(get_backend)) -> MoveUUID:
    """Request the robot to wake up."""
    return create_move_task(coordinated_move(backend.wake_up()))


@router.post("/play/goto_sleep")
async def play_goto_sleep(backend: Backend = Depends(get_backend)) -> MoveUUID:
    """Request the robot to go to sleep."""
    return create_move_task(coordinated_move(backend.goto_sleep()))


@router.post("/play/recorded-move-dataset/{dataset_name:path}/{move_name}")
async def play_recorded_move_dataset(
    dataset_name: str,
    move_name: str,
    backend: Backend = Depends(get_backend),
) -> MoveUUID:
    """Request the robot to play a predefined recorded move from a dataset."""
    # Use cached RecordedMoves to avoid re-fetching from HuggingFace
    if dataset_name not in _dataset_cache:
        print(f"[MoveRouter] Loading dataset: {dataset_name}")
        _dataset_cache[dataset_name] = RecordedMoves(dataset_name)
    else:
        print(f"[MoveRouter] Using cached dataset: {dataset_name}")

    move = _dataset_cache[dataset_name].get(move_name)
    return create_move_task(coordinated_move(backend.play_move(move)))


@router.post("/stop")
async def stop_move(uuid: MoveUUID) -> dict[str, str]:
    """Stop a running move task."""
    return await stop_move_task(uuid.uuid)


# --- FullBodyTarget streaming and single set_target ---
@router.post("/set_target")
async def set_target(
    target: FullBodyTarget,
    backend: Backend = Depends(get_backend),
) -> dict[str, str]:
    """POST route to set a single FullBodyTarget."""
    backend.set_target(
        head=target.target_head_pose.to_pose_array()
        if target.target_head_pose
        else None,
        antennas=np.array(target.target_antennas) if target.target_antennas else None,
        body_yaw=target.target_body_yaw if target.target_body_yaw is not None else 0.0,
    )
    return {"status": "ok"}


@router.websocket("/ws/set_target")
async def ws_set_target(
    websocket: WebSocket, backend: Backend = Depends(ws_get_backend)
) -> None:
    """WebSocket route to stream FullBodyTarget set_target calls."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            try:
                target = FullBodyTarget.model_validate_json(data)
                await set_target(target, backend)

            except Exception as e:
                await websocket.send_text(
                    json.dumps({"status": "error", "detail": str(e)})
                )
    except WebSocketDisconnect:
        pass
