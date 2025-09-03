"""Movement-related API routes.

This exposes:
- goto
- play (wake_up, goto_sleep)
- stop running moves
- set_target and streaming set_target
"""

import asyncio
import json
from enum import Enum
from typing import Coroutine
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from ....daemon.backend.abstract import Backend
from ..dependencies import get_backend, ws_get_backend
from ..models import AnyPose, FullBodyTarget

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
    duration: float
    interpolation: InterpolationMode = InterpolationMode.MINJERK


class MoveUUID(BaseModel):
    """Model representing a unique identifier for a move task."""

    uuid: UUID


move_tasks: dict[UUID, asyncio.Task] = {}


def create_move_task(coro: Coroutine) -> MoveUUID:
    """Create a new move task using async task coroutine."""
    uuid = uuid4()

    task = asyncio.create_task(coro)

    task.add_done_callback(lambda t: move_tasks.pop(uuid, None))
    move_tasks[uuid] = task

    return MoveUUID(uuid=uuid)


async def stop_move_task(uuid: UUID):
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


@router.post("/goto")
async def goto(
    goto_req: GotoModelRequest, backend: Backend = Depends(get_backend)
) -> MoveUUID:
    """Request a movement to a specific target."""
    return create_move_task(
        backend.async_goto_target(
            head=goto_req.head_pose.to_pose_array() if goto_req.head_pose else None,
            antennas=list(goto_req.antennas) if goto_req.antennas else None,
            duration=goto_req.duration,
        )
    )


@router.post("/play/wake_up")
async def play_wake_up(backend: Backend = Depends(get_backend)) -> MoveUUID:
    """Request the robot to wake up."""
    return create_move_task(backend.wake_up())


@router.post("/play/goto_sleep")
async def play_goto_sleep(backend: Backend = Depends(get_backend)) -> MoveUUID:
    """Request the robot to go to sleep."""
    return create_move_task(backend.goto_sleep())


@router.post("/stop")
async def stop_move(uuid: MoveUUID):
    """Stop a running move task."""
    return await stop_move_task(uuid.uuid)


# --- FullBodyTarget streaming and single set_target ---
@router.post("/set_target")
async def set_target(
    target: FullBodyTarget,
    backend: Backend = Depends(get_backend),
) -> dict:
    """POST route to set a single FullBodyTarget."""
    backend.set_target(
        head=target.target_head_pose.to_pose_array()
        if target.target_head_pose
        else None,
        antennas=list(target.target_antennas) if target.target_antennas else None,
    )
    return {"status": "ok"}


@router.websocket("/ws/set_target")
async def ws_set_target(
    websocket: WebSocket, backend: Backend = Depends(ws_get_backend)
):
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
