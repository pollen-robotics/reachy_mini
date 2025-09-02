import asyncio
from enum import Enum
from typing import Coroutine
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ....daemon.backend.abstract import Backend
from ..dependencies import get_backend
from ..models import AnyPose

router = APIRouter(
    prefix="/move",
)


class InterpolationMode(str, Enum):
    LINEAR = "linear"
    MINJERK = "minjerk"
    EASE = "ease"
    CARTOON = "cartoon"


class GotoModelRequest(BaseModel):
    head_pose: AnyPose | None = None
    antennas: tuple[float, float] | None = None
    duration: float
    interpolation: InterpolationMode = InterpolationMode.MINJERK


class MoveUUID(BaseModel):
    uuid: UUID


move_tasks: dict[UUID, asyncio.Task] = {}


def create_move_task(coro: Coroutine) -> MoveUUID:
    uuid = uuid4()

    task = asyncio.create_task(coro)

    task.add_done_callback(lambda t: move_tasks.pop(uuid, None))
    move_tasks[uuid] = task

    return MoveUUID(uuid=uuid)


async def stop_move_task(uuid: UUID):
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
    return create_move_task(
        backend.async_goto_target(
            head=goto_req.head_pose.to_pose_array() if goto_req.head_pose else None,
            antennas=list(goto_req.antennas) if goto_req.antennas else None,
            duration=goto_req.duration,
        )
    )


@router.post("/play/wake_up")
async def play_wake_up(backend: Backend = Depends(get_backend)) -> MoveUUID:
    return create_move_task(backend.wake_up())


@router.post("/play/goto_sleep")
async def play_goto_sleep(backend: Backend = Depends(get_backend)) -> MoveUUID:
    return create_move_task(backend.goto_sleep())


@router.post("/stop")
async def stop_move(uuid: MoveUUID):
    return await stop_move_task(uuid.uuid)
