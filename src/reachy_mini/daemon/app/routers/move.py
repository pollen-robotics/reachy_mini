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
from typing import Any, Coroutine
from uuid import UUID, uuid4

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from huggingface_hub.errors import RepositoryNotFoundError
from pydantic import BaseModel

from reachy_mini.motion.recorded_move import RecordedMoves
from reachy_mini.utils.interpolation import InterpolationTechnique

from ....daemon.backend.abstract import Backend
from ..dependencies import get_backend, ws_get_backend
from ..models import AnyPose, FullBodyTarget

move_tasks: dict[UUID, asyncio.Task[None]] = {}
move_listeners: list[WebSocket] = []
logger = logging.getLogger(__name__)
MOVE_CANCEL_TIMEOUT_S = 2.0
MOVE_NOTIFICATION_TIMEOUT_S = 0.2


router = APIRouter(prefix="/move")


class GotoModelRequest(BaseModel):
    """Request model for the goto endpoint."""

    head_pose: AnyPose | None = None
    antennas: tuple[float, float] | None = None
    body_yaw: float | None = None
    duration: float
    interpolation: InterpolationTechnique = InterpolationTechnique.MIN_JERK

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
                    "body_yaw": 0.0,
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


def create_move_task(coro: Coroutine[Any, Any, None]) -> MoveUUID:
    """Create a new move task using async task coroutine."""
    uuid = uuid4()

    async def notify_listeners(message: str, details: str = "") -> None:
        # A notification is best effort and must not hold up move cleanup. One
        # budget covers the entire broadcast, regardless of listener count.
        ws = None
        notification_timeout = asyncio.timeout(MOVE_NOTIFICATION_TIMEOUT_S)
        try:
            async with notification_timeout:
                for ws in list(move_listeners):
                    try:
                        await ws.send_json(
                            {"type": message, "uuid": str(uuid), "details": details}
                        )
                    except (RuntimeError, WebSocketDisconnect):
                        if ws in move_listeners:
                            move_listeners.remove(ws)
        except TimeoutError:
            if not notification_timeout.expired():
                raise  # Preserve a real send failure; only our own deadline is best effort.
            logger.warning("Timed out sending %s for move %s", message, uuid)
            if ws in move_listeners:
                move_listeners.remove(ws)

    async def wrap_coro() -> None:
        try:
            await notify_listeners("move_started")
            await coro
            await notify_listeners("move_completed")
        except Exception as e:
            logger.exception("Move %s failed: %s", uuid, e)
            await notify_listeners("move_failed", details=str(e))
        except asyncio.CancelledError:
            await notify_listeners("move_cancelled")
            raise
        finally:
            move_tasks.pop(uuid, None)
            # Cancellation may arrive while the start notification is waiting,
            # before the supplied move coroutine was ever awaited.
            coro.close()

    task = asyncio.create_task(wrap_coro())
    move_tasks[uuid] = task

    def task_finished(done: asyncio.Task[None]) -> None:
        # Also handles cancellation before wrap_coro got its first timeslice.
        move_tasks.pop(uuid, None)
        coro.close()
        if not done.cancelled() and (error := done.exception()) is not None:
            logger.error("Move %s cleanup failed: %s", uuid, error)

    task.add_done_callback(task_finished)
    return MoveUUID(uuid=uuid)


async def _cancel_move_tasks(tasks: list[asyncio.Task[None]], timeout: float) -> None:
    for task in tasks:
        task.cancel()
    if not tasks:
        return
    # wait(), unlike wait_for(), does not wait past the deadline for a task
    # that suppresses cancellation. Pending tasks retain their registry entries.
    done, pending = await asyncio.wait(tasks, timeout=timeout)
    for task in done:
        if not task.cancelled():
            task.result()
    if pending:
        raise TimeoutError(
            f"{len(pending)} move task(s) did not stop within {timeout}s"
        )


async def cancel_all_move_tasks(timeout: float = MOVE_CANCEL_TIMEOUT_S) -> None:
    """Cancel registered moves together and await cleanup within one deadline."""
    await _cancel_move_tasks(list(move_tasks.values()), timeout)


async def stop_move_task(uuid: UUID) -> dict[str, str]:
    """Stop a move, retaining ownership until its coroutine actually finishes."""
    if uuid not in move_tasks:
        raise KeyError(f"Running move with UUID {uuid} not found")
    await _cancel_move_tasks([move_tasks[uuid]], MOVE_CANCEL_TIMEOUT_S)
    return {"message": f"Stopped move with UUID: {uuid}"}


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
        backend.goto_target(
            head=goto_req.head_pose.to_pose_array() if goto_req.head_pose else None,
            antennas=np.array(goto_req.antennas) if goto_req.antennas else None,
            body_yaw=goto_req.body_yaw,
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


@router.get("/recorded-move-datasets/list/{dataset_name:path}")
async def list_recorded_move_dataset(
    dataset_name: str,
) -> list[str]:
    """List available recorded moves in a dataset."""
    try:
        moves = RecordedMoves(dataset_name)
    except RepositoryNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return moves.list_moves()


@router.post("/play/recorded-move-dataset/{dataset_name:path}/{move_name}")
async def play_recorded_move_dataset(
    dataset_name: str,
    move_name: str,
    backend: Backend = Depends(get_backend),
) -> MoveUUID:
    """Request the robot to play a predefined recorded move from a dataset."""
    try:
        recorded_moves = RecordedMoves(dataset_name)
    except RepositoryNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    try:
        move = recorded_moves.get(move_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return create_move_task(backend.play_move(move))


@router.post("/stop")
async def stop_move(uuid: MoveUUID) -> dict[str, str]:
    """Stop a running move task."""
    return await stop_move_task(uuid.uuid)


@router.websocket("/ws/updates")
async def ws_move_updates(
    websocket: WebSocket,
) -> None:
    """WebSocket route to stream move updates."""
    await websocket.accept()
    try:
        move_listeners.append(websocket)
        while True:
            _ = await websocket.receive_text()
    except WebSocketDisconnect:
        move_listeners.remove(websocket)


# --- FullBodyTarget streaming and single set_target ---
@router.post("/set_target")
async def set_target(
    target: FullBodyTarget,
    backend: Backend = Depends(get_backend),
) -> dict[str, str]:
    """POST route to set a single FullBodyTarget."""
    if backend.is_move_running:
        # Avoid fighting with the daemon while a trajectory is running
        backend.logger.warning("Ignoring set_target request: move already running.")
        return {"status": "ignored", "reason": "move_running"}
    backend.set_target(
        head=target.target_head_pose.to_pose_array()
        if target.target_head_pose
        else None,
        antennas=np.array(target.target_antennas) if target.target_antennas else None,
        body_yaw=target.target_body_yaw,
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


@router.websocket("/ws/raw/write")
async def write(
    websocket: WebSocket,
    backend: Backend = Depends(ws_get_backend),
) -> None:
    """WebSocket endpoint to stream raw packet to the serialport and return any response buffer.

    Returns an empty bytes if no response is received.
    """
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_bytes()
            raw_response_packet: bytes = backend.write_raw_packet(data)
            await websocket.send_bytes(raw_response_packet)
    except WebSocketDisconnect:
        pass
