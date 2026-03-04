"""WebSocket server for Reachy Mini.

Provides a FastAPI WebSocket endpoint at /ws/sdk for SDK<->daemon communication.
Each connected client gets a dedicated asyncio.Queue for outbound messages.
The backend's 50 Hz publishers push state into all client queues via a
thread-safe bridge (loop.call_soon_threadsafe).

Client->Server messages use {"type": "...", ...} (parsed via command_adapter).

Server->Client messages are Pydantic models serialized to JSON, e.g.:
    {"type": "joint_positions", "head_joint_positions": [...], ...}
    {"type": "task_progress", "uuid": "...", "finished": true, ...}
"""

import asyncio
import logging
import threading
from datetime import datetime
from typing import Any

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from reachy_mini.daemon.backend.abstract import Backend
from reachy_mini.io.abstract import AbstractServer
from reachy_mini.io.protocol import (
    AnyCommand,
    GotoTaskRequest,
    PlayMoveTaskRequest,
    TaskProgress,
    TaskRequest,
    message_adapter,
)
from reachy_mini.io.publisher import Publisher

logger = logging.getLogger(__name__)


class WSServer(AbstractServer):
    """WebSocket server for Reachy Mini."""

    def __init__(self, backend: Backend) -> None:
        """Initialize the WebSocket server."""
        self.backend = backend
        self._cmd_event = threading.Event()

        # Connected client queues (populated when clients connect via handle_client)
        self._clients: set[asyncio.Queue[str]] = set()

        # Captured lazily in handle_client() from uvicorn's event loop
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self) -> None:
        """Wire publishers into the backend.

        The event loop is captured lazily when the first client connects
        (in handle_client), so this can be called from any thread.
        """
        # All state models include a "type" field, so a single broadcast
        # publisher works for every topic.
        publisher = Publisher(self._broadcast)
        self.backend.set_joint_positions_publisher(publisher)
        self.backend.set_pose_publisher(publisher)
        self.backend.set_imu_publisher(publisher)
        self.backend.set_recording_publisher(publisher)

    def stop(self) -> None:
        """Stop the WebSocket server."""
        self._clients.clear()
        self._loop = None

    def command_received_event(self) -> threading.Event:
        """Return the event that is set when a command is received."""
        return self._cmd_event

    # ------------------------------------------------------------------
    # Broadcasting
    # ------------------------------------------------------------------

    def _broadcast(self, msg: str) -> None:
        """Thread-safe broadcast to all connected client queues."""
        if self._loop is None:
            return

        def _put(msg: str = msg) -> None:
            for q in list(self._clients):
                try:
                    q.put_nowait(msg)
                except asyncio.QueueFull:
                    try:
                        q.get_nowait()
                        q.put_nowait(msg)
                    except (asyncio.QueueEmpty, asyncio.QueueFull):
                        pass

        self._loop.call_soon_threadsafe(_put)

    def publish_status(self, json_str: str) -> None:
        """Publish daemon status to all connected clients.

        Called from the daemon's 1 Hz status thread.
        """
        self._broadcast(json_str)

    # ------------------------------------------------------------------
    # WebSocket client handler (called from the FastAPI router)
    # ------------------------------------------------------------------

    async def handle_client(self, websocket: WebSocket) -> None:
        """Handle a single SDK WebSocket client connection."""
        # Capture the event loop on the first client connection.
        # This must happen here (not in start()) because start() may be called
        # from a different thread/loop than the one uvicorn runs on.
        if self._loop is None:
            self._loop = asyncio.get_running_loop()

        await websocket.accept()

        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=100)
        self._clients.add(queue)

        send_task = asyncio.create_task(self._send_loop(websocket, queue))
        try:
            await self._recv_loop(websocket)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"WS client error: {e}")
        finally:
            send_task.cancel()
            self._clients.discard(queue)

    async def _send_loop(self, websocket: WebSocket, queue: asyncio.Queue[str]) -> None:
        """Forward queued messages to the WebSocket client."""
        while True:
            msg = await queue.get()
            await websocket.send_text(msg)

    async def _recv_loop(self, websocket: WebSocket) -> None:
        """Receive and dispatch client messages."""
        while True:
            raw = await websocket.receive_text()

            try:
                msg = message_adapter.validate_json(raw)
            except Exception as e:
                logger.warning(f"WS invalid message: {e}")
                continue

            if isinstance(msg, TaskRequest):
                await self._handle_task_request(msg)
            else:
                self._handle_command(msg)

    # ------------------------------------------------------------------
    # Command handling (delegates to Backend.process_command)
    # ------------------------------------------------------------------

    def _handle_command(self, cmd: AnyCommand) -> None:
        """Dispatch a validated command through Backend.process_command."""
        def send(resp: dict[str, Any]) -> None:
            pass  # SDK commands are fire-and-forget

        self.backend.process_command(cmd, send_response=send)
        self._cmd_event.set()

    async def _handle_task_request(self, task_req: TaskRequest) -> None:
        """Handle a task request (goto, play_move) and broadcast progress."""
        if isinstance(task_req.req, GotoTaskRequest):
            req = task_req.req

            async def run_task() -> None:
                error = None
                try:
                    await self.backend.goto_target(
                        head=np.array(req.head).reshape(4, 4) if req.head else None,
                        antennas=np.array(req.antennas) if req.antennas else None,
                        duration=req.duration,
                        method=req.method,
                        body_yaw=req.body_yaw,
                    )
                except Exception as e:
                    error = str(e)

                progress = TaskProgress(
                    uuid=task_req.uuid,
                    finished=True,
                    error=error,
                    timestamp=datetime.now(),
                )
                self._broadcast(progress.model_dump_json())

            asyncio.create_task(run_task())

        elif isinstance(task_req.req, PlayMoveTaskRequest):
            logger.warning("PlayMoveTaskRequest not yet implemented over WS")

        else:
            logger.error(
                f"Unknown task request type: {task_req.req.__class__.__name__}"
            )
