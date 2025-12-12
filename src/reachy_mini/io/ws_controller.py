"""Async WebSocket Controller for remote control and streaming of the robot."""

import asyncio
import json
import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from websockets.asyncio.client import ClientConnection, connect

from reachy_mini.daemon.backend.abstract import Backend

logger = logging.getLogger("reachy_mini.ws_controller")


@dataclass
class Movement:
    """Movement data for the WebSocket controller."""

    name: str
    x: float = 0
    y: float = 0
    z: float = 0
    roll: float = 0
    pitch: float = 0
    yaw: float = 0
    body_yaw: float = 0
    left_antenna: Optional[float] = None
    right_antenna: Optional[float] = None
    duration: float = 1.0


class AsyncWebSocketController:
    """WebSocket controller for remote control and streaming of the robot."""

    ws_uri: str
    backend: Backend
    loop: asyncio.AbstractEventLoop
    thread: threading.Thread
    stop_flag: bool

    def __init__(self, ws_uri: str, backend: Backend) -> None:
        """Initialize the WebSocket controller."""
        self.ws_uri = ws_uri
        self.backend = backend
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.stop_flag = False
        self.thread.start()

    def _run_loop(self) -> None:
        """Run the WebSocket controller loop."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._run())

    async def on_command(self, cmd: Dict[str, Any]) -> None:
        """Handle a command from the WebSocket."""
        typ = cmd.get("type")

        if typ == "movement":
            logger.debug("[Daemon] Movement command received")
            mov = cmd.get("movement", {})
            logger.debug("[Daemon] Movement command: %s", mov)

            head = mov.get("head")
            if head is not None:
                head_arr = np.array(head, dtype=float).reshape(4, 4)
            else:
                head_arr = None

            antennas = mov.get("antennas")
            if antennas is not None:
                antennas_arr = np.array(antennas, dtype=float)
            else:
                antennas_arr = None

            try:
                await self.backend.goto_target(
                    head=head_arr,
                    antennas=antennas_arr,
                    duration=mov.get("duration", 1.0),
                    body_yaw=mov.get("body_yaw", 0.0),
                )
            except Exception as e:
                logger.debug("[Daemon] Error in goto_target: %s", e)
        elif typ == "ping":
            logger.debug("[Daemon] Ping received")
            return
        else:
            logger.debug("[Daemon] Unknown command type: %s", typ)

    async def _run(self) -> None:
        """Run the WebSocket controller loop."""
        while not self.stop_flag:
            try:
                ws: ClientConnection
                async with connect(self.ws_uri, ping_interval=5, ping_timeout=10) as ws:
                    logger.info("[WS] Connected to Space")
                    async for msg in ws:
                        try:
                            data = json.loads(msg)
                        except Exception as e:
                            logger.debug("[WS] Bad JSON: %s raw: %s", e, msg)
                            continue

                        # Now this is awaited inside the same loop
                        await self.on_command(data)

            except Exception as e:
                logger.info("[WS] Connection failed: %s", e)
                # small backoff before reconnect
                await asyncio.sleep(1)

    def stop(self) -> None:
        """Stop the WebSocket controller."""
        self.stop_flag = True
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(lambda: None)
