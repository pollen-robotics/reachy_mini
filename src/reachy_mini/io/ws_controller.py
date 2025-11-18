"""Async WebSocket Controller for remote control and streaming of the robot."""
import asyncio
import json
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import websockets

from reachy_mini.daemon.backend.abstract import Backend


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
            mov = cmd.get("movement", {})

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
                print("[Daemon] Error in goto_target:", e)
        else:
            print("[Daemon] Unknown command type:", typ)

    async def _run(self) -> None:
        """Run the WebSocket controller loop."""
        while not self.stop_flag:
            try:
                async with websockets.connect(self.ws_uri) as ws:
                    print("[WS] Connected to Space")
                    async for msg in ws:
                        try:
                            data = json.loads(msg)
                        except Exception as e:
                            print("[WS] Bad JSON:", e, "raw:", msg)
                            continue

                        # Now this is awaited inside the same loop
                        await self.on_command(data)

            except Exception as e:
                print("[WS] Connection failed:", e)
                # small backoff before reconnect
                await asyncio.sleep(1)

    def stop(self) -> None:
        """Stop the WebSocket controller."""
        self.stop_flag = True
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
