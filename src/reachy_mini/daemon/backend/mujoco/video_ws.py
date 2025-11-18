"""Async WebSocket Frame Sender for the Mujoco backend."""
import asyncio
import threading
from queue import Empty, Queue
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import websockets


class AsyncWebSocketFrameSender:
    """Async WebSocket frame sender for the Mujoco backend."""

    ws_uri: str
    queue: "Queue[bytes]"
    loop: asyncio.AbstractEventLoop
    thread: threading.Thread
    connected: threading.Event
    stop_flag: bool

    def __init__(self, ws_uri: str) -> None:
        """Initialize the WebSocket frame sender."""
        self.ws_uri = ws_uri
        self.queue = Queue() 
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.connected = threading.Event()
        self.stop_flag = False
        self.thread.start()

    def _run_loop(self) -> None:
        """Run the WebSocket frame sender loop."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._run())

    async def _run(self) -> None:
        """Run the WebSocket frame sender loop."""
        while not self.stop_flag:
            try:
                async with websockets.connect(self.ws_uri) as ws:
                    print("[WS] Connected to Space")
                    self.connected.set()

                    while not self.stop_flag:
                        try:
                            frame = self.queue.get(timeout=0.1)
                        except Empty:
                            await asyncio.sleep(0.01)
                            continue

                        try:
                            await ws.send(frame)
                        except Exception as e:
                            print("[WS] Send error:", e)
                            break

            except Exception as e:
                print("[WS] Connection failed:", e)
                await asyncio.sleep(1)

            self.connected.clear()

    def send_frame(self, frame: npt.NDArray[np.uint8]) -> None:
        """Send a frame to the WebSocket.

        This method is called from the MuJoCo thread. It is non-blocking.

        Args:
            frame (np.ndarray): The frame to be sent, in RGB format.

        """
        if not self.stop_flag:
            # Encode frame to JPEG bytes for websocket binary send.
            ok: bool
            jpeg_bytes: Any
            ok, jpeg_bytes = cv2.imencode(
                ".jpg",
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), 80],
            )
            if not ok:
                return

            data = jpeg_bytes.tobytes()
            self.queue.put(data)

    def close(self) -> None:
        """Close the WebSocket frame sender."""
        self.stop_flag = True
        self.loop.call_soon_threadsafe(self.loop.stop)
