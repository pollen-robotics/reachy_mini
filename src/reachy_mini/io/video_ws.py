"""Async WebSocket Frame Sender."""

import asyncio
import logging
import threading
from queue import Empty, Full, Queue
from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger("reachy_mini.io.video_ws")


class AsyncWebSocketFrameSender:
    """Async WebSocket frame sender."""

    ws_uri: str
    queue: "Queue[bytes]"
    loop: asyncio.AbstractEventLoop
    thread: threading.Thread
    connected: threading.Event
    stop_flag: bool
    _last_frame: Optional[npt.NDArray[np.uint8]]

    def __init__(self, ws_uri: str) -> None:
        """Initialize the WebSocket frame sender."""
        self.ws_uri = ws_uri
        self.queue = Queue(maxsize=2)
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.connected = threading.Event()
        self.stop_flag = False
        self._last_frame = None
        self.thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._run())

    def _clear_queue(self) -> None:
        """Empty the queue so we don't send 10 seconds of old video on reconnect."""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except Empty:
                break

    async def _run(self) -> None:
        """Run the WebSocket frame sender loop."""
        while not self.stop_flag:
            try:
                ws: ClientConnection
                async with connect(
                    self.ws_uri,
                    ping_interval=5,  # Every 5 seconds is plenty
                    ping_timeout=10,  # Give it 10s to respond
                    close_timeout=1,  # Don't wait long for polite closes
                ) as ws:
                    logger.info("[WS Video] Connected to Space")
                    self.connected.set()
                    self._clear_queue()  # Ensure we start fresh

                    while not self.stop_flag:
                        try:
                            frame = self.queue.get_nowait()

                            await ws.send(frame)

                        except Empty:
                            # Queue is empty, just yield to event loop
                            await asyncio.sleep(0.05)
                            continue

                        except Exception as e:
                            logger.error(f"[WS Video] Send error: {e}")
                            break

            except ConnectionClosed as e:
                # Common network errors, retry quickly
                logger.error(
                    f"[WS Video] Connection lost ({type(e).__name__}). Retrying..."
                )
                self.connected.clear()
                self._last_frame = None
                await asyncio.sleep(0.5)  # Wait briefly before reconnecting

            except Exception as e:
                logger.error(f"[WS Video] Unexpected error: {e}")
                await asyncio.sleep(1)

            self.connected.clear()
            self._last_frame = None

    def send_frame(self, frame: npt.NDArray[np.uint8]) -> None:
        """Send a frame to the WebSocket (Non-blocking)."""
        if not self.stop_flag:
            # 1. Frame Deduplication
            if self._last_frame is not None:
                if np.array_equal(frame, self._last_frame):
                    return
            self._last_frame = frame.copy()

            # 2. Encode
            ok, jpeg_bytes = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 80],
            )
            if not ok:
                return

            data = jpeg_bytes.tobytes()

            # If queue is full (network lagging), remove the old frame and put the new one.
            try:
                self.queue.put_nowait(data)
            except Full:
                # Queue is full, network is likely slower than camera.
                # Drop the OLDEST frame to make room for the NEWEST.
                try:
                    self.queue.get_nowait()  # Pop old frame
                    self.queue.put_nowait(data)  # Push new frame
                except Empty:
                    logger.error(
                        "[WS Video] Queue is full and empty, this should not happen."
                    )

    def close(self) -> None:
        """Close the WebSocket frame sender."""
        self.stop_flag = True
        self.loop.call_soon_threadsafe(self.loop.stop)
