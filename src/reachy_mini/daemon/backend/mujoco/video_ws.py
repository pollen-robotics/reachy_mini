"""Async WebSocket Frame Sender for the Mujoco backend."""
import asyncio
import json
import logging
import threading
import time
from queue import Empty, Queue
from typing import Any, Optional

import cv2
import numpy as np
import numpy.typing as npt
import websockets

logger = logging.getLogger("reachy_mini.mujoco.video_ws")

class AsyncWebSocketFrameSender:
    """Async WebSocket frame sender for the Mujoco backend."""

    ws_uri: str
    queue: "Queue[bytes]"
    loop: asyncio.AbstractEventLoop
    thread: threading.Thread
    connected: threading.Event
    stop_flag: bool
    _last_frame: Optional[npt.NDArray[np.uint8]] # Store last frame for comparison

    def __init__(self, ws_uri: str) -> None:
        """Initialize the WebSocket frame sender."""
        self.ws_uri = ws_uri
        self.queue = Queue()
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.connected = threading.Event()
        self.stop_flag = False
        self._last_frame = None 
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
                    logger.info("[WS] Connected to Space")
                    self.connected.set()
                    
                    # Track last time we sent data
                    last_activity = time.time() 
                    keep_alive_interval = 5.0  # Seconds

                    while not self.stop_flag:
                        try:
                            # Wait briefly for a frame
                            frame = self.queue.get(timeout=0.1)
                            
                            # If we get here, we have a frame
                            await ws.send(frame)
                            last_activity = time.time()
                            
                        except Empty:
                            # No frame to send. Check if we need to ping.
                            now = time.time()
                            if (now - last_activity) > keep_alive_interval:
                                try:
                                    # Send a text-based JSON ping
                                    await ws.send(json.dumps({"type": "ping"}))
                                    last_activity = now
                                    logger.debug("[WS] Sent keep-alive ping") 
                                except Exception as e:
                                    logger.info(f"[WS] Ping failed: {e}")
                                    break
                            
                            await asyncio.sleep(0.01)
                            continue

                        except Exception as e:
                            logger.info(f"[WS] Send error: {e}")
                            break

            except Exception as e:
                logger.info(f"[WS] Connection failed: {e}")
                await asyncio.sleep(1)

            self.connected.clear()
            self._last_frame = None # Reset frame cache on disconnect

    def send_frame(self, frame: npt.NDArray[np.uint8]) -> None:
        """Send a frame to the WebSocket.

        This method is called from the MuJoCo thread. It is non-blocking.

        Args:
            frame (np.ndarray): The frame to be sent, in RGB format.

        """
        if not self.stop_flag:
            # 1. Frame Deduplication
            # Check if this frame is identical to the last one sent
            if self._last_frame is not None:
                if np.array_equal(frame, self._last_frame):
                    return # Skip encoding and sending

            # Store a copy for the next comparison
            self._last_frame = frame.copy()

            # 2. Encode and Queue
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