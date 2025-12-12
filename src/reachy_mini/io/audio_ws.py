"""Async WebSocket Audio Streamer."""

import asyncio
import logging
import threading
import time
from queue import Empty, Queue
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from websockets.asyncio.client import ClientConnection, connect

logger = logging.getLogger("reachy_mini.io.audio_ws")


class AsyncWebSocketAudioStreamer:
    """Async WebSocket audio streamer with send and receive support."""

    ws_uri: str
    send_queue: "Queue[bytes]"
    recv_queue: "Queue[bytes]"
    loop: asyncio.AbstractEventLoop
    thread: threading.Thread
    connected: threading.Event
    stop_flag: bool
    keep_alive_interval: float

    # --- CONFIGURATION ---
    # Target ~2048 samples per packet (approx 128ms)
    # 2048 samples * 2 bytes (int16) = 4096 bytes
    BATCH_SIZE_BYTES = 4096
    # Don't hold audio longer than 200ms even if buffer isn't full
    BATCH_TIMEOUT = 0.2

    def __init__(self, ws_uri: str, keep_alive_interval: float = 2.0) -> None:
        """Initialize the WebSocket audio streamer.

        Args:
            ws_uri: WebSocket URI to connect to.
            keep_alive_interval: Interval in seconds to send keep-alive pings
                when no audio is flowing.

        """
        self.ws_uri = ws_uri
        self.send_queue = Queue()
        self.recv_queue = Queue()
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.connected = threading.Event()
        self.stop_flag = False
        self.keep_alive_interval = keep_alive_interval
        self.thread.start()

    def _run_loop(self) -> None:
        """Run the WebSocket streamer loop in a background thread."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._run())

    async def _run(self) -> None:
        """Run the main reconnect loop."""
        while not self.stop_flag:
            try:
                async with connect(self.ws_uri) as ws:
                    logger.info("[WS-AUDIO] Connected to Space")
                    self.connected.set()

                    send_task = asyncio.create_task(self._send_loop(ws))
                    recv_task = asyncio.create_task(self._recv_loop(ws))

                    done, pending = await asyncio.wait(
                        {send_task, recv_task},
                        return_when=asyncio.FIRST_EXCEPTION,
                    )

                    # Cancel the other task if one fails or finishes
                    for task in pending:
                        task.cancel()
                        try:
                            await task
                        except Exception:
                            pass

            except Exception as e:
                logger.info(f"[WS-AUDIO] Connection failed: {e}")
                await asyncio.sleep(1.0)

            self.connected.clear()

    async def _send_loop(self, ws: ClientConnection) -> None:
        """Send outgoing audio chunks and keep-alive pings.

        To avoid audible artifacts, this method aggregates small chunks into larger batches before sending.
        """
        last_activity = time.time()

        # Buffer to hold small chunks
        batch_buffer = bytearray()
        # Track when we started filling this specific batch
        batch_start_time = time.time()

        while not self.stop_flag:
            try:
                # 1. Try to pull data from the queue
                # Use a short timeout so we can check time-based conditions frequently
                chunk = self.send_queue.get(timeout=0.01)

                # If this is the first chunk in the buffer, reset timer
                if len(batch_buffer) == 0:
                    batch_start_time = time.time()

                batch_buffer.extend(chunk)

            except Empty:
                pass
            except Exception as e:
                logger.info(f"[WS-AUDIO] Queue error: {e}")
                break

            # 2. Check if we should send the batch
            now = time.time()

            # Condition A: Buffer is full enough (Size based)
            is_full = len(batch_buffer) >= self.BATCH_SIZE_BYTES

            # Condition B: Buffer has data, but it's getting too old (Time based)
            # This ensures that if the robot says a short word, it sends it
            # after 100ms instead of waiting forever for more data.
            is_timed_out = (len(batch_buffer) > 0) and (
                (now - batch_start_time) > self.BATCH_TIMEOUT
            )
            if is_full or is_timed_out:
                try:
                    # Send the aggregated buffer
                    await ws.send(batch_buffer)  # type: ignore

                    # Reset
                    batch_buffer = bytearray()
                    last_activity = now
                except Exception as e:
                    logger.info(f"[WS-AUDIO] Send error: {e}")
                    break

            # 3. Keep-Alive Ping (Only if completely idle)
            # We only ping if the buffer is empty AND we haven't sent anything recently
            if (
                len(batch_buffer) == 0
                and (now - last_activity) > self.keep_alive_interval
            ):
                try:
                    await ws.send("ping")
                    last_activity = now
                    logger.debug("[WS-AUDIO] Sent keep-alive ping")
                except Exception as e:
                    logger.info(f"[WS-AUDIO] Ping failed: {e}")
                    break

            # Tiny sleep to yield control if we are just spinning
            if len(batch_buffer) == 0:
                await asyncio.sleep(0.001)

    async def _recv_loop(self, ws: ClientConnection) -> None:
        """Receive incoming audio chunks."""
        while not self.stop_flag:
            try:
                msg = await ws.recv()
            except Exception as e:
                logger.info(f"[WS-AUDIO] Receive error: {e}")
                break

            if isinstance(msg, bytes):
                try:
                    self.recv_queue.put_nowait(msg)
                except Exception as e:
                    logger.debug(f"[WS-AUDIO] Failed to enqueue received audio: {e}")
            else:
                logger.debug(f"[WS-AUDIO] Received non-binary message: {msg}")

    # ------------------------
    # Public API
    # ------------------------

    def send_audio_chunk(
        self,
        audio: Union[bytes, npt.NDArray[np.int16], npt.NDArray[np.float32]],
    ) -> None:
        """Queue an audio chunk to be sent.

        Args:
            audio: Either raw bytes or a numpy array of int16 or float32.
                   Float32 arrays are assumed to be in [-1, 1] and will
                   be converted to int16 PCM.

        """
        if self.stop_flag:
            return

        if isinstance(audio, bytes):
            data = audio
        else:
            # Convert only if needed
            arr = np.asarray(audio)
            # Handle stereo if accidentally passed (take channel 0)
            if arr.ndim > 1:
                arr = arr[:, 0]

            if arr.dtype == np.float32 or arr.dtype == np.float64:
                # If any value is above 1 or below -1, scale the entire array so the max abs value is 1 or less
                max_abs = np.max(np.abs(arr))
                if max_abs > 1.0:
                    arr = arr / max_abs
                # Convert float audio [-1,1] to int16 PCM
                arr = np.clip(arr, -1.0, 1.0)
                arr = (arr * 32767.0).astype(np.int16)  # type: ignore
            elif arr.dtype != np.int16:
                arr = arr.astype(np.int16)

            data = arr.tobytes()

        self.send_queue.put(data)

    def get_audio_chunk(
        self, timeout: Optional[float] = 0.01
    ) -> Optional[npt.NDArray[np.float32]]:
        """Retrieve a received audio chunk, if any."""
        try:
            if timeout == 0:
                audio_bytes = self.recv_queue.get_nowait()
            else:
                audio_bytes = self.recv_queue.get(timeout=timeout)
            # bytes -> int16 -> float32 in [-1, 1]
            int16_arr = np.frombuffer(audio_bytes, dtype=np.int16)
            float_arr = int16_arr.astype(np.float32) / 32767.0
            return float_arr
        except Empty:
            return None

    def close(self) -> None:
        """Close the WebSocket audio streamer."""
        self.stop_flag = True
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
