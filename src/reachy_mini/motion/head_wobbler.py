"""Audio-reactive head wobbler.

Receives PCM audio samples, analyses them via :class:`SwayRollRT`, and
dispatches 6-DOF movement offsets through a callback at the correct timing.

The wobbler runs a background thread that:

1. Consumes audio chunks from an internal queue.
2. Feeds them to :class:`SwayRollRT` to obtain per-hop sway dicts.
3. Applies each sway dict at the right wall-clock time via the callback.

Adapted from *reachy_mini_conversation_app*.
"""

import logging
import queue
import threading
import time
from collections.abc import Callable
from typing import Any

from numpy.typing import NDArray

from reachy_mini.motion.speech_tapper import HOP_MS, SwayRollRT

logger = logging.getLogger(__name__)

MOVEMENT_LATENCY_S = 0.2  # seconds between audio arrival and robot movement

# Type alias for the 6-DOF offset callback.
SpeechOffsets = tuple[float, float, float, float, float, float]


class HeadWobbler:
    """Convert PCM audio into time-synchronised head movement offsets."""

    def __init__(self, set_speech_offsets: Callable[[SpeechOffsets], None]) -> None:
        """Initialize the head wobbler with a callback for speech offsets."""
        self._apply_offsets = set_speech_offsets
        self._base_ts: float | None = None
        self._hops_done: int = 0

        self.audio_queue: queue.Queue[tuple[int, int, NDArray[Any]]] = queue.Queue()
        self.sway = SwayRollRT()

        self._state_lock = threading.Lock()
        self._sway_lock = threading.Lock()
        self._generation = 0

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed_pcm(self, pcm: NDArray[Any], sample_rate: int) -> None:
        """Thread-safe: enqueue raw PCM audio for processing.

        Args:
            pcm: Audio samples (any dtype/shape accepted by SwayRollRT).
            sample_rate: Sample rate of *pcm*.

        """
        with self._state_lock:
            generation = self._generation
        self.audio_queue.put((generation, sample_rate, pcm))

    def start(self) -> None:
        """Start the background processing thread (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._working_loop, daemon=True)
        self._thread.start()
        logger.debug("Head wobbler started")

    def stop(self) -> None:
        """Stop the background thread and reset offsets to zero."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._apply_offsets((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        logger.debug("Head wobbler stopped")

    def reset(self) -> None:
        """Reset internal state, drain queued audio, zero offsets."""
        with self._state_lock:
            self._generation += 1
            self._base_ts = None
            self._hops_done = 0

        # Drain queued audio from previous generation
        while True:
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except queue.Empty:
                break

        with self._sway_lock:
            self.sway.reset()

        self._apply_offsets((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _working_loop(self) -> None:
        hop_dt = HOP_MS / 1000.0

        while not self._stop_event.is_set():
            try:
                chunk_generation, sr, chunk = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                with self._state_lock:
                    current_generation = self._generation
                if chunk_generation != current_generation:
                    continue

                if self._base_ts is None:
                    with self._state_lock:
                        if self._base_ts is None:
                            self._base_ts = time.monotonic()

                with self._sway_lock:
                    results = self.sway.feed(chunk, sr)

                i = 0
                while i < len(results):
                    with self._state_lock:
                        if self._generation != current_generation:
                            break
                        base_ts = self._base_ts
                        hops_done = self._hops_done

                    if base_ts is None:
                        base_ts = time.monotonic()
                        with self._state_lock:
                            if self._base_ts is None:
                                self._base_ts = base_ts
                                hops_done = self._hops_done

                    target_time = base_ts + MOVEMENT_LATENCY_S + hops_done * hop_dt
                    now = time.monotonic()

                    # Drop frames if running behind
                    if now - target_time >= hop_dt:
                        lag_hops = int((now - target_time) / hop_dt)
                        drop = min(lag_hops, len(results) - i - 1)
                        if drop > 0:
                            with self._state_lock:
                                self._hops_done += drop
                            i += drop
                            continue

                    # Sleep if ahead of schedule
                    if target_time > now:
                        time.sleep(target_time - now)
                        with self._state_lock:
                            if self._generation != current_generation:
                                break

                    r = results[i]
                    offsets: SpeechOffsets = (
                        r["x_mm"] / 1000.0,
                        r["y_mm"] / 1000.0,
                        r["z_mm"] / 1000.0,
                        r["roll_rad"],
                        r["pitch_rad"],
                        r["yaw_rad"],
                    )

                    with self._state_lock:
                        if self._generation != current_generation:
                            break

                    self._apply_offsets(offsets)

                    with self._state_lock:
                        self._hops_done += 1
                    i += 1
            finally:
                self.audio_queue.task_done()

        logger.debug("Head wobbler thread exited")
