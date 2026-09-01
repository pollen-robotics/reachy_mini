"""Match daemon camera frames to capture-time monotonic timestamps."""

from __future__ import annotations

import threading
from collections import OrderedDict


class CameraFrameTimestamps:
    """Bounded, thread-safe timestamp handoff keyed by camera frame offset.

    The daemon's outgoing camera pipeline and its local IPC readers share the
    buffer ``offset`` even though ``unixfdsrc`` does not preserve the original
    PTS. The media pipeline records the time here before IPC; the face tracker
    consumes it after receiving the same frame offset.
    """

    def __init__(self, capacity: int = 128, max_source_age_s: float = 2.0) -> None:
        """Create a bounded registry and maximum accepted producer-frame age."""
        self._capacity = max(int(capacity), 1)
        self._max_source_age_s = max(float(max_source_age_s), 0.0)
        self._timestamps: OrderedDict[int, float] = OrderedDict()
        self._lock = threading.Lock()

    def record(
        self,
        frame_offset: int,
        handoff_monotonic: float,
        *,
        frame_running_time_s: float | None = None,
        pipeline_running_time_s: float | None = None,
    ) -> float:
        """Record one frame and return the selected monotonic timestamp.

        When the producer PTS is usable, its age relative to the producer
        pipeline clock is subtracted from the handoff time. Otherwise the
        handoff time itself is the closest reliable pre-IPC timestamp.
        """
        timestamp = float(handoff_monotonic)
        if frame_running_time_s is not None and pipeline_running_time_s is not None:
            age_s = float(pipeline_running_time_s) - float(frame_running_time_s)
            if 0.0 <= age_s <= self._max_source_age_s:
                timestamp -= age_s

        with self._lock:
            self._timestamps[int(frame_offset)] = timestamp
            self._timestamps.move_to_end(int(frame_offset))
            while len(self._timestamps) > self._capacity:
                self._timestamps.popitem(last=False)
        return timestamp

    def pop(self, frame_offset: int) -> float | None:
        """Consume the timestamp associated with ``frame_offset`` if present."""
        with self._lock:
            return self._timestamps.pop(int(frame_offset), None)

    def clear(self) -> None:
        """Discard timestamps from a stopped or rebuilt media pipeline."""
        with self._lock:
            self._timestamps.clear()
