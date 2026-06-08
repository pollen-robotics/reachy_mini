"""In-process daemon log stream for hosts without ``journalctl``.

On a real robot the daemon runs under systemd and its logs are read back
with ``journalctl -u reachy-mini-daemon``. On a developer machine or the
desktop / mockup daemon there is no journald, so ``journalctl`` is absent
and the log-streaming feature (the ``/logs/ws/daemon`` WebSocket and the
``subscribe_logs`` DataChannel command) used to fail outright with
``journalctl not found``.

This module is the portable fallback: a :class:`logging.Handler` that keeps
a bounded ring buffer of recent records and fans new records out to any
number of live async subscribers. Each line is formatted like
``journalctl --output short-iso`` (``"<iso-ts> <message>"``) so existing
client parsers work unchanged regardless of which source produced the line.

The handler is installed once at daemon startup (see
``daemon/app/main.py::run_app``). When ``journalctl`` is available the
real thing is still used; this buffer only backs the fallback path.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import datetime, timezone
from typing import AsyncIterator, Deque, Optional, Set

# Default ring-buffer size. Large enough to give a useful scrollback when a
# client first subscribes, small enough to stay negligible in memory.
_DEFAULT_CAPACITY = 2000


class _IsoFormatter(logging.Formatter):
    """Prefix each record with an ISO-8601 UTC timestamp and a space.

    Mirrors ``journalctl --output short-iso`` so consumers can split the
    line once on the first space into ``(timestamp, message)`` exactly as
    they do for the journalctl path.
    """

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        return f"{ts} {super().format(record)}"


class _RingBufferLogHandler(logging.Handler):
    """Keep recent log lines and push new ones to live subscribers.

    ``emit`` can be called from any thread (asyncio, uvicorn, GStreamer),
    so live delivery hops back onto the bound event loop via
    ``call_soon_threadsafe``. The buffer itself relies on ``deque`` being
    safe for single appends under CPython's GIL.
    """

    def __init__(self, capacity: int = _DEFAULT_CAPACITY) -> None:
        super().__init__()
        self._buffer: Deque[str] = deque(maxlen=capacity)
        self._subscribers: Set["asyncio.Queue[str]"] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Record the loop on which subscriber queues live."""
        self._loop = loop

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
        except Exception:  # never let logging raise into the caller
            return
        self._buffer.append(line)
        loop = self._loop
        if loop is None:
            return
        for queue in list(self._subscribers):
            loop.call_soon_threadsafe(self._safe_put, queue, line)

    @staticmethod
    def _safe_put(queue: "asyncio.Queue[str]", line: str) -> None:
        try:
            queue.put_nowait(line)
        except asyncio.QueueFull:
            # A slow consumer drops lines rather than stalling the daemon.
            pass

    def snapshot(self) -> list[str]:
        return list(self._buffer)

    def subscribe(self) -> "asyncio.Queue[str]":
        queue: "asyncio.Queue[str]" = asyncio.Queue(maxsize=10_000)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: "asyncio.Queue[str]") -> None:
        self._subscribers.discard(queue)


_HANDLER: Optional[_RingBufferLogHandler] = None


def install(capacity: int = _DEFAULT_CAPACITY) -> None:
    """Attach the ring-buffer handler to the root logger (idempotent)."""
    global _HANDLER
    if _HANDLER is not None:
        return
    handler = _RingBufferLogHandler(capacity=capacity)
    handler.setFormatter(_IsoFormatter("%(name)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(handler)
    _HANDLER = handler


def is_available() -> bool:
    """Return True once :func:`install` has run (the fallback can serve)."""
    return _HANDLER is not None


async def stream(
    idle_keepalive: Optional[float] = None,
) -> AsyncIterator[Optional[str]]:
    """Yield buffered lines, then live lines, until the consumer stops.

    Lines are formatted like ``journalctl --output short-iso``.

    If ``idle_keepalive`` is set, the iterator yields ``None`` whenever no
    new line arrives within that many seconds, letting a WebSocket consumer
    send a keepalive and notice a dropped peer. When ``None`` (the default)
    the iterator simply blocks until the next line; callers that manage
    liveness by cancelling the task (the DataChannel path) want this.

    Raises:
        RuntimeError: the buffer was never installed (``install`` not
            called); callers should fall back to their previous behaviour.

    """
    handler = _HANDLER
    if handler is None:
        raise RuntimeError("in-memory log buffer not installed")
    handler.bind_loop(asyncio.get_running_loop())
    # Subscribe before snapshotting so no line emitted in between is lost
    # (a handful of duplicates across the boundary is acceptable for logs).
    queue = handler.subscribe()
    try:
        for line in handler.snapshot():
            yield line
        while True:
            if idle_keepalive is None:
                yield await queue.get()
            else:
                try:
                    yield await asyncio.wait_for(queue.get(), timeout=idle_keepalive)
                except asyncio.TimeoutError:
                    yield None
    finally:
        handler.unsubscribe(queue)
