"""Feature-flagged daemon instrumentation helpers."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Any

INSTRUMENT_ENV_VAR = "REACHY_DAEMON_INSTRUMENT"


class InstrumentMode(str, Enum):
    """Supported daemon instrumentation modes."""

    OFF = "off"
    BASIC = "basic"
    TRACE = "trace"
    REMOTE = "remote"


def get_instrument_mode(raw: str | None = None) -> InstrumentMode:
    """Return the configured instrumentation mode."""
    value = (raw if raw is not None else os.getenv(INSTRUMENT_ENV_VAR, "basic")).strip().lower()
    try:
        return InstrumentMode(value)
    except ValueError:
        logging.getLogger(__name__).warning(
            "Invalid %s=%r, falling back to basic",
            INSTRUMENT_ENV_VAR,
            value,
        )
        return InstrumentMode.BASIC


def is_structured_mode(mode: InstrumentMode) -> bool:
    """Return whether logs should be emitted as JSON records."""
    return mode in {InstrumentMode.BASIC, InstrumentMode.TRACE, InstrumentMode.REMOTE}


class JsonLogFormatter(logging.Formatter):
    """Format log records as one compact JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a logging record as JSON."""
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event", record.getMessage()),
            "message": record.getMessage(),
            "attrs": getattr(record, "attrs", {}),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _make_handler(stream: Any, mode: InstrumentMode, log_level: str) -> logging.Handler:
    handler = logging.StreamHandler(stream)
    handler.setLevel(log_level)
    if is_structured_mode(mode):
        handler.setFormatter(JsonLogFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
    return handler


def configure_daemon_logging(log_level: str, log_file: str | None = None) -> InstrumentMode:
    """Configure root daemon logging and return the selected instrumentation mode."""
    mode = get_instrument_mode()
    effective_level = "WARNING" if mode is InstrumentMode.OFF else log_level

    root_logger = logging.getLogger()
    root_logger.setLevel(effective_level)
    root_logger.handlers.clear()
    root_logger.addHandler(_make_handler(sys.stderr, mode, effective_level))

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, mode="a")
        file_handler.setLevel(effective_level)
        if is_structured_mode(mode):
            file_handler.setFormatter(JsonLogFormatter())
        else:
            file_handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
        root_logger.addHandler(file_handler)

    logging.getLogger(__name__).info(
        "Daemon instrumentation configured",
        extra={
            "event": "instrumentation.configured",
            "attrs": {
                "mode": mode.value,
                "log_level": effective_level,
                "log_file": log_file,
            },
        },
    )
    return mode


class timing_event:
    """Context manager that emits a trace timing event when trace mode is enabled."""

    def __init__(self, name: str, **attrs: Any) -> None:
        """Create a timing event with a name and optional attributes."""
        self.name = name
        self.attrs = attrs
        self.started = 0.0

    def __enter__(self) -> "timing_event":
        """Start timing."""
        self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Emit a completed timing event when trace mode is active."""
        if get_instrument_mode() not in {InstrumentMode.TRACE, InstrumentMode.REMOTE}:
            return
        attrs = dict(self.attrs)
        attrs["duration_ms"] = round((time.perf_counter() - self.started) * 1000, 3)
        attrs["outcome"] = "error" if exc is not None else "ok"
        if exc is not None:
            attrs["error.type"] = type(exc).__name__
            attrs["error.message"] = str(exc)
        logging.getLogger(__name__).info(
            self.name,
            extra={
                "event": "span.end",
                "attrs": {
                    "name": self.name,
                    **attrs,
                },
            },
        )
