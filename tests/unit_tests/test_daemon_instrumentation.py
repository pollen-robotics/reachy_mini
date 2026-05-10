"""Tests for daemon instrumentation helpers."""

from __future__ import annotations

import json
import logging
from collections.abc import Generator
from pathlib import Path

import pytest

from reachy_mini.daemon.daemon import Daemon
from reachy_mini.daemon.instrumentation import (
    INSTRUMENT_ENV_VAR,
    InstrumentMode,
    configure_daemon_logging,
    get_instrument_mode,
    timing_event,
)


@pytest.fixture()
def restore_root_logging() -> Generator[None, None, None]:
    """Restore root logger handlers after instrumentation tests."""
    root = logging.getLogger()
    handlers = list(root.handlers)
    level = root.level
    yield
    for handler in root.handlers:
        handler.close()
    root.handlers.clear()
    root.handlers.extend(handlers)
    root.setLevel(level)


def test_get_instrument_mode_defaults_to_basic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default instrumentation mode is basic."""
    monkeypatch.delenv(INSTRUMENT_ENV_VAR, raising=False)

    assert get_instrument_mode() is InstrumentMode.BASIC


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("off", InstrumentMode.OFF),
        ("basic", InstrumentMode.BASIC),
        ("trace", InstrumentMode.TRACE),
        ("remote", InstrumentMode.REMOTE),
        (" TRACE ", InstrumentMode.TRACE),
    ],
)
def test_get_instrument_mode_parses_known_modes(raw: str, expected: InstrumentMode) -> None:
    """Known instrumentation mode values are normalized."""
    assert get_instrument_mode(raw) is expected


def test_get_instrument_mode_falls_back_to_basic() -> None:
    """Unknown instrumentation modes fall back to basic."""
    assert get_instrument_mode("surprise") is InstrumentMode.BASIC


def test_configure_daemon_logging_writes_jsonl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    restore_root_logging: None,
) -> None:
    """Structured modes write JSONL log files."""
    monkeypatch.setenv(INSTRUMENT_ENV_VAR, "basic")
    log_file = tmp_path / "daemon.jsonl"

    mode = configure_daemon_logging("INFO", str(log_file))
    logging.getLogger("reachy_mini.test").info(
        "hello daemon",
        extra={
            "event": "test.event",
            "attrs": {"answer": 42},
        },
    )

    assert mode is InstrumentMode.BASIC
    lines = log_file.read_text().splitlines()
    assert len(lines) == 2
    payload = json.loads(lines[-1])
    assert payload["level"] == "INFO"
    assert payload["logger"] == "reachy_mini.test"
    assert payload["event"] == "test.event"
    assert payload["message"] == "hello daemon"
    assert payload["attrs"] == {"answer": 42}


def test_configure_daemon_logging_off_raises_level_to_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    restore_root_logging: None,
) -> None:
    """Off mode suppresses info logs and keeps plain warning output."""
    monkeypatch.setenv(INSTRUMENT_ENV_VAR, "off")
    log_file = tmp_path / "daemon.log"

    mode = configure_daemon_logging("INFO", str(log_file))
    logging.getLogger("reachy_mini.test").info("hidden")
    logging.getLogger("reachy_mini.test").warning("visible")

    assert mode is InstrumentMode.OFF
    assert log_file.read_text().splitlines() == [
        "reachy_mini.test - WARNING - visible",
    ]


def test_trace_mode_writes_timing_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    restore_root_logging: None,
) -> None:
    """Trace mode writes span-style timing events."""
    monkeypatch.setenv(INSTRUMENT_ENV_VAR, "trace")
    log_file = tmp_path / "daemon.jsonl"

    configure_daemon_logging("INFO", str(log_file))
    with timing_event("daemon.test", stage="unit-test"):
        pass

    payload = json.loads(log_file.read_text().splitlines()[-1])
    assert payload["event"] == "span.end"
    assert payload["attrs"]["name"] == "daemon.test"
    assert payload["attrs"]["stage"] == "unit-test"
    assert payload["attrs"]["duration_ms"] >= 0
    assert payload["attrs"]["outcome"] == "ok"


def test_mockup_backend_setup_writes_trace_span(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    restore_root_logging: None,
) -> None:
    """Mockup backend setup emits a startup trace span."""
    monkeypatch.setenv(INSTRUMENT_ENV_VAR, "trace")
    log_file = tmp_path / "daemon.jsonl"

    configure_daemon_logging("INFO", str(log_file))
    daemon = Daemon(no_media=True)
    backend = daemon._setup_backend(
        wireless_version=False,
        sim=False,
        mockup_sim=True,
        serialport="auto",
        scene="empty",
        check_collision=False,
        kinematics_engine="AnalyticalKinematics",
        headless=True,
        use_audio=False,
    )
    backend.close()

    spans = [
        json.loads(line)
        for line in log_file.read_text().splitlines()
        if json.loads(line).get("event") == "span.end"
    ]
    assert any(
        span["attrs"]["name"] == "daemon.backend.construct"
        and span["attrs"]["backend_mode"] == "mockup"
        for span in spans
    )
