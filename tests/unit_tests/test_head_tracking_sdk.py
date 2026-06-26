"""Tests for SDK head-tracking controls."""

import logging
from collections.abc import Mapping

import requests

from reachy_mini import ReachyMini


class HeadTrackingResponse:
    """Small response object for head-tracking REST calls."""

    def __init__(self, payload: Mapping[str, object]) -> None:
        """Initialize the fake response payload."""
        self._payload = payload

    def raise_for_status(self) -> None:
        """Mirror requests.Response.raise_for_status."""

    def json(self) -> Mapping[str, object]:
        """Return the fake JSON payload."""
        return self._payload


def _make_robot() -> ReachyMini:
    """Create a ReachyMini instance without opening daemon connections."""
    robot = ReachyMini.__new__(ReachyMini)
    robot._daemon_http_url = "http://reachy-mini.local:8000"
    robot.logger = logging.getLogger("test_head_tracking_sdk")
    return robot


def test_start_head_tracking_returns_enabled_state(
    monkeypatch,
) -> None:
    """The SDK should expose whether the daemon accepted tracking."""
    calls: list[tuple[str, Mapping[str, float] | None, float]] = []

    def post(
        url: str,
        json: Mapping[str, float] | None = None,
        timeout: float = 0.0,
    ) -> HeadTrackingResponse:
        calls.append((url, json, timeout))
        return HeadTrackingResponse({"status": "ok", "enabled": True})

    monkeypatch.setattr(requests, "post", post)

    assert _make_robot().start_head_tracking(weight=0.6) is True
    assert calls == [
        (
            "http://reachy-mini.local:8000/api/media/tracking/enable",
            {"weight": 0.6},
            5.0,
        )
    ]


def test_start_head_tracking_returns_false_when_unavailable(monkeypatch) -> None:
    """The SDK should not report success when the daemon rejects tracking."""

    def post(
        url: str,
        json: Mapping[str, float] | None = None,
        timeout: float = 0.0,
    ) -> HeadTrackingResponse:
        return HeadTrackingResponse({"status": "unavailable", "enabled": False})

    monkeypatch.setattr(requests, "post", post)

    assert _make_robot().start_head_tracking(weight=0.6) is False
