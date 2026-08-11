"""Tests for SDK head-tracking controls."""

import logging

from reachy_mini import ReachyMini
from reachy_mini.io.protocol import SetHeadTrackingCmd


class _FakeClient:
    """Capture commands the SDK sends to the daemon."""

    def __init__(self) -> None:
        """Initialize the recorded-command list."""
        self.commands: list[object] = []

    def send_command(self, cmd: object) -> None:
        """Record a command instead of sending it."""
        self.commands.append(cmd)


def _make_robot() -> ReachyMini:
    """Create a ReachyMini instance without opening daemon connections."""
    robot = ReachyMini.__new__(ReachyMini)
    robot.client = _FakeClient()
    robot.logger = logging.getLogger("test_head_tracking_sdk")
    return robot


def test_start_head_tracking_sends_enable_command() -> None:
    """Starting tracking sends a weighted enable command over the SDK client."""
    robot = _make_robot()
    robot.start_head_tracking(weight=0.6)
    assert robot.client.commands == [SetHeadTrackingCmd(enabled=True, weight=0.6)]


def test_stop_head_tracking_sends_disable_command() -> None:
    """Stopping tracking sends a disable command over the SDK client."""
    robot = _make_robot()
    robot.stop_head_tracking()
    assert robot.client.commands == [SetHeadTrackingCmd(enabled=False)]
