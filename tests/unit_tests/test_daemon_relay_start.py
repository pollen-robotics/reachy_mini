"""Daemon.start_central_relay_if_running: the login routes' relay hook."""

from unittest.mock import AsyncMock, Mock

import pytest

from reachy_mini.daemon.daemon import Daemon
from reachy_mini.io.protocol import DaemonState


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("state", "media_released", "expected_calls"),
    [
        (DaemonState.STOPPED, False, 0),
        (DaemonState.RUNNING, True, 0),
        (DaemonState.RUNNING, False, 1),
    ],
)
async def test_relay_started_only_when_daemon_runs(
    state: DaemonState, media_released: bool, expected_calls: int
) -> None:
    """The relay starter runs only when the daemon is RUNNING with media acquired."""
    daemon = Daemon(no_media=True)
    daemon._start_central_signaling_relay = AsyncMock()  # type: ignore[method-assign]
    daemon._status.state = state
    daemon._media_released = media_released

    await daemon.start_central_relay_if_running()

    assert daemon._start_central_signaling_relay.await_count == expected_calls


@pytest.mark.asyncio
async def test_acquire_media_failure_keeps_media_released() -> None:
    """A pipeline that does not start leaves media released and the relay down."""
    daemon = Daemon(no_media=True)
    daemon._media_server = Mock(start=Mock(return_value=False))
    daemon._start_central_signaling_relay = AsyncMock()  # type: ignore[method-assign]
    daemon._media_released = True

    with pytest.raises(RuntimeError):
        await daemon.acquire_media()

    assert daemon.media_released
    daemon._start_central_signaling_relay.assert_not_awaited()
