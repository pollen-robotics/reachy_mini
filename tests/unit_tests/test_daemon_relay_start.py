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


def _daemon_with_media(running: bool, starts: bool = True) -> Daemon:
    daemon = Daemon(no_media=True)
    daemon._media_server = Mock(is_running=running, start=Mock(return_value=starts))
    daemon._start_central_signaling_relay = AsyncMock()  # type: ignore[method-assign]
    daemon._stop_central_signaling_relay = AsyncMock()  # type: ignore[method-assign]
    return daemon


@pytest.mark.asyncio
async def test_acquire_media_failure_keeps_media_released() -> None:
    """A pipeline that does not start leaves media released and the relay down."""
    daemon = _daemon_with_media(running=False, starts=False)
    daemon._media_released = True

    with pytest.raises(RuntimeError):
        await daemon.acquire_media()

    assert daemon.media_released
    daemon._start_central_signaling_relay.assert_not_awaited()


@pytest.mark.asyncio
async def test_acquire_media_restarts_a_pipeline_that_stopped() -> None:
    """Without a release, acquire restarts a pipeline that stopped on its own."""
    daemon = _daemon_with_media(running=False)

    await daemon.acquire_media()

    daemon._media_server.start.assert_called_once()
    daemon._start_central_signaling_relay.assert_awaited_once()
    assert not daemon.media_released


@pytest.mark.asyncio
async def test_acquire_media_is_a_noop_while_running() -> None:
    """A running pipeline is left alone."""
    daemon = _daemon_with_media(running=True)

    await daemon.acquire_media()

    daemon._media_server.start.assert_not_called()


@pytest.mark.asyncio
async def test_failed_acquire_stops_the_relay_of_a_stopped_pipeline() -> None:
    """The relay must not keep looping on the dead pipeline's signalling server."""
    daemon = _daemon_with_media(running=False, starts=False)

    with pytest.raises(RuntimeError):
        await daemon.acquire_media()

    daemon._stop_central_signaling_relay.assert_awaited_once()


@pytest.mark.asyncio
async def test_no_relay_on_login_while_the_pipeline_is_stopped() -> None:
    """No relay on a login while the pipeline (and its signalling server) is down."""
    daemon = _daemon_with_media(running=False)
    daemon._status.state = DaemonState.RUNNING

    await daemon.start_central_relay_if_running()

    daemon._start_central_signaling_relay.assert_not_awaited()


@pytest.mark.parametrize(
    ("released", "running", "available"),
    [(False, True, True), (False, False, False), (True, False, False)],
)
def test_media_available_follows_the_pipeline(
    released: bool, running: bool, available: bool
) -> None:
    """A pipeline that stopped on a failure is not available, release or not."""
    daemon = _daemon_with_media(running=running)
    daemon._media_released = released

    assert daemon.media_available is available
