"""Daemon.rest: /api/daemon/stop rests the robot but keeps media + relay up."""

from unittest.mock import AsyncMock, Mock

import pytest

from reachy_mini.daemon.daemon import Daemon
from reachy_mini.io.protocol import DaemonState, MotorControlMode


def _running_daemon() -> Daemon:
    daemon = Daemon(no_media=True)
    daemon.backend = Mock(
        reset_to_sleep=AsyncMock(),
        stop_running_move=AsyncMock(),
        wake_up=AsyncMock(),
        get_status=Mock(return_value=Mock(error=None)),
    )
    daemon._media_server = Mock(is_running=True)
    daemon._start_central_signaling_relay = AsyncMock()  # type: ignore[method-assign]
    daemon._stop_central_signaling_relay = AsyncMock()  # type: ignore[method-assign]
    daemon._status.state = DaemonState.RUNNING
    return daemon


@pytest.mark.asyncio
@pytest.mark.parametrize("goto_sleep", [True, False])
async def test_rest_keeps_backend_media_and_relay(goto_sleep: bool) -> None:
    """The robot rests, nothing is torn down."""
    daemon = _running_daemon()
    backend = daemon.backend
    assert backend is not None

    assert await daemon.rest(goto_sleep=goto_sleep) == DaemonState.STOPPED

    if goto_sleep:
        backend.reset_to_sleep.assert_awaited_once()
    else:
        backend.set_motor_control_mode.assert_called_once_with(
            MotorControlMode.Disabled
        )
    assert daemon.backend is backend
    backend.close.assert_not_called()
    daemon._media_server.stop.assert_not_called()
    daemon._stop_central_signaling_relay.assert_not_awaited()


@pytest.mark.asyncio
async def test_start_after_rest_only_wakes() -> None:
    """Start on a resting daemon wakes the robot without rebuilding the backend."""
    daemon = _running_daemon()
    await daemon.rest(goto_sleep=True)
    daemon._setup_backend = Mock()  # type: ignore[method-assign]

    assert await daemon.start(wake_up_on_start=True) == DaemonState.RUNNING

    daemon._setup_backend.assert_not_called()
    assert daemon.backend is not None
    daemon.backend.wake_up.assert_awaited_once()


@pytest.mark.asyncio
async def test_stop_after_rest_tears_down_without_moving() -> None:
    """Process shutdown still tears a resting daemon down, without a sleep move."""
    daemon = _running_daemon()
    await daemon.rest(goto_sleep=False)
    backend = daemon.backend
    assert backend is not None

    assert await daemon.stop(goto_sleep_on_stop=True) == DaemonState.STOPPED

    backend.reset_to_sleep.assert_not_awaited()
    backend.close.assert_called_once()
    daemon._media_server.stop.assert_called_once()
    daemon._stop_central_signaling_relay.assert_awaited_once()
    assert daemon.backend is None


@pytest.mark.asyncio
async def test_login_starts_the_relay_while_resting() -> None:
    """A sign-in while resting brings the robot onto central."""
    daemon = _running_daemon()
    await daemon.rest(goto_sleep=False)

    await daemon.start_central_relay_if_running()

    daemon._start_central_signaling_relay.assert_awaited_once()
