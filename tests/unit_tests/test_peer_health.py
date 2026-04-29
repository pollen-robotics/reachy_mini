"""Truth table for ``reachy_mini.daemon.peer_health.compute``.

The relay exposes only what this function returns to listeners, so any
regression here shows up immediately as wrong badges in the mobile app
(or worse, auto-connect to a dead robot). Keep this file exhaustive.
"""

from __future__ import annotations

import pytest

from reachy_mini.daemon.peer_health import (
    ERR_BACKEND_NOT_READY,
    ERR_DAEMON_FATAL,
    ERR_MEDIA,
    ERR_MOTOR_COMM,
    ERR_NO_BACKEND,
    compute,
)
from reachy_mini.io.protocol import (
    DaemonState,
    DaemonStatus,
    MotorControlMode,
    RobotBackendStatus,
)


def _backend(ready: bool, error: str | None = None) -> RobotBackendStatus:
    """Build a minimal RobotBackendStatus for the truth table.

    The control loop fields are required by the pydantic model but
    ``compute()`` doesn't read them, so we keep them at trivial values.
    """
    return RobotBackendStatus(
        ready=ready,
        motor_control_mode=MotorControlMode.Enabled,
        last_alive=None,
        control_loop_stats={},
        error=error,
    )


def _status(
    state: DaemonState,
    backend: RobotBackendStatus | None = None,
    no_media: bool = False,
    camera_specs_name: str = "wireless",
) -> DaemonStatus:
    return DaemonStatus(
        robot_name="reachy_mini",
        install_id="x" * 32,
        state=state,
        wireless_version=False,
        desktop_app_daemon=False,
        simulation_enabled=None,
        mockup_sim_enabled=None,
        no_media=no_media,
        camera_specs_name=camera_specs_name,
        backend_status=backend,
        error=None,
    )


# ----------------------------------------------------------------------
# Fatal daemon states short-circuit everything else
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "state",
    [DaemonState.STOPPING, DaemonState.STOPPED],
)
def test_stopping_and_stopped_are_error_daemon_fatal(state):  # noqa: D103
    s = _status(state, backend=_backend(ready=True))
    h = compute(s)
    assert h.health == "error"
    assert h.error_code == ERR_DAEMON_FATAL


def test_error_state_with_motor_error_reports_motor_comm():  # noqa: D103
    s = _status(DaemonState.ERROR, backend=_backend(ready=True, error="bus 0 timeout"))
    h = compute(s)
    assert h.health == "error"
    assert h.error_code == ERR_MOTOR_COMM


def test_error_state_with_no_backend_reports_no_backend():  # noqa: D103
    s = _status(DaemonState.ERROR, backend=None)
    h = compute(s)
    assert h.health == "error"
    assert h.error_code == ERR_NO_BACKEND


def test_error_state_with_ready_backend_no_error_falls_back_to_daemon_fatal():  # noqa: D103
    s = _status(DaemonState.ERROR, backend=_backend(ready=True))
    h = compute(s)
    assert h.health == "error"
    assert h.error_code == ERR_DAEMON_FATAL


# ----------------------------------------------------------------------
# Non-fatal states - the backend decides
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "state",
    [DaemonState.NOT_INITIALIZED, DaemonState.STARTING, DaemonState.RUNNING],
)
def test_no_backend_in_non_fatal_state_is_no_backend_error(state):  # noqa: D103
    s = _status(state, backend=None)
    h = compute(s)
    assert h.health == "error"
    assert h.error_code == ERR_NO_BACKEND


def test_backend_present_but_not_ready_is_backend_not_ready():  # noqa: D103
    s = _status(DaemonState.STARTING, backend=_backend(ready=False))
    h = compute(s)
    assert h.health == "error"
    assert h.error_code == ERR_BACKEND_NOT_READY


def test_ready_backend_with_recoverable_error_is_degraded():  # noqa: D103
    # Backend is up and accepting commands, but logged a transient
    # serial error. We still let the user connect (degraded), we just
    # surface a warning so they know.
    s = _status(DaemonState.RUNNING, backend=_backend(ready=True, error="transient blip"))
    h = compute(s)
    assert h.health == "degraded"
    assert h.error_code == ERR_MOTOR_COMM


def test_ready_backend_no_error_is_ok():  # noqa: D103
    s = _status(DaemonState.RUNNING, backend=_backend(ready=True))
    h = compute(s)
    assert h.health == "ok"
    assert h.error_code is None


def test_no_media_does_not_count_as_degraded_when_explicitly_disabled():  # noqa: D103
    # ``--no-media`` is a deliberate choice, not a failure.
    s = _status(
        DaemonState.RUNNING,
        backend=_backend(ready=True),
        no_media=True,
        camera_specs_name="",
    )
    h = compute(s)
    assert h.health == "ok"
    assert h.error_code is None


def test_media_failure_is_degraded_not_error():  # noqa: D103
    # Media was requested (no_media=False) but the camera/audio init
    # never set ``camera_specs_name``. Treat as degraded so the user
    # can still drive the robot but knows audio/video is missing.
    s = _status(
        DaemonState.RUNNING,
        backend=_backend(ready=True),
        no_media=False,
        camera_specs_name="",
    )
    h = compute(s)
    assert h.health == "degraded"
    assert h.error_code == ERR_MEDIA


# ----------------------------------------------------------------------
# Frozen-dataclass guarantees - prevent accidental mutation
# ----------------------------------------------------------------------


def test_peer_health_is_immutable():  # noqa: D103
    s = _status(DaemonState.RUNNING, backend=_backend(ready=True))
    h = compute(s)
    with pytest.raises(Exception):
        h.health = "error"  # type: ignore[misc]
