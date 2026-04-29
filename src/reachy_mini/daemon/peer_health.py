"""Health derivation for the central signaling ``meta`` payload.

This module owns *one* question: given a snapshot of the daemon's state,
what value of ``meta.health`` should the central listing publish?

It deliberately knows nothing about:

- the central signaling relay (we are called *by* it, never the other way),
- WebRTC, HTTP, BLE, or any transport,
- the difference between a desktop tray and a robot Pi (the caller has
  that knowledge and passes it explicitly).

The result is a small dataclass the caller copies into the ``meta`` dict.
Keeping the logic in a pure function makes the truth table easy to test
and lets the central relay treat health as opaque data.

See ``docs/SIGNALING.md`` for the published contract values and the
client-side policy attached to each.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from reachy_mini.io.protocol import DaemonState, DaemonStatus

HealthValue = Literal["ok", "degraded", "error"]
"""Tri-state matching the values mobile/desktop clients understand.

- ``ok``       fully operational, auto-connect allowed
- ``degraded`` daemon up, optional subsystem missing (camera, audio, …)
- ``error``    do not auto-connect, surface to user
"""


@dataclass(frozen=True)
class PeerHealth:
    """The health verdict for a single status snapshot.

    ``health`` is the user-facing tri-state. ``error_code`` is a short
    stable taxonomy attached when ``health != "ok"`` so clients can
    branch (e.g. show a "no hardware" prompt vs. a "media subsystem
    failed" warning) without parsing free-text messages. It is
    intentionally a small set; do not invent codes per call site.
    """

    health: HealthValue
    error_code: Optional[str]


# Stable error code taxonomy. Add values here and update SIGNALING.md
# (the "stable taxonomy" sentence) whenever a new code ships.
ERR_NO_BACKEND = "no_backend"  # tray/robot with no backend instance attached
ERR_BACKEND_NOT_READY = (
    "backend_not_ready"  # backend object exists but ready Event never set
)
ERR_MOTOR_COMM = "motor_comm"  # serial/USB error reported by the backend
ERR_DAEMON_FATAL = "daemon_fatal"  # daemon state == ERROR with no more specific cause
ERR_MEDIA = "media"  # camera/audio missing - degraded only, not error


def compute(status: DaemonStatus) -> PeerHealth:
    """Derive the published ``health`` from a daemon status snapshot.

    The caller is expected to pass the ``DaemonStatus`` it just produced
    via ``daemon.status()``. Everything we need lives there:

    - ``state`` for fatal transitions (ERROR, STOPPING, …)
    - ``backend_status.ready`` for hardware readiness
    - ``backend_status.error`` for explicit motor / serial errors
    - ``no_media`` to avoid flagging "media missing" when it was disabled
      on purpose (``--no-media``)

    No I/O, no mutation, no logging. This function must remain a pure
    transformation of its argument so the truth-table tests in
    ``tests/test_peer_health.py`` stay reliable.
    """
    # Hard fatal states first - they short-circuit every other check.
    if status.state in (DaemonState.STOPPING, DaemonState.STOPPED):
        # We're tearing down on purpose. Don't tell listeners we're "ok"
        # for the few seconds it takes to actually close the relay; the
        # client should immediately stop showing us as auto-connectable.
        return PeerHealth(health="error", error_code=ERR_DAEMON_FATAL)

    if status.state == DaemonState.ERROR:
        # Surface the most informative code we can find.
        backend = status.backend_status
        if backend is not None and getattr(backend, "error", None):
            return PeerHealth(health="error", error_code=ERR_MOTOR_COMM)
        if backend is None:
            return PeerHealth(health="error", error_code=ERR_NO_BACKEND)
        return PeerHealth(health="error", error_code=ERR_DAEMON_FATAL)

    # Daemon is in a non-fatal state. Now look at the backend.
    backend = status.backend_status
    if backend is None:
        # Most common reason: a desktop tray daemon was started but no
        # USB hardware was detected, so we never instantiated a
        # RobotBackend. From a remote-listing point of view this is an
        # error: nothing on this peer can serve a session.
        return PeerHealth(health="error", error_code=ERR_NO_BACKEND)

    if not getattr(backend, "ready", False):
        # Backend object exists but its readiness Event is not set.
        # During NOT_INITIALIZED / STARTING this can happen briefly;
        # we still report ``error`` so the client doesn't auto-connect
        # mid-bootstrap. The next status tick will correct it once the
        # Event flips.
        return PeerHealth(health="error", error_code=ERR_BACKEND_NOT_READY)

    if getattr(backend, "error", None):
        # Backend is "ready" but has logged a recoverable error
        # (transient motor read failure, etc.). Treat as degraded for
        # now - the user can still control the robot, we just want a
        # warning surface.
        return PeerHealth(health="degraded", error_code=ERR_MOTOR_COMM)

    # Media subsystem was requested but failed to come up.
    if not status.no_media and not status.camera_specs_name:
        return PeerHealth(health="degraded", error_code=ERR_MEDIA)

    return PeerHealth(health="ok", error_code=None)
