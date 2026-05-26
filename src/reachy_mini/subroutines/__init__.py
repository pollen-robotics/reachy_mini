"""Subroutines: ship-and-call JS skills running inside the daemon.

A subroutine is a small JS module a client ships over the data channel
that runs inside an embedded QuickJS context on the robot. The
subroutine exports an ``api`` object whose methods the original client
can invoke remotely, may run idle loops (``setInterval`` /
``setTimeout``) on its own, and may push events back to the owner via
the ``emit(name, payload)`` binding.

End-goal: a Space's full app logic can live in a subroutine on the
robot, so the Space's browser tab becomes a thin UI shell — close it,
walk away, the behaviour keeps running.

See :class:`SubroutineRuntime` for the runtime; protocol commands and
messages are in :mod:`reachy_mini.io.protocol`.
"""

from .runtime import (
    QUICKJS_AVAILABLE,
    SubroutineHost,
    SubroutineRuntime,
    SubroutineRuntimeError,
    SubroutineSender,
)

__all__ = [
    "QUICKJS_AVAILABLE",
    "SubroutineHost",
    "SubroutineRuntime",
    "SubroutineRuntimeError",
    "SubroutineSender",
]
