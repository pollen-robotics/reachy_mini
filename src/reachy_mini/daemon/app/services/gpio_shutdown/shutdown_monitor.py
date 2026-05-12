"""Monitor GPIO23 for the body shutdown-button signal.

Hardware: GPIO23 is wired to a soft-power latch on the robot body. The
latch sits IN during normal operation (pin HIGH, ``is_pressed=True``);
the user signals "shutdown" by popping the latch OUT (pin LOW,
``is_pressed=False``) and leaving it OUT for at least :data:`HOLD_TIME`
seconds. Briefly popping the latch out and back in is NOT a shutdown
gesture.

``pull_up=False`` keeps gpiozero from overriding the kernel-side pull
configured in the device tree.

Issue #1109: motor-coil EMI couples brief LOW spikes onto GPIO23 during
sustained ``set_target`` activity. The previous busy-wait debounce in
PR #505 polls ``is_pressed`` on a 1 ms schedule for 200 ms after each
``when_released``; sub-millisecond EMI spikes can pass between samples,
so the poll never observes the pin returning HIGH and a spurious release
becomes a real shutdown.

This module replaces the busy-wait with an edge-driven design:

* ``when_released`` (falling edge) starts a
  ``threading.Timer(HOLD_TIME, shutdown_now)``.
* ``when_pressed`` (rising edge) cancels any pending Timer.

Because gpiozero's edge handlers are interrupt-style (kernel GPIO events,
not sampled polls), even sub-millisecond HIGH spikes that flip the pin
HIGH→LOW→HIGH cancel the pending Timer. A deliberate :data:`HOLD_TIME`-
second release with no intervening HIGH spikes still fires
:func:`shutdown_now`.

Tradeoffs:

* ``Timer.cancel()`` is best-effort. If the worker thread has already
  begun executing the callback when ``cancel()`` is called, the shutdown
  proceeds. This race window is sub-millisecond and acceptable for a
  shutdown gesture.
* HIGH spikes superimposed on an already-LOW pin during a real user
  release also cancel the Timer. That matches PR #505's "bounce, ignore"
  behaviour (no regression versus current main), but heavy motor
  activity can theoretically delay or suppress an intentional shutdown
  gesture. Not observed empirically.

HOLD_TIME sizing:

EMI bursts are filtered by the cancel-on-press edge, not by HOLD_TIME --
any rising edge during the hold window cancels the pending Timer
independent of HOLD_TIME's value. HOLD_TIME is therefore a UX /
gesture-recognition parameter, not an EMI filter parameter. Its only
hard upper bound is that ``shutdown_now()`` plus the systemd shutdown
dispatch must complete before the Wireless's latch-OUT-cuts-power-rail
deadline (~2 s, per review on PR #1110); the lower bound is "long
enough to distinguish a deliberate latch pull from an incidental
brush". 0.5 s sits comfortably inside that window.
"""

from __future__ import annotations

import threading
from signal import pause
from subprocess import call

from gpiozero import Button

#: Sustained-release duration (seconds) before the shutdown handler fires.
#: UX / gesture-recognition parameter; not an EMI filter (the cancel-on-press
#: edge handles EMI orthogonally). Must complete shutdown_now() before the
#: Wireless's ~2 s latch-OUT power-rail cut; long enough to distinguish a
#: deliberate latch pull from an incidental brush.
HOLD_TIME: float = 0.5

shutdown_button = Button(23, pull_up=False)

_pending_timer: threading.Timer | None = None
_state_lock = threading.Lock()


def shutdown_now() -> None:
    """Issue ``sudo shutdown -h now``. Invoked by the hold-detection Timer."""
    print("Shutdown button held in released state, shutting down...")
    call(["sudo", "shutdown", "-h", "now"])


def _schedule_shutdown() -> None:
    """Falling-edge handler. Start (or restart) the hold-detection Timer."""
    global _pending_timer
    with _state_lock:
        if _pending_timer is not None:
            _pending_timer.cancel()
        # Lambda defers the ``shutdown_now`` lookup to fire-time so tests
        # can ``monkeypatch.setattr(module, "shutdown_now", ...)`` after
        # the Timer is scheduled.
        timer = threading.Timer(HOLD_TIME, lambda: shutdown_now())
        timer.daemon = True
        _pending_timer = timer
        timer.start()


def _cancel_shutdown() -> None:
    """Rising-edge handler. Cancel any pending hold-detection Timer."""
    global _pending_timer
    with _state_lock:
        if _pending_timer is not None:
            _pending_timer.cancel()
            _pending_timer = None


shutdown_button.when_released = _schedule_shutdown
shutdown_button.when_pressed = _cancel_shutdown


if __name__ == "__main__":
    print(f"Monitoring GPIO23 for {HOLD_TIME}s sustained release...")
    pause()
