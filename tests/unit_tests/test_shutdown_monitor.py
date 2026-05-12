"""Unit tests for the GPIO23 shutdown-button monitor.

Regression coverage for issue #1109: motor-coil EMI on GPIO23 must not
fire :func:`shutdown_now` (HIGH-burst-during-release suppression test)
while a deliberate sustained release still must (long-hold test). Brief
release+press gestures must also not fire shutdown_now (gesture-cancel
test).

Drives ``gpiozero`` via its ``MockFactory`` so the tests run on any host
— no real GPIO required. Each test reloads the module under the mock
factory and patches :data:`HOLD_TIME` to 0.3 s so the suite stays fast;
the production constant ``HOLD_TIME = 0.5`` is verified separately by
``test_module_constants_match_design``.

Pin-state setup pattern: ``Button(pull_up=False)`` resets the
MockFactory pin to LOW at construction. After the module reload, drive
HIGH to establish the "latch IN / running normally" state, then drive
LOW to simulate the user popping the latch OUT.
"""

from __future__ import annotations

import importlib
import time
from unittest.mock import patch

import pytest

# gpiozero is a Linux-only optional dep in pyproject.toml; on macOS / Windows
# the wheel may be installable but pin operations require a backend. Skip
# this module if gpiozero is unimportable rather than failing the suite.
gpiozero = pytest.importorskip("gpiozero")
from gpiozero import Device  # noqa: E402
from gpiozero.pins.mock import MockFactory  # noqa: E402


@pytest.fixture(autouse=True, scope="session")
def _bind_mock_pin_factory_for_session():
    """Bind a session-scoped ``MockFactory`` and restore the previous value.

    The ``shutdown_monitor`` module constructs ``Button(23, pull_up=False)``
    at import time; without a pin_factory pre-set, gpiozero attempts to
    auto-detect one and raises ``BadPinFactory`` on hosts without Linux
    GPIO (CI runners, dev macOS / Windows). Setting the factory at session
    start ensures the per-test ``shutdown_module`` fixture always finds a
    valid factory when it reloads the module.

    Capturing+restoring (rather than mutating ``Device.pin_factory`` at
    module-import time) means importing this test file does NOT leak a
    MockFactory into other test modules that may rely on the default
    pin-factory auto-detect behaviour.
    """
    previous = Device.pin_factory
    Device.pin_factory = MockFactory()
    try:
        yield
    finally:
        Device.pin_factory = previous


@pytest.fixture
def shutdown_module(monkeypatch):
    """Reload the module under a fresh ``MockFactory`` with a fast hold_time.

    Why reload: the module wires ``shutdown_button = Button(23, pull_up=False)``
    at import time, capturing whatever ``Device.pin_factory`` is then. Tests
    must swap to ``MockFactory`` first.

    Why monkeypatch ``HOLD_TIME``: 0.3 s is faster than the production
    value and is short enough that the "brief release < HOLD_TIME"
    gestures the suite simulates run inside CI's wall-clock budget.
    ``_schedule_shutdown`` reads ``HOLD_TIME`` at fire-time, so patching
    after import works.
    """
    previous = Device.pin_factory
    Device.pin_factory = MockFactory()

    from reachy_mini.daemon.app.services.gpio_shutdown import shutdown_monitor

    importlib.reload(shutdown_monitor)
    monkeypatch.setattr(shutdown_monitor, "HOLD_TIME", 0.3)

    try:
        yield shutdown_monitor
    finally:
        # Cancel any in-flight Timer so it can't fire after the test ends.
        shutdown_monitor._cancel_shutdown()
        Device.pin_factory.reset()
        Device.pin_factory = previous


def _settle() -> None:
    """Yield briefly so gpiozero's edge-dispatch thread runs handlers."""
    time.sleep(0.02)


def _wait_until(predicate, timeout: float = 2.0, poll: float = 0.01) -> bool:
    """Poll ``predicate`` until True or ``timeout`` elapses.

    Returns the final ``predicate()`` value (True if it became true within
    the window, False if it never did). Used to make Timer-fire assertions
    robust against scheduling delay on slow / loaded CI runners — see
    Copilot review feedback on PR #1110.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll)
    return predicate()


def test_module_constants_match_design():
    """Production HOLD_TIME = 0.5 s.

    Bounded above by the Wireless's ~2 s latch-OUT-to-power-rail-cut
    deadline (per review on PR #1110: ``shutdown_now()`` must dispatch
    inside that window); bounded below by gesture-recognition UX (long
    enough to distinguish a deliberate latch pull from an incidental
    brush). EMI bursts are filtered by the cancel-on-press edge,
    independent of HOLD_TIME.
    """
    from reachy_mini.daemon.app.services.gpio_shutdown.shutdown_monitor import (
        HOLD_TIME,
    )

    assert HOLD_TIME == 0.5


def test_brief_release_then_press_does_not_fire_shutdown(shutdown_module):
    """A brief release ended by a re-press must NOT fire ``shutdown_now``.

    User gesture: pop the latch out and push it back in within < HOLD_TIME.
    The pending Timer must be cancelled by ``when_pressed`` before it fires.
    """
    fired: list[bool] = []
    monkey_target = (
        "reachy_mini.daemon.app.services.gpio_shutdown.shutdown_monitor.shutdown_now"
    )
    pin = Device.pin_factory.pin(23)
    pin.drive_high()  # latch IN (cancel-pending no-op since no Timer scheduled).
    _settle()

    with patch(monkey_target, side_effect=lambda: fired.append(True)):
        pin.drive_low()   # release at t=0; Timer would fire at t=0.3.
        time.sleep(0.1)   # 0.1 < 0.3.
        pin.drive_high()  # cancels the pending Timer.
        time.sleep(0.4)   # well past the original hold deadline.

    assert fired == [], (
        f"brief release+press fired shutdown_now: fired={fired}"
    )


def test_sustained_release_fires_shutdown_once(shutdown_module):
    """A continuous release > HOLD_TIME MUST fire ``shutdown_now`` exactly once."""
    fired: list[bool] = []
    monkey_target = (
        "reachy_mini.daemon.app.services.gpio_shutdown.shutdown_monitor.shutdown_now"
    )
    pin = Device.pin_factory.pin(23)
    pin.drive_high()  # latch IN.
    _settle()

    with patch(monkey_target, side_effect=lambda: fired.append(True)):
        pin.drive_low()   # release at t=0; Timer scheduled to fire at t=HOLD_TIME.
        # Poll for fire instead of fixed sleep — robust to slow / loaded CI runners.
        assert _wait_until(lambda: fired == [True], timeout=2.0), (
            f"sustained release did not fire shutdown_now within 2s: fired={fired}"
        )


def test_emi_burst_during_release_cancels_pending_timer(shutdown_module):
    """A transient HIGH burst during a release must cancel the pending Timer.

    Simulates the motor-coil EMI failure mode of #1109: a brief HIGH spike
    superimposed on a continuously LOW pin must not let the original Timer
    fire on its original schedule. A new Timer schedules from the next
    release; it fires after a fresh HOLD_TIME, not from the first release.
    The first assertion verifies cancellation; the second verifies that a
    real, sustained release after the burst still fires.
    """
    fired: list[bool] = []
    monkey_target = (
        "reachy_mini.daemon.app.services.gpio_shutdown.shutdown_monitor.shutdown_now"
    )
    pin = Device.pin_factory.pin(23)
    pin.drive_high()  # latch IN.
    _settle()

    with patch(monkey_target, side_effect=lambda: fired.append(True)):
        pin.drive_low()   # Release at t=0; Timer #1 would fire at t=0.3.
        time.sleep(0.1)   # t=0.1.
        pin.drive_high()  # EMI burst cancels Timer #1.
        pin.drive_low()   # Release continues; Timer #2 schedules → fires at t≈0.4.

        # At t≈0.35 — past Timer #1's deadline (t=0.3), before Timer #2 (t≈0.4).
        time.sleep(0.25)
        assert fired == [], (
            f"EMI burst did not cancel Timer #1 before its original deadline: "
            f"fired={fired}"
        )

        # Timer #2's deadline is t≈0.4; poll for fire instead of fixed sleep
        # so the test is robust to scheduling delay on slow / loaded CI runners.
        assert _wait_until(lambda: fired == [True], timeout=2.0), (
            f"Timer #2 did not fire after the EMI burst within 2s: fired={fired}"
        )


def test_shutdown_now_invokes_shutdown_command():
    """``shutdown_now`` must call ``sudo shutdown -h now`` exactly once."""
    from reachy_mini.daemon.app.services.gpio_shutdown.shutdown_monitor import (
        shutdown_now,
    )

    target = "reachy_mini.daemon.app.services.gpio_shutdown.shutdown_monitor.call"
    with patch(target) as mock_call:
        shutdown_now()
    mock_call.assert_called_once_with(["sudo", "shutdown", "-h", "now"])


def test_scheduled_timer_is_daemon_thread(shutdown_module):
    """Pending Timer must be ``daemon=True`` so process exit isn't blocked.

    Critical safety property: if the Python process exits while a Timer is
    pending, the Timer thread must not keep the process alive (which would
    hang the SDK shutdown sequence). The fix is the explicit
    ``timer.daemon = True`` in :func:`shutdown_monitor._schedule_shutdown`;
    this regression test pins that behaviour.
    """
    pin = Device.pin_factory.pin(23)
    pin.drive_high()  # latch IN.
    _settle()
    pin.drive_low()  # release schedules a pending Timer.
    _settle()

    assert shutdown_module._pending_timer is not None, (
        "release must schedule a pending Timer"
    )
    assert shutdown_module._pending_timer.daemon is True, (
        "scheduled Timer must be daemon=True to prevent process hang on exit"
    )
