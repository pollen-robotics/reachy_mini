"""Unit tests for the demand-driven Direction-of-Arrival cache.

The DoA poller must be lazy (no USB traffic while nobody consumes the
cache), self-stopping (idle timeout), and it is the only reader of the
device: consumers get the cache and never block. Teardown must never
dispose the USB device while a read is still in flight, which is why
``FakeDoA`` can simulate the blocking tail of a real ``ReSpeaker.read()``
(``read_delay``).

The ReSpeaker itself is faked; the real USB layer is exercised in
``test_audio_control_utils.py``.
"""

from __future__ import annotations

import time

from reachy_mini.daemon.backend.mockup_sim.backend import MockupSimBackend
from reachy_mini.io.protocol import DoaSnapshot


class FakeDoA:
    """AudioDoA stand-in: counts reads, can fail or block, no USB."""

    def __init__(
        self,
        reading: tuple[float, bool] | None = (1.25, True),
        available: bool = True,
        raise_on_read: bool = False,
        read_delay: float = 0.0,
    ) -> None:
        """Configure the fixed reading, failure and blocking behaviour."""
        self.reading = reading
        self.available = available
        self.raise_on_read = raise_on_read
        self.read_delay = read_delay
        self.calls = 0
        self.reads_in_flight = 0
        self.closed_mid_read = False

    def get_DoA(self) -> tuple[float, bool] | None:  # noqa: N802 - mirrors AudioDoA
        """Return the configured reading, blocking `read_delay` seconds."""
        self.calls += 1
        self.reads_in_flight += 1
        try:
            if self.read_delay:
                time.sleep(self.read_delay)
            if self.raise_on_read:
                raise RuntimeError("usb read failed")
            return self.reading
        finally:
            self.reads_in_flight -= 1

    def close(self) -> None:
        """Mimic AudioDoA.close, recording a close during an active read.

        On the real device that interleaving is a libusb use-after-free
        (`dispose_resources` on a device mid-`ctrl_transfer`), so any test
        that ends with ``closed_mid_read`` True has caught a real bug.
        """
        if self.reads_in_flight:
            self.closed_mid_read = True
        self.available = False


def _make_backend(doa: FakeDoA | None = None) -> MockupSimBackend:
    """Backend without audio, with an optionally injected fake DoA."""
    backend = MockupSimBackend(use_audio=False)
    backend.doa = doa  # type: ignore[assignment]
    return backend


def _wait_for(predicate, timeout: float = 2.0) -> bool:
    """Poll `predicate` until true or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_no_device_no_thread_no_reading() -> None:
    """Without a DoA helper nothing is spawned and the reading is None."""
    backend = _make_backend(doa=None)
    assert backend.read_doa() is None
    assert backend._doa_thread is None


def test_unavailable_device_never_polls() -> None:
    """A helper whose USB probe failed must never be read nor polled."""
    fake = FakeDoA(available=False)
    backend = _make_backend(fake)
    assert backend.read_doa() is None
    assert backend._doa_thread is None
    assert fake.calls == 0


def test_read_doa_starts_poller_and_serves_cache() -> None:
    """First read is a cache miss but warms the poller; then it serves."""
    fake = FakeDoA(reading=(0.5, False))
    backend = _make_backend(fake)
    backend.DOA_POLL_INTERVAL_S = 0.01  # type: ignore[misc]

    # Cold cache: no reading yet, but demand spawns the poller.
    assert backend.read_doa() is None
    assert backend._doa_thread is not None and backend._doa_thread.is_alive()

    assert _wait_for(lambda: backend.read_doa() is not None)
    assert backend.read_doa() == DoaSnapshot(angle=0.5, speech_detected=False)

    backend._doa_stop.set()


def test_poller_stops_when_demand_ceases_and_clears_cache() -> None:
    """The poller exits after DOA_IDLE_STOP_S and never serves a pre-idle reading."""
    fake = FakeDoA()
    backend = _make_backend(fake)
    backend.DOA_POLL_INTERVAL_S = 0.01  # type: ignore[misc]
    backend.DOA_IDLE_STOP_S = 0.05  # type: ignore[misc]

    backend.read_doa()
    thread = backend._doa_thread
    assert thread is not None and thread.is_alive()

    # No further demand: the thread must die by itself...
    assert _wait_for(lambda: not thread.is_alive())
    # ...leaving no stale reading behind for a later consumer.
    assert backend._last_doa is None


def test_poller_restarts_on_new_demand() -> None:
    """Demand after an idle stop spawns a fresh poller thread."""
    fake = FakeDoA()
    backend = _make_backend(fake)
    backend.DOA_POLL_INTERVAL_S = 0.01  # type: ignore[misc]
    backend.DOA_IDLE_STOP_S = 0.05  # type: ignore[misc]

    backend.read_doa()
    first = backend._doa_thread
    assert first is not None
    assert _wait_for(lambda: not first.is_alive())

    backend.read_doa()
    second = backend._doa_thread
    assert second is not None and second is not first and second.is_alive()

    backend._doa_stop.set()


def test_poller_swallows_usb_errors() -> None:
    """A failing USB read reads as 'no DoA' and never kills the poller."""
    fake = FakeDoA(raise_on_read=True)
    backend = _make_backend(fake)
    backend.DOA_POLL_INTERVAL_S = 0.01  # type: ignore[misc]

    backend.read_doa()
    assert _wait_for(lambda: fake.calls >= 3)
    assert backend.read_doa() is None
    assert backend._doa_thread is not None and backend._doa_thread.is_alive()

    backend._doa_stop.set()


def test_close_joins_poller_and_releases_device() -> None:
    """Backend.close() joins the poller thread and drops the USB handle."""
    fake = FakeDoA(reading=(2.0, True))
    backend = _make_backend(fake)
    backend.DOA_POLL_INTERVAL_S = 0.01  # type: ignore[misc]

    # Spin the poller up through a normal demand path.
    backend.read_doa()
    assert _wait_for(lambda: backend.read_doa() is not None)
    thread = backend._doa_thread
    assert thread is not None and thread.is_alive()

    backend.close()

    assert not thread.is_alive()
    assert backend._doa_thread is None
    # The handle was closed and dropped, so post-close reads are inert.
    assert fake.available is False
    assert backend.doa is None
    assert backend.read_doa() is None


def test_close_waits_for_inflight_read() -> None:
    """close() must not dispose the device while a read is in flight.

    The blocking tail is longer than close()'s 1 s join timeout, so only
    taking `_doa_usb_lock` before disposing makes this pass: past the join
    timeout, the fake would otherwise be closed mid-read (a libusb
    use-after-free on real hardware).
    """
    fake = FakeDoA(reading=(2.0, True), read_delay=1.5)
    backend = _make_backend(fake)
    backend.DOA_POLL_INTERVAL_S = 0.01  # type: ignore[misc]

    backend.read_doa()
    assert _wait_for(lambda: fake.reads_in_flight > 0)

    backend.close()

    assert fake.closed_mid_read is False
    assert fake.available is False
    assert backend.doa is None


def test_build_state_dict_carries_doa() -> None:
    """The pushed frame always has a `doa` key (None when unavailable)."""
    backend = _make_backend(doa=None)
    state = backend.build_state_dict()
    assert "doa" in state
    assert state["doa"] is None
