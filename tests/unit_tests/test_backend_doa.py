"""Unit tests for the demand-driven Direction-of-Arrival cache.

The DoA poller must be lazy (no USB traffic while nobody consumes the
cache), self-stopping (idle timeout), and every USB read - poller or
on-demand REST - must go through the shared ``_doa_usb_lock`` path so a
multi-step ``ReSpeaker.read()`` conversation is never interleaved.

The ReSpeaker itself is faked; the real USB layer is exercised in
``test_audio_control_utils.py``.
"""

from __future__ import annotations

import time

from reachy_mini.daemon.backend.mockup_sim.backend import MockupSimBackend
from reachy_mini.io.protocol import DoaSnapshot


class FakeDoA:
    """AudioDoA stand-in: counts reads, can fail, no USB."""

    def __init__(
        self,
        reading: tuple[float, bool] | None = (1.25, True),
        available: bool = True,
        raise_on_read: bool = False,
    ) -> None:
        """Configure the fixed reading and failure behaviour."""
        self.reading = reading
        self.available = available
        self.raise_on_read = raise_on_read
        self.calls = 0

    def get_DoA(self) -> tuple[float, bool] | None:  # noqa: N802 - mirrors AudioDoA
        """Return the configured reading, counting every call."""
        self.calls += 1
        if self.raise_on_read:
            raise RuntimeError("usb read failed")
        return self.reading

    def close(self) -> None:
        """Mimic AudioDoA.close by dropping availability."""
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


def test_no_device_no_thread_no_snapshot() -> None:
    """Without a DoA helper nothing is spawned and the snapshot is None."""
    backend = _make_backend(doa=None)
    assert backend._doa_snapshot() is None
    assert backend._doa_thread is None


def test_unavailable_device_never_polls() -> None:
    """A helper whose USB probe failed must never be read nor polled."""
    fake = FakeDoA(available=False)
    backend = _make_backend(fake)
    assert backend._doa_snapshot() is None
    assert backend.read_doa() is None
    assert backend._doa_thread is None
    assert fake.calls == 0


def test_snapshot_starts_poller_and_serves_cache() -> None:
    """First snapshot is a cache miss but warms the poller; then it serves."""
    fake = FakeDoA(reading=(0.5, False))
    backend = _make_backend(fake)
    backend.DOA_POLL_INTERVAL_S = 0.01  # type: ignore[misc]

    # Cold cache: no reading yet, but demand spawns the poller.
    assert backend._doa_snapshot() is None
    assert backend._doa_thread is not None and backend._doa_thread.is_alive()

    assert _wait_for(lambda: backend._doa_snapshot() is not None)
    snapshot = backend._doa_snapshot()
    assert snapshot == DoaSnapshot(angle=0.5, speech_detected=False)

    backend._doa_stop.set()


def test_poller_stops_when_demand_ceases() -> None:
    """The poller exits on its own after DOA_IDLE_STOP_S without consumers."""
    fake = FakeDoA()
    backend = _make_backend(fake)
    backend.DOA_POLL_INTERVAL_S = 0.01  # type: ignore[misc]
    backend.DOA_IDLE_STOP_S = 0.05  # type: ignore[misc]

    backend._doa_snapshot()
    thread = backend._doa_thread
    assert thread is not None and thread.is_alive()

    # No further demand: the thread must die by itself.
    assert _wait_for(lambda: not thread.is_alive())


def test_poller_restarts_on_new_demand() -> None:
    """Demand after an idle stop spawns a fresh poller thread."""
    fake = FakeDoA()
    backend = _make_backend(fake)
    backend.DOA_POLL_INTERVAL_S = 0.01  # type: ignore[misc]
    backend.DOA_IDLE_STOP_S = 0.05  # type: ignore[misc]

    backend._doa_snapshot()
    first = backend._doa_thread
    assert first is not None
    assert _wait_for(lambda: not first.is_alive())

    backend._doa_snapshot()
    second = backend._doa_thread
    assert second is not None and second is not first and second.is_alive()

    backend._doa_stop.set()


def test_read_doa_cold_cache_reads_directly() -> None:
    """A one-shot REST read gets a real reading without waiting on the poller."""
    fake = FakeDoA(reading=(2.0, True))
    backend = _make_backend(fake)
    # Huge interval: the poller can't be the one producing the reading.
    backend.DOA_POLL_INTERVAL_S = 60.0  # type: ignore[misc]

    assert backend.read_doa() == (2.0, True)
    assert fake.calls == 1

    backend._doa_stop.set()


def test_read_doa_serves_fresh_cache_without_usb() -> None:
    """A second read inside the freshness window doesn't touch USB again."""
    fake = FakeDoA(reading=(2.0, True))
    backend = _make_backend(fake)
    backend.DOA_POLL_INTERVAL_S = 60.0  # type: ignore[misc]

    assert backend.read_doa() == (2.0, True)
    assert backend.read_doa() == (2.0, True)
    assert fake.calls == 1

    backend._doa_stop.set()


def test_read_doa_swallows_usb_errors() -> None:
    """A failing USB read reads as 'no DoA', never as an exception."""
    fake = FakeDoA(raise_on_read=True)
    backend = _make_backend(fake)
    backend.DOA_POLL_INTERVAL_S = 60.0  # type: ignore[misc]

    assert backend.read_doa() is None

    backend._doa_stop.set()


def test_snapshot_ignores_stale_cache() -> None:
    """Entries older than DOA_CACHE_FRESH_S are treated as absent."""
    fake = FakeDoA(reading=(2.0, True))
    backend = _make_backend(fake)
    backend.DOA_POLL_INTERVAL_S = 60.0  # type: ignore[misc]

    assert backend.read_doa() == (2.0, True)
    backend._last_doa_ts = time.monotonic() - 10.0
    assert backend._doa_snapshot() is None

    backend._doa_stop.set()


def test_close_joins_poller_and_releases_device() -> None:
    """Backend.close() joins the poller thread and drops the USB handle."""
    fake = FakeDoA(reading=(2.0, True))
    backend = _make_backend(fake)

    # Spin the poller up through a normal demand path.
    assert backend.read_doa() == (2.0, True)
    thread = backend._doa_thread
    assert thread is not None and thread.is_alive()

    backend.close()

    assert not thread.is_alive()
    assert backend._doa_thread is None
    # The handle was closed and dropped, so post-close reads are inert.
    assert fake.available is False
    assert backend.doa is None
    assert backend.read_doa() is None
    assert backend._doa_snapshot() is None


def test_build_state_dict_carries_doa_and_no_dead_payload() -> None:
    """The pushed frame has `doa` and no duplicated antenna payload."""
    backend = _make_backend(doa=None)
    state = backend.build_state_dict()
    assert "doa" in state
    # Per-motor head values are a real feature of the pose stream...
    assert "head_joint_positions" in state
    # ...but the antenna twin duplicated `antennas` verbatim and was dropped.
    assert "antennas_joint_positions" not in state
