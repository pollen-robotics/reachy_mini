"""PTS-driven head wobbler.

Drives 6-DOF head movement offsets from PCM audio analysed by
:class:`SwayRollRT` (the speech tapper). Each call to :meth:`feed`
turns one PCM chunk into a list of per-hop sway dicts and registers a
``GLib.timeout_add`` for each, firing the offset callback at the
audio's actual playback time (computed by the caller from buffer PTS +
audiosink latency).

There is no background thread: scheduling runs on whichever GLib main
loop the caller's pipeline already uses for its bus watch.
"""

import logging
import os
import threading
import time
from collections.abc import Callable
from types import ModuleType
from typing import Any

from gi.repository import GLib
from numpy.typing import NDArray

from reachy_mini.motion import (
    speech_tapper,
    speech_tapper_v1,
    speech_tapper_v2,
    speech_tapper_v3,
)

logger = logging.getLogger(__name__)

# Public type alias; re-exported by ``media/*`` modules.
SpeechOffsets = tuple[float, float, float, float, float, float]


class HeadWobbler:
    """PTS-driven scheduler that turns audio into timed head offsets."""

    _TAPPER_VERSIONS: dict[str, ModuleType] = {
        "v1": speech_tapper_v1,
        "v2": speech_tapper_v2,
        "v3": speech_tapper_v3,
    }

    _ZERO_OFFSETS: SpeechOffsets = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    @classmethod
    def _load_sway_class(cls) -> tuple[int, type]:
        """Return (HOP_MS, SwayRollRT class) based on WOBBLER_VERSION env var."""
        mod = cls._TAPPER_VERSIONS.get(
            os.environ.get("WOBBLER_VERSION", ""), speech_tapper
        )
        return mod.HOP_MS, mod.SwayRollRT

    def __init__(
        self,
        set_speech_offsets: Callable[[SpeechOffsets], None],
        sample_rate: int,
    ) -> None:
        """Initialize the wobbler with the offset callback and audio rate.

        Args:
            set_speech_offsets: Called with a 6-tuple of head offsets per hop.
            sample_rate: Sample rate of the PCM that will be fed via
                :meth:`feed` — must match the wobbler appsink's caps.

        """
        self._apply_offsets = set_speech_offsets

        self._hop_ms, sway_cls = self._load_sway_class()
        version = os.environ.get("WOBBLER_VERSION", "v0")
        logger.info("Using wobbler version: %s (%s)", version, sway_cls.__module__)
        self._sway_cls = sway_cls
        self._sample_rate = int(sample_rate)
        self.sway = sway_cls(sample_rate=self._sample_rate)

        self._lock = threading.Lock()
        self._sway_lock = threading.Lock()
        # Bumped on stop/reset so in-flight GLib timeouts no-op when fired.
        self._generation = 0

    def start(self) -> None:
        """Reset DSP and hop generation. Idempotent."""
        with self._lock:
            self._generation += 1
        with self._sway_lock:
            self.sway.reset()
        logger.debug("Head wobbler started")

    def stop(self) -> None:
        """Cancel pending offsets and zero the head."""
        with self._lock:
            self._generation += 1
        self._apply_offsets(self._ZERO_OFFSETS)
        logger.debug("Head wobbler stopped")

    def reset(self) -> None:
        """Cancel pending offsets, recreate DSP state, zero the head."""
        with self._lock:
            self._generation += 1
        with self._sway_lock:
            self.sway = self._sway_cls(sample_rate=self._sample_rate)
        self._apply_offsets(self._ZERO_OFFSETS)

    def feed(
        self,
        pcm: NDArray[Any],
        play_at_monotonic_ns: int,
    ) -> None:
        """Schedule per-hop offsets for *pcm* against its playback time.

        Args:
            pcm: Float32 mono samples at this wobbler's ``sample_rate``.
            play_at_monotonic_ns: ``time.monotonic_ns()``-comparable
                instant at which the *first* sample of *pcm* will be
                heard from the speaker. Subsequent hops are scheduled at
                ``play_at_monotonic_ns + i * HOP_MS * 1_000_000``.

        """
        with self._sway_lock:
            results = self.sway.feed(pcm)
        if not results:
            return

        with self._lock:
            generation = self._generation

        hop_ns = self._hop_ms * 1_000_000
        now_ns = time.monotonic_ns()

        # Skip hops more than one hop's worth in the past (genuinely
        # stale); clamp small sub-hop negatives to 0 so they fire on
        # the next main-loop iteration.
        stale_threshold_ms = -self._hop_ms
        for i, hop in enumerate(results):
            target_ns = play_at_monotonic_ns + i * hop_ns
            delay_ms = (target_ns - now_ns) // 1_000_000
            if delay_ms < stale_threshold_ms:
                continue
            offsets: SpeechOffsets = (
                hop["x_mm"] / 1000.0,
                hop["y_mm"] / 1000.0,
                hop["z_mm"] / 1000.0,
                hop["roll_rad"],
                hop["pitch_rad"],
                hop["yaw_rad"],
            )
            GLib.timeout_add(max(0, int(delay_ms)), self._fire, offsets, generation)

    def _fire(self, offsets: SpeechOffsets, generation: int) -> bool:
        """GLib timeout callback. Returns False so the source is removed."""
        with self._lock:
            current = self._generation
        if generation == current:
            self._apply_offsets(offsets)
        return False  # one-shot
