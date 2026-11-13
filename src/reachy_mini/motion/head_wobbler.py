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

import importlib
import logging
import os
import threading
import time
from collections.abc import Callable
from typing import Any

from gi.repository import GLib
from numpy.typing import NDArray

from reachy_mini.motion import speech_tapper

logger = logging.getLogger(__name__)

# Public type alias; re-exported by ``media/*`` modules.
SpeechOffsets = tuple[float, float, float, float, float, float]

# Opt-in speech tappers, selected with the WOBBLER_VERSION environment
# variable. Unset (or "v0") keeps the official wobbler, ``speech_tapper``.
# Imported lazily so the default path pays nothing for them.
TAPPER_VERSIONS: dict[str, str] = {
    "v4": "reachy_mini.motion.speech_tapper_v4",
    "v5": "reachy_mini.motion.speech_tapper_v5",
    "v6": "reachy_mini.motion.speech_tapper_v6",
}


def _load_tapper() -> tuple[str, Any]:
    """Return (version, tapper module) for the WOBBLER_VERSION env var."""
    version = os.environ.get("WOBBLER_VERSION", "").strip().lower()
    if version in ("", "v0"):
        return "v0", speech_tapper
    module_name = TAPPER_VERSIONS.get(version)
    if module_name is None:
        logger.warning(
            "Unknown WOBBLER_VERSION %r, using the official wobbler (v0)", version
        )
        return "v0", speech_tapper
    return version, importlib.import_module(module_name)


def _energy_from_env() -> float:
    """Return WOBBLER_ENERGY as a float, 1.0 when unset or not a number."""
    raw = os.environ.get("WOBBLER_ENERGY", "")
    if not raw.strip():
        return 1.0
    try:
        return float(raw)
    except ValueError:
        logger.warning("WOBBLER_ENERGY %r is not a number, using 1.0", raw)
        return 1.0


class HeadWobbler:
    """PTS-driven scheduler that turns audio into timed head offsets."""

    _ZERO_OFFSETS: SpeechOffsets = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

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

        self.version, tapper = _load_tapper()
        if self.version != "v0":
            logger.info("Head wobbler uses speech tapper %s", self.version)
        self._sway_cls = tapper.SwayRollRT
        self._hop_ms = int(tapper.HOP_MS)
        self._sample_rate = int(sample_rate)
        # Emotional colouring, only for tappers that accept it (v6). Both
        # inputs are optional: WOBBLER_EMOTION defaults to "neutral" and
        # WOBBLER_ENERGY to 1.0. set_emotion() overrides them at runtime.
        self._sway_kwargs: dict[str, Any] = {}
        if hasattr(self._sway_cls, "set_emotion"):
            self._sway_kwargs = {
                "emotion": os.environ.get("WOBBLER_EMOTION", "neutral"),
                "energy": _energy_from_env(),
            }
        self.sway = self._make_sway()

        self._lock = threading.Lock()
        self._sway_lock = threading.Lock()
        # Bumped on stop/reset so in-flight GLib timeouts no-op when fired.
        self._generation = 0

    def _make_sway(self) -> Any:
        return self._sway_cls(sample_rate=self._sample_rate, **self._sway_kwargs)

    def set_emotion(self, emotion: str, energy: float | None = None) -> None:
        """Change the emotional colouring at runtime.

        No-op with a tapper that has no emotion input (v0, v4, v5). Unknown
        names fall back to neutral with a warning. ``energy`` is kept as is
        when omitted. The choice survives :meth:`reset`.
        """
        with self._sway_lock:
            if not self._sway_kwargs:
                logger.debug("Speech tapper %s has no emotion input", self.version)
                return
            self.sway.set_emotion(emotion, energy)
            self._sway_kwargs["emotion"] = emotion
            if energy is not None:
                self._sway_kwargs["energy"] = energy

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
            self.sway = self._make_sway()
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
