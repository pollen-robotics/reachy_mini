"""Privacy mode: cut the camera and the microphones for every consumer.

Both cuts are made where no consumer can be missed:

* the **microphones** are muted by the audio chip's own mute circuit. On-device
  apps read the sound card directly, without going through the daemon, so
  nothing in the daemon's pipeline could silence them.
* the **camera** frames are blacked out inside the daemon's media pipeline,
  before it splits into the local feed and the WebRTC stream.

The pipeline itself keeps running, so apps and remote sessions stay connected:
they simply receive silence and a black picture until privacy mode ends.

Privacy mode is on while at least one *source* asks for it: ``"api"`` (REST,
SDK or data-channel request) or ``"hat"`` (the NFC privacy hat is on the head,
wired up together with the NFC module).
A request from one source never clears another's, so software cannot switch
off a hat that is still on the head.

Nothing is saved: the daemon always starts with privacy mode off, so a
forgotten request can never leave a robot without camera or microphones after
a reboot or an update. A hat still on the head is reported again by the NFC
reader on its first read.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

from reachy_mini.media.audio_control_utils import set_microphones_muted

logger = logging.getLogger(__name__)


class PrivacyMode:
    """The daemon's privacy switch."""

    def __init__(
        self,
        media_server: Any | None,
        mute_microphones: Callable[[bool], bool] = set_microphones_muted,
    ) -> None:
        """Start with privacy mode off and apply that, without any sound.

        Applying it is deliberate: a daemon that crashed while private must
        not leave the microphones muted for ever.

        Args:
            media_server: the daemon's ``GstMediaServer``, or None when the
                daemon runs without media.
            mute_microphones: mutes or unmutes the microphones, and returns
                whether the audio chip confirmed it (injectable for tests).

        """
        self._media_server = media_server
        self._mute_microphones = mute_microphones
        self._lock = threading.Lock()
        self._sources: set[str] = set()
        self._apply(False)

    @property
    def enabled(self) -> bool:
        """Whether privacy mode is on."""
        return bool(self._sources)

    @property
    def sources(self) -> list[str]:
        """Who is asking for privacy mode right now."""
        return sorted(self._sources)

    def set(self, source: str, enabled: bool) -> bool:
        """Record one source's request and return whether privacy mode is on.

        The sound is played after the cut, so hearing it means the robot is
        already private.
        """
        with self._lock:
            was_enabled = self.enabled
            if enabled:
                self._sources.add(source)
            else:
                self._sources.discard(source)

            if self.enabled != was_enabled:
                self._apply(self.enabled)
                if self._media_server is not None:
                    self._media_server.play_sound(
                        "privacy_on.wav" if self.enabled else "privacy_off.wav"
                    )
                logger.info(
                    "Privacy mode %s (sources: %s).",
                    "on" if self.enabled else "off",
                    self.sources,
                )
            return self.enabled

    def _apply(self, enabled: bool) -> None:
        if not self._mute_microphones(enabled) and enabled:
            logger.warning(
                "Privacy mode: the audio chip did not confirm the microphone mute."
            )
        if self._media_server is not None:
            self._media_server.set_privacy(enabled)
