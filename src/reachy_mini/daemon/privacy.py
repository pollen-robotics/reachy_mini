"""Privacy mode: cut the camera and the microphones for every consumer.

Both cuts are made where no consumer can be missed:

* the **microphones** are muted by the audio chip's own mute circuit. On-device
  apps read the sound card directly, without going through the daemon, so
  nothing in the daemon's pipeline could silence them.
* the **camera** frames are blacked out inside the daemon's media pipeline,
  before it splits into the local feed and the WebRTC stream.

The pipeline itself keeps running, so apps and remote sessions stay connected:
they simply receive silence and a black picture until privacy mode ends.

Privacy mode is on while at least one *source* asks for it: ``"hat"`` (the NFC
privacy hat is on the head) or ``"api"`` (REST, SDK or data-channel request).
A request from one source never clears another's, so software cannot switch
off a hat that is still on the head.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from reachy_mini.media.audio_control_utils import set_microphones_muted

if TYPE_CHECKING:
    from reachy_mini.nfc import NfcTag

logger = logging.getLogger(__name__)

# Text carried by the NFC tag of a privacy hat.
PRIVACY_TAG_CONTENT = "privacy"

# The only source that survives a daemon restart: a hat still on the head is
# reported again by the reader within one poll.
PERSISTED_SOURCE = "api"

_STATE_PATH = Path.home() / ".config" / "reachy_mini" / "privacy.json"


def is_privacy_tag(tag: NfcTag) -> bool:
    """Whether the tag on the reader is a privacy hat."""
    return tag.present and tag.content == PRIVACY_TAG_CONTENT


class PrivacyMode:
    """The daemon's privacy switch."""

    def __init__(
        self,
        media_server: Any | None,
        state_path: Path = _STATE_PATH,
        mute_microphones: Callable[[bool], bool] = set_microphones_muted,
    ) -> None:
        """Restore the saved state and apply it, without any sound.

        Applying it even when privacy mode is off is deliberate: a daemon that
        crashed while private must not leave the microphones muted for ever.

        Args:
            media_server: the daemon's ``GstMediaServer``, or None when the
                daemon runs without media.
            state_path: where an API request is remembered across restarts.
            mute_microphones: mutes or unmutes the microphones, and returns
                whether the audio chip confirmed it (injectable for tests).

        """
        self._media_server = media_server
        self._state_path = state_path
        self._mute_microphones = mute_microphones
        self._lock = threading.Lock()
        self._sources: set[str] = set()
        if self._load():
            self._sources.add(PERSISTED_SOURCE)
        self._apply(self.enabled)

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
            if source == PERSISTED_SOURCE:
                self._save(enabled)

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

    def _load(self) -> bool:
        try:
            return json.loads(self._state_path.read_text()).get("enabled") is True
        except FileNotFoundError:
            return False
        except Exception as e:  # noqa: BLE001 - never stop the daemon over this
            logger.warning("Could not read the privacy setting: %s", e)
            return False

    def _save(self, enabled: bool) -> None:
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(json.dumps({"enabled": enabled}))
        except Exception as e:  # noqa: BLE001 - never stop the daemon over this
            logger.warning("Could not write the privacy setting: %s", e)
