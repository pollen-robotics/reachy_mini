"""Shared software acoustic echo cancellation (AEC) helpers.

The Reachy Mini hardware path normally relies on the XVF3800 / XMOS audio
board for acoustic echo cancellation: on Wireless robots through the
``.asoundrc`` ALSA loopback, on Lite / desktop through the ReSpeaker USB
dongle. When neither is in the audio path (typically simulation mode
``--sim`` / ``--mockup-sim`` on a developer workstation), the
conversation loop hears its own voice through the laptop speakers and
collapses on feedback.

This module centralises the detection logic and the GStreamer element
chains used to insert ``webrtcdsp`` + ``webrtcechoprobe`` (from
``gst-plugins-bad``) inline on the record / playback pipelines. Two
backends consume these helpers:

* :mod:`reachy_mini.media.audio_gstreamer` for the LOCAL Python SDK
  audio backend (``MediaManager(backend=LOCAL)``).
* :mod:`reachy_mini.media.media_server` for the WebRTC capture +
  playback pipelines that bridge the daemon to remote clients (e.g.
  the mobile conversation app).

The ``webrtcdsp`` element on the recording side and the
``webrtcechoprobe`` element on the playback side find each other
through a process-wide registry keyed by the same shared name (see
:data:`AEC_PROBE_NAME`). Both must agree on rate, channel count and
sample format, hence :data:`AEC_SAMPLE_RATE` / :data:`AEC_CHANNELS`
and :func:`make_aec_capsfilter`.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import gi

from reachy_mini.media.audio_utils import has_reachymini_asoundrc
from reachy_mini.media.device_detection import get_audio_device

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

AEC_PROBE_NAME = "reachymini_aec_probe"
"""Shared name that pairs the recording-side ``webrtcdsp`` with the
playback-side ``webrtcechoprobe``. Both elements communicate via a
process-wide registry indexed by this string; the value just needs to
be stable and unique within a process."""

AEC_SAMPLE_RATE = 16_000
"""Sample rate (Hz) the ``webrtcdsp`` / ``webrtcechoprobe`` pair share.
16 kHz mono / stereo is the canonical voice rate WebRTC's AEC was
trained on."""

AEC_CHANNELS = 2
"""Channel count for the AEC stage. The dsp and the probe MUST agree
on channel count, so we canonicalise both sides to this value."""

_AEC_ELEMENT_NAMES: Tuple[str, str] = ("webrtcdsp", "webrtcechoprobe")


def resolve_sw_aec_enabled(logger: logging.Logger) -> bool:
    """Decide whether to insert a ``webrtcdsp`` + ``webrtcechoprobe`` pair.

    Pure auto-detection from the same audio device parsing the rest
    of the backend uses. Returns ``True`` only when no hardware AEC
    sits in the audio path and the matching ``gst-plugins-bad``
    elements are actually packaged:

    * ``has_reachymini_asoundrc()`` → Wireless XMOS does AEC, skip.
    * ``get_audio_device("Source")`` → XVF3800 / ReSpeaker on USB
      does AEC, skip.
    * ``webrtcdsp`` or ``webrtcechoprobe`` missing → skip with a loud
      warning so callers can install the missing plugin.

    Anything else (typically ``--sim`` / ``--mockup-sim`` or a
    developer workstation) gets the software AEC.

    Args:
        logger: Logger to use for diagnostic output. Each call to
            ``resolve_sw_aec_enabled`` emits exactly one log line so
            startup logs always tell the operator why the AEC stage
            was or wasn't inserted.

    Returns:
        ``True`` when the software AEC stage should be added,
        ``False`` otherwise.

    """
    if has_reachymini_asoundrc():
        logger.info(
            "Wireless `.asoundrc` detected - relying on XMOS "
            "hardware AEC, skipping software AEC stage."
        )
        return False
    if get_audio_device("Source") is not None:
        logger.info(
            "ReSpeaker / Reachy Mini Audio card detected - "
            "relying on XVF3800 hardware AEC, skipping software "
            "AEC stage."
        )
        return False

    for name in _AEC_ELEMENT_NAMES:
        if Gst.ElementFactory.find(name) is None:
            logger.warning(
                "No XVF3800 audio board detected AND GStreamer "
                "element '%s' is unavailable in this build "
                "(install gst-plugins-bad). Conversation audio "
                "will loop back through the laptop mic - use a "
                "headset to work around it.",
                name,
            )
            return False

    logger.warning(
        "No XVF3800 audio board detected - enabling software AEC "
        "(webrtcdsp + webrtcechoprobe). Quality is best-effort; "
        "for production voice loops use the ReSpeaker USB dongle "
        "or a Wireless robot with the XMOS AEC loopback."
    )
    return True


def make_aec_capsfilter(
    sample_rate: int = AEC_SAMPLE_RATE,
    channels: int = AEC_CHANNELS,
) -> Gst.Element:
    """Force ``webrtcdsp``-friendly caps (S16LE interleaved).

    ``webrtcdsp`` accepts ``S16LE/interleaved`` or
    ``F32LE/non-interleaved``; most surrounding pipelines use
    ``F32LE/interleaved``, so we pin S16LE on the AEC link and let
    the surrounding ``audioconvert`` elements bridge the format gap
    on both sides.

    The dsp and the probe MUST agree on rate and channel count, so
    callers should pass identical values on both sides of the AEC
    chain. Defaults map to :data:`AEC_SAMPLE_RATE` /
    :data:`AEC_CHANNELS`.
    """
    capsfilter = Gst.ElementFactory.make("capsfilter")
    if capsfilter is None:
        raise RuntimeError("Failed to create capsfilter element for AEC link")
    capsfilter.set_property(
        "caps",
        Gst.Caps.from_string(
            "audio/x-raw,"
            "format=S16LE,"
            "layout=interleaved,"
            f"rate={sample_rate},"
            f"channels={channels}"
        ),
    )
    return capsfilter


def build_aec_dsp_chain(
    sample_rate: int = AEC_SAMPLE_RATE,
    channels: int = AEC_CHANNELS,
    probe_name: str = AEC_PROBE_NAME,
) -> Tuple[Gst.Element, Gst.Element, Gst.Element]:
    """Build the recording-side AEC chain.

    Returns a (capsfilter, webrtcdsp, audioconvert) tuple. Callers
    are expected to insert it between their existing
    ``audioresample`` (or whatever produces float audio) and the
    downstream sink. The ``audioconvert`` after the dsp converts the
    canonical S16LE format back to whatever the rest of the pipeline
    consumes.

    Linking order::

        upstream → capsfilter → webrtcdsp → audioconvert → downstream
    """
    capsfilter = make_aec_capsfilter(sample_rate=sample_rate, channels=channels)
    dsp = Gst.ElementFactory.make("webrtcdsp")
    ac_post_dsp = Gst.ElementFactory.make("audioconvert")
    if not all([capsfilter, dsp, ac_post_dsp]):
        raise RuntimeError("Failed to create AEC DSP elements (webrtcdsp chain)")
    dsp.set_property("probe", probe_name)
    dsp.set_property("delay-agnostic", True)
    dsp.set_property("extended-filter", True)
    return capsfilter, dsp, ac_post_dsp


def make_aec_probe(probe_name: str = AEC_PROBE_NAME) -> Gst.Element:
    """Create a standalone ``webrtcechoprobe`` element.

    The probe registers itself in a process-wide registry keyed by its
    ``name`` property as soon as the property is set. This means the
    matching ``webrtcdsp`` can find it during its own
    ``set_state(PLAYING)`` even before the probe is added to any
    pipeline.

    Callers can therefore pre-create the probe at startup, then add it
    to an actual playback pipeline later (e.g. when a WebRTC peer
    arrives). Reusing the same probe element across peer reconnects
    keeps the registry entry stable.
    """
    probe = Gst.ElementFactory.make("webrtcechoprobe")
    if probe is None:
        raise RuntimeError("Failed to create webrtcechoprobe element")
    probe.set_property("name", probe_name)
    return probe


def build_aec_probe_chain(
    sample_rate: int = AEC_SAMPLE_RATE,
    channels: int = AEC_CHANNELS,
    probe_name: str = AEC_PROBE_NAME,
    probe: Optional[Gst.Element] = None,
) -> Tuple[Gst.Element, Gst.Element, Gst.Element]:
    """Build the playback-side AEC chain.

    Returns an (audioconvert, capsfilter, webrtcechoprobe) tuple.
    Callers are expected to insert it between their decoded /
    decompressed audio source and the speaker / tee branch. The
    leading ``audioconvert`` converts whatever the upstream produced
    to the canonical S16LE format the probe consumes.

    Linking order::

        upstream → audioconvert → capsfilter → webrtcechoprobe → downstream

    The probe is a passthrough element data-wise: every speaker
    sample is also handed to the matching ``webrtcdsp`` (paired by
    :data:`AEC_PROBE_NAME`) as the reference signal it must subtract
    from the mic stream.

    Args:
        sample_rate: Sample rate the probe will run at. Must match the
            paired ``webrtcdsp``'s rate, hence the :data:`AEC_SAMPLE_RATE`
            default.
        channels: Channel count the probe will run at. Must match the
            paired ``webrtcdsp``.
        probe_name: Name used to pair the probe with its ``webrtcdsp``.
        probe: Optional pre-created probe element to insert into the
            chain instead of allocating a fresh one. Used by callers
            that need the probe to exist before the playback pipeline
            is ready (e.g. so the paired ``webrtcdsp`` finds it at
            startup time).

    """
    ac_pre_probe = Gst.ElementFactory.make("audioconvert")
    capsfilter_probe = make_aec_capsfilter(sample_rate=sample_rate, channels=channels)
    if probe is None:
        probe = make_aec_probe(probe_name=probe_name)
    if not all([ac_pre_probe, capsfilter_probe, probe]):
        raise RuntimeError(
            "Failed to create AEC probe elements (webrtcechoprobe chain)"
        )
    return ac_pre_probe, capsfilter_probe, probe


__all__ = [
    "AEC_PROBE_NAME",
    "AEC_SAMPLE_RATE",
    "AEC_CHANNELS",
    "resolve_sw_aec_enabled",
    "make_aec_capsfilter",
    "make_aec_probe",
    "build_aec_dsp_chain",
    "build_aec_probe_chain",
]
