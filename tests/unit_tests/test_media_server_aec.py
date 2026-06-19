"""Unit tests for the WebRTC media server's software AEC wiring.

When no Reachy Mini Audio card is present the mic source falls back to
``autoaudiosrc`` (so the XMOS hardware AEC is unavailable) and
``GstMediaServer`` inserts ``webrtcdsp`` on the mic path, paired by name
(``AEC_PROBE_NAME``) with a ``webrtcechoprobe`` on the incoming-audio speaker
branch. These tests cover the gating (only on ``autoaudiosrc``) and the
S16LE/48 kHz convert-resample-caps chain both elements require.

``GstMediaServer.__init__`` boots a full GStreamer pipeline and a GLib main
loop, which is overkill here, so we bypass it with ``object.__new__`` and only
set the attributes the audio-config code touches — same approach as
``test_media_server_watchdog.py``.
"""

from __future__ import annotations

import logging
from typing import cast
from unittest.mock import MagicMock

import gi
import pytest

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

from reachy_mini.media.audio_base import (  # noqa: E402
    AEC_CHANNELS,
    AEC_PROBE_NAME,
    AEC_RATE,
)
from reachy_mini.media.media_server import GstMediaServer  # noqa: E402

Gst.init([])


def _aec_plugins_available() -> bool:
    """Return whether the optional webrtcdsp/webrtcechoprobe elements exist."""
    return all(
        Gst.ElementFactory.find(name) is not None
        for name in ("webrtcdsp", "webrtcechoprobe")
    )


def _make_server() -> GstMediaServer:
    """Build a minimal ``GstMediaServer`` without booting GStreamer/GLib."""
    server = cast(GstMediaServer, object.__new__(GstMediaServer))
    server._logger = logging.getLogger("test_media_server_aec")
    server._aec_enabled = False
    server._webrtcechoprobe = None
    # Destructor (__del__ -> close()) touches these at GC time.
    server._loop = MagicMock()
    server._bus_sender = MagicMock()
    return server


def _iter_elements(pipeline: Gst.Pipeline) -> list[Gst.Element]:
    elems: list[Gst.Element] = []
    it = pipeline.iterate_elements()
    while True:
        ok, elem = it.next()
        if ok != Gst.IteratorResult.OK:
            break
        elems.append(elem)
    return elems


def _factory_names(pipeline: Gst.Pipeline) -> set[str]:
    return {
        e.get_factory().get_name()
        for e in _iter_elements(pipeline)
        if e.get_factory() is not None
    }


def test_make_aec_caps_chain_emits_s16le_48k() -> None:
    """The chain is audioconvert → audioresample → capsfilter(S16LE @ AEC_RATE)."""
    server = _make_server()

    chain = GstMediaServer._make_aec_caps_chain(server)

    assert [e.get_factory().get_name() for e in chain] == [
        "audioconvert",
        "audioresample",
        "capsfilter",
    ]
    caps = chain[-1].get_property("caps").to_string()
    assert "audio/x-raw" in caps
    assert "format=(string)S16LE" in caps
    assert f"rate=(int){AEC_RATE}" in caps
    assert f"channels=(int){AEC_CHANNELS}" in caps


def test_aec_rate_is_a_supported_webrtc_rate() -> None:
    """webrtcdsp/webrtcechoprobe only accept 8/16/32/48 kHz."""
    assert AEC_RATE in (8_000, 16_000, 32_000, 48_000)


@pytest.mark.skipif(
    not _aec_plugins_available(), reason="webrtcdsp/webrtcechoprobe not installed"
)
def test_configure_audio_enables_aec_on_autoaudiosrc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The autoaudiosrc fallback inserts webrtcdsp wired to AEC_PROBE_NAME."""
    server = _make_server()
    monkeypatch.setattr(
        server, "_build_audio_source", lambda: Gst.ElementFactory.make("autoaudiosrc")
    )

    pipeline = Gst.Pipeline.new("test_sender")
    # `identity` stands in for webrtcsink: a single-pad, always-linkable sink.
    webrtcsink = Gst.ElementFactory.make("identity")
    pipeline.add(webrtcsink)

    server._configure_audio(pipeline, webrtcsink)

    assert server._aec_enabled is True
    assert "webrtcdsp" in _factory_names(pipeline)

    dsp = next(
        e for e in _iter_elements(pipeline) if e.get_factory().get_name() == "webrtcdsp"
    )
    assert dsp.get_property("probe") == AEC_PROBE_NAME

    # The probe must be created up front (it is what webrtcdsp binds to at
    # start) and named so the global registry lookup in webrtcdsp succeeds.
    assert server._webrtcechoprobe is not None
    assert server._webrtcechoprobe.get_name() == AEC_PROBE_NAME


def test_configure_audio_skips_aec_for_named_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A detected card (non-autoaudiosrc source) leaves AEC off, mic linked plainly."""
    server = _make_server()
    # audiotestsrc stands in for a platform-native source (alsasrc/pulsesrc/…):
    # any factory whose name is not "autoaudiosrc" must not trigger AEC.
    monkeypatch.setattr(
        server, "_build_audio_source", lambda: Gst.ElementFactory.make("audiotestsrc")
    )

    pipeline = Gst.Pipeline.new("test_sender_card")
    webrtcsink = Gst.ElementFactory.make("identity")
    pipeline.add(webrtcsink)

    server._configure_audio(pipeline, webrtcsink)

    assert server._aec_enabled is False
    assert "webrtcdsp" not in _factory_names(pipeline)


def _dsp_start_error(probe_name: str) -> str | None:
    """Start audiotestsrc→…→webrtcdsp(probe_name) and return any ERROR message.

    webrtcdsp looks its probe up by name in a process-global registry at start,
    so callers use a unique ``probe_name`` to stay isolated from other tests'
    probes that linger in that registry until garbage-collected.
    """
    pipeline = Gst.Pipeline.new("dsp_start")
    src = Gst.ElementFactory.make("audiotestsrc")
    src.set_property("is-live", True)
    ac = Gst.ElementFactory.make("audioconvert")
    ar = Gst.ElementFactory.make("audioresample")
    cf = Gst.ElementFactory.make("capsfilter")
    cf.set_property(
        "caps",
        Gst.Caps.from_string(
            f"audio/x-raw,format=S16LE,rate={AEC_RATE},"
            f"channels={AEC_CHANNELS},layout=interleaved"
        ),
    )
    dsp = Gst.ElementFactory.make("webrtcdsp")
    dsp.set_property("probe", probe_name)
    sink = Gst.ElementFactory.make("fakesink")
    for el in (src, ac, ar, cf, dsp, sink):
        pipeline.add(el)
    src.link(ac)
    ac.link(ar)
    ar.link(cf)
    cf.link(dsp)
    dsp.link(sink)

    pipeline.set_state(Gst.State.PLAYING)
    pipeline.get_state(2 * Gst.SECOND)
    err = None
    bus = pipeline.get_bus()
    msg = bus.pop_filtered(Gst.MessageType.ERROR)
    if msg is not None:
        err = msg.parse_error()[0].message
    pipeline.set_state(Gst.State.NULL)
    return err


@pytest.mark.skipif(
    not _aec_plugins_available(), reason="webrtcdsp/webrtcechoprobe not installed"
)
def test_webrtcdsp_fails_to_start_without_a_probe() -> None:
    """Regression: webrtcdsp errors at start if no echo probe of its name exists."""
    name = "test_aec_probe_absent"
    err = _dsp_start_error(name)
    assert err is not None
    assert name in err


@pytest.mark.skipif(
    not _aec_plugins_available(), reason="webrtcdsp/webrtcechoprobe not installed"
)
def test_webrtcdsp_starts_when_probe_exists() -> None:
    """A pre-created, named webrtcechoprobe lets webrtcdsp start cleanly."""
    name = "test_aec_probe_present"
    probe = Gst.ElementFactory.make("webrtcechoprobe")
    probe.set_property("name", name)
    try:
        assert _dsp_start_error(name) is None
    finally:
        probe.set_state(Gst.State.NULL)
