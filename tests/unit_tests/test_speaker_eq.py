"""Unit tests for the runtime speaker-EQ gains resolution and element builder."""

import logging
import types

import gi
import numpy as np
import pytest

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

from reachy_mini.daemon import startup_app_config  # noqa: E402
from reachy_mini.media.audio_base import make_speaker_eq  # noqa: E402
from reachy_mini.media.audio_gstreamer import GStreamerAudio  # noqa: E402
from reachy_mini.media.audio_utils import (  # noqa: E402
    DEFAULT_SPEAKER_EQ_GAINS,
    resolve_speaker_eq_gains,
)

Gst.init([])
_LOG = logging.getLogger(__name__)


def _find_element(bin_: Gst.Bin, factory_name: str) -> Gst.Element | None:
    """Return the first child element built from ``factory_name``, or None."""
    it = bin_.iterate_recurse()
    while True:
        result, elem = it.next()
        if result != Gst.IteratorResult.OK:
            return None
        if elem.get_factory().get_name() == factory_name:
            return elem


def test_resolve_gains_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Falls back to the tested default when the config has no entry."""
    monkeypatch.setattr(startup_app_config, "get_speaker_eq_gains", lambda: None)
    assert resolve_speaker_eq_gains() == DEFAULT_SPEAKER_EQ_GAINS

    monkeypatch.setattr(startup_app_config, "get_speaker_eq_gains", lambda: [1.5] * 10)
    assert resolve_speaker_eq_gains() == [1.5] * 10


def test_config_gains_validation(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The daemon-config reader accepts a 10-float list and rejects the rest."""
    import json
    from pathlib import Path

    cfg = Path(str(tmp_path)) / "daemon_config.json"
    monkeypatch.setattr(startup_app_config, "_config_path", lambda: cfg)

    assert startup_app_config.get_speaker_eq_gains() is None  # missing file

    cfg.write_text(json.dumps({"speaker_eq_gains": [float(i) for i in range(10)]}))
    assert startup_app_config.get_speaker_eq_gains() == [float(i) for i in range(10)]

    cfg.write_text(json.dumps({"speaker_eq_gains": [1.0, 2.0, 3.0]}))  # wrong length
    assert startup_app_config.get_speaker_eq_gains() is None

    cfg.write_text(json.dumps({"speaker_eq_gains": "nope"}))  # not a list
    assert startup_app_config.get_speaker_eq_gains() is None

    # NaN / Infinity (json parses these) and out-of-range are rejected.
    cfg.write_text(json.dumps({"speaker_eq_gains": [0.0] * 9 + [float("nan")]}))
    assert startup_app_config.get_speaker_eq_gains() is None

    cfg.write_text(json.dumps({"speaker_eq_gains": [0.0] * 9 + [float("inf")]}))
    assert startup_app_config.get_speaker_eq_gains() is None

    cfg.write_text(json.dumps({"speaker_eq_gains": [0.0] * 9 + [13.0]}))  # > +12 dB
    assert startup_app_config.get_speaker_eq_gains() is None

    cfg.write_text(json.dumps({"speaker_eq_gains": [-30.0] + [0.0] * 9}))  # < -24 dB
    assert startup_app_config.get_speaker_eq_gains() is None

    # A JSON int too large to fit a float must be rejected, not crash.
    cfg.write_text(json.dumps({"speaker_eq_gains": [10**400] + [0.0] * 9}))
    assert startup_app_config.get_speaker_eq_gains() is None


def test_invalid_config_warns(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A present-but-invalid entry logs a warning; an absent key stays silent."""
    import json
    from pathlib import Path

    cfg = Path(str(tmp_path)) / "daemon_config.json"
    monkeypatch.setattr(startup_app_config, "_config_path", lambda: cfg)

    cfg.write_text(json.dumps({"other": 1}))  # key absent
    with caplog.at_level(logging.WARNING):
        assert startup_app_config.get_speaker_eq_gains() is None
    assert not caplog.records

    cfg.write_text(json.dumps({"speaker_eq_gains": [99.0] * 10}))  # present, invalid
    with caplog.at_level(logging.WARNING):
        assert startup_app_config.get_speaker_eq_gains() is None
    assert any("speaker_eq_gains" in r.message for r in caplog.records)


def test_make_speaker_eq_default_active(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no config the tested default is applied (EQ active out of the box)."""
    monkeypatch.setattr(startup_app_config, "get_speaker_eq_gains", lambda: None)
    # Pretend the Reachy Mini Audio output is present so the device gate passes.
    monkeypatch.setattr(
        "reachy_mini.media.audio_base.has_reachymini_asoundrc", lambda: True
    )
    eq_bin = make_speaker_eq(_LOG)
    assert eq_bin is not None
    eq = _find_element(eq_bin, "equalizer-10bands")
    assert eq is not None
    assert eq.get_property("band1") == pytest.approx(DEFAULT_SPEAKER_EQ_GAINS[1])
    assert _find_element(eq_bin, "audiodynamic") is not None  # limiter present


def test_make_speaker_eq_skipped_off_reachy_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EQ is skipped when the output is not the Reachy Mini Audio device."""
    monkeypatch.setattr(startup_app_config, "get_speaker_eq_gains", lambda: None)
    monkeypatch.setattr(
        "reachy_mini.media.audio_base.has_reachymini_asoundrc", lambda: False
    )
    monkeypatch.setattr(
        "reachy_mini.media.audio_base.get_audio_device", lambda _kind: None
    )
    assert make_speaker_eq(_LOG) is None


def test_make_speaker_eq_noop_when_all_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """All-zero gains disable the EQ (returns None, direct link kept)."""
    monkeypatch.setattr(startup_app_config, "get_speaker_eq_gains", lambda: [0.0] * 10)
    assert make_speaker_eq(_LOG) is None


def _force_reachy_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(startup_app_config, "get_speaker_eq_gains", lambda: None)
    monkeypatch.setattr(
        "reachy_mini.media.audio_base.has_reachymini_asoundrc", lambda: True
    )


def test_eq_bin_prevents_clipping(monkeypatch: pytest.MonkeyPatch) -> None:
    """A full-scale tone in a boosted band comes out of the EQ bin without clipping."""
    _force_reachy_output(monkeypatch)
    eq_bin = make_speaker_eq(_LOG)
    assert eq_bin is not None

    pipeline = Gst.Pipeline.new("clip-test")
    appsrc = Gst.ElementFactory.make("appsrc")
    appsink = Gst.ElementFactory.make("appsink")
    caps = Gst.Caps.from_string(
        "audio/x-raw,format=F32LE,rate=16000,channels=1,layout=interleaved"
    )
    appsrc.set_property("caps", caps)
    appsink.set_property("caps", caps)
    appsink.set_property("sync", False)
    for el in (appsrc, eq_bin, appsink):
        pipeline.add(el)
    appsrc.link(eq_bin)
    eq_bin.link(appsink)

    pipeline.set_state(Gst.State.PLAYING)
    # Full-scale sine at 3770 Hz, the band the reviewer measured boosting ~+8 dB,
    # so without the limiter the output would exceed full scale and clip.
    n = 8192
    t = np.arange(n) / 16000.0
    sine = np.sin(2 * np.pi * 3770.0 * t).astype(np.float32)
    appsrc.emit("push-buffer", Gst.Buffer.new_wrapped(sine.tobytes()))
    appsrc.emit("end-of-stream")

    peak = 0.0
    while True:
        sample = appsink.emit("try-pull-sample", Gst.SECOND)
        if sample is None:
            break
        buf = sample.get_buffer()
        ok, info = buf.map(Gst.MapFlags.READ)
        if ok:
            out = np.frombuffer(info.data, dtype=np.float32)
            if out.size:
                peak = max(peak, float(np.abs(out).max()))
            buf.unmap(info)
    pipeline.set_state(Gst.State.NULL)

    assert peak > 0.0  # audio actually flowed through the bin
    assert peak <= 1.0  # limiter kept the boosted tone from clipping


def _build_tee_bin(monkeypatch: pytest.MonkeyPatch) -> Gst.Bin:
    monkeypatch.setattr(
        "reachy_mini.media.audio_base.has_reachymini_asoundrc", lambda: True
    )
    inst = object.__new__(GStreamerAudio)
    inst.logger = _LOG
    # Satisfy the wobbler callback and __del__/cleanup on this half-built instance.
    inst._head_wobbler = None
    inst._doa = types.SimpleNamespace(close=lambda: None)
    inst._loop = types.SimpleNamespace(quit=lambda: None)
    inst._bus = types.SimpleNamespace(remove_watch=lambda: None)
    return GStreamerAudio._build_audiosink_tee_bin(inst)


def test_playback_path_includes_eq(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real speaker branch carries the EQ when gains are non-zero."""
    monkeypatch.setattr(startup_app_config, "get_speaker_eq_gains", lambda: None)
    assert _find_element(_build_tee_bin(monkeypatch), "equalizer-10bands") is not None


def test_playback_path_skips_eq_when_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real speaker branch has no EQ when all gains are zero."""
    monkeypatch.setattr(startup_app_config, "get_speaker_eq_gains", lambda: [0.0] * 10)
    assert _find_element(_build_tee_bin(monkeypatch), "equalizer-10bands") is None
