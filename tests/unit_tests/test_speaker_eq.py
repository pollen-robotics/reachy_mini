"""Unit tests for the runtime speaker-EQ gains resolution and element builder."""

from typing import cast

import gi
import pytest

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

from reachy_mini.daemon import startup_app_config  # noqa: E402
from reachy_mini.media.audio_gstreamer import GStreamerAudio  # noqa: E402
from reachy_mini.media.audio_utils import (  # noqa: E402
    DEFAULT_SPEAKER_EQ_GAINS,
    resolve_speaker_eq_gains,
)

Gst.init([])


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


def test_make_speaker_eq_default_active(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no config the tested default is applied (EQ active out of the box)."""
    # _make_speaker_eq ignores self, so a dummy instance is enough to exercise it.
    monkeypatch.setattr(startup_app_config, "get_speaker_eq_gains", lambda: None)
    dummy = cast(GStreamerAudio, object())
    eq = GStreamerAudio._make_speaker_eq(dummy)
    assert eq is not None
    assert eq.get_factory().get_name() == "equalizer-10bands"
    assert eq.get_property("band1") == pytest.approx(DEFAULT_SPEAKER_EQ_GAINS[1])


def test_make_speaker_eq_noop_when_all_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """All-zero gains disable the EQ (helper returns None, direct link kept)."""
    monkeypatch.setattr(startup_app_config, "get_speaker_eq_gains", lambda: [0.0] * 10)
    dummy = cast(GStreamerAudio, object())
    assert GStreamerAudio._make_speaker_eq(dummy) is None
