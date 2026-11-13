"""Unit tests for the opt-in speech tappers (v4, v5, v6) and their selection."""  # noqa: D100

import importlib
import logging
import math

import numpy as np
import pytest

from reachy_mini.motion import speech_tapper_v5, speech_tapper_v6

SR = 16_000
AXES = ("pitch_rad", "yaw_rad", "roll_rad", "x_mm", "y_mm", "z_mm")
VERSIONS = ("v4", "v5", "v6")
EMOTIONS = speech_tapper_v6.EMOTIONS


def _speech_like(seconds: float = 12.0, amplitude: float = 0.25) -> np.ndarray:
    """Synthetic speech: harmonic source, moving F0, 4 Hz syllables, 2.1 s phrases."""
    rng = np.random.default_rng(0)
    t = np.arange(int(SR * seconds)) / SR
    f0 = 140 + 40 * np.sin(2 * np.pi * 0.35 * t) + 25 * np.sin(2 * np.pi * 1.3 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    voiced = sum(np.sin(k * phase) / k for k in range(1, 6))
    syllables = 0.35 + 0.65 * np.clip(np.sin(2 * np.pi * 4.0 * t), 0, None) ** 1.5
    phrases = ((t % 3.0) < 2.1).astype(float)
    noise = 0.0003 * rng.standard_normal(t.size)
    return (amplitude * voiced * syllables * phrases + noise).astype(np.float32)


def _noise(db: float, seconds: float = 2.0) -> np.ndarray:
    rng = np.random.default_rng(1)
    rms = 10.0 ** (db / 20.0)
    return (rms * rng.standard_normal(int(SR * seconds))).astype(np.float32)


def _tapper(version: str, **kwargs):
    module = importlib.import_module(f"reachy_mini.motion.speech_tapper_{version}")
    return module.SwayRollRT(sample_rate=SR, **kwargs)


def _as_array(hops: list[dict[str, float]]) -> np.ndarray:
    return np.array([[hop[k] for k in AXES] for hop in hops])


# ---------------------------------------------------------------------------
# Strict gate: silences are exactly silent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize(
    "signal",
    [np.zeros(2 * SR, dtype=np.float32), _noise(-70.0)],
    ids=["digital_silence", "noise_-70dB"],
)
def test_silence_gives_exactly_zero(version, signal):  # noqa: D103
    out = _as_array(_tapper(version).feed(signal))
    assert out.shape == (40, 6)
    assert not out.any()


@pytest.mark.parametrize("emotion", EMOTIONS)
@pytest.mark.parametrize(
    "signal",
    [np.zeros(2 * SR, dtype=np.float32), _noise(-70.0)],
    ids=["digital_silence", "noise_-70dB"],
)
def test_v6_silence_gives_exactly_zero_for_every_emotion(emotion, signal):  # noqa: D103
    out = _as_array(_tapper("v6", emotion=emotion, energy=1.5).feed(signal))
    assert out.shape == (40, 6)
    assert not out.any()


@pytest.mark.parametrize("emotion", EMOTIONS)
def test_v6_returns_to_exactly_zero_after_speech(emotion):  # noqa: D103
    """The gate memory (120 ms) and the phrase-final tilt (300 ms) both end."""
    rt = _tapper("v6", emotion=emotion, energy=1.5)
    speech = _as_array(rt.feed(_speech_like(seconds=2.0)))
    assert np.abs(speech).max() > 0.01  # sanity: it did move
    tail = _as_array(rt.feed(_noise(-70.0, seconds=1.5)))
    settle_hops = 600 // speech_tapper_v6.HOP_MS
    assert not tail[settle_hops:].any()


# ---------------------------------------------------------------------------
# v6: clamps, emotion handling, relation to v5
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("emotion", EMOTIONS)
def test_v6_output_stays_within_the_axis_clamps(emotion):  # noqa: D103
    m = speech_tapper_v6
    # Energy is clamped to ENERGY_MAX, asking for more must not get past it.
    rt = _tapper("v6", emotion=emotion, energy=99.0)
    assert rt.energy == m.ENERGY_MAX == 1.5
    out = np.abs(_as_array(rt.feed(_speech_like(amplitude=0.6))))
    limits = np.array(
        [m.SAFE_PITCH_RAD, m.SAFE_YAW_RAD, m.SAFE_ROLL_RAD, m.SAFE_X_MM, m.SAFE_Y_MM, m.SAFE_Z_MM]
    )
    assert out.max() > 0.01
    assert (out.max(axis=0) <= limits).all()
    assert np.isfinite(out).all()


def test_v6_emotion_and_energy_are_optional():  # noqa: D103
    rt = speech_tapper_v6.SwayRollRT(sample_rate=SR)
    assert rt.profile.name == "neutral"
    assert rt.energy == 1.0


def test_v6_unknown_emotion_warns_and_falls_back(caplog):  # noqa: D103
    with caplog.at_level(logging.WARNING, logger="reachy_mini.motion.speech_tapper_v6"):
        rt = _tapper("v6", emotion="flabbergasted")
    assert rt.profile.name == "neutral"
    assert "flabbergasted" in caplog.text

    rt.set_emotion("angry", 1.2)
    assert rt.profile.name == "angry"
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="reachy_mini.motion.speech_tapper_v6"):
        rt.set_emotion("not-an-emotion")
    assert rt.profile.name == "neutral"
    assert rt.energy == 1.2  # energy is kept when omitted
    assert "not-an-emotion" in caplog.text


def test_v6_emotions_differ_from_neutral():  # noqa: D103
    speech = _speech_like(seconds=6.0)
    neutral = _as_array(_tapper("v6").feed(speech))
    for emotion in EMOTIONS:
        if emotion == "neutral":
            continue
        coloured = _as_array(_tapper("v6", emotion=emotion).feed(speech))
        assert np.abs(coloured - neutral).max() > 0.01, emotion


def test_v6_set_emotion_cross_fades_without_a_step():  # noqa: D103
    """Switching mid-phrase must not jump more than a normal hop-to-hop move."""
    speech = _speech_like(seconds=4.0)[: 2 * SR]  # first phrase only, no pause
    half = SR
    steady = _as_array(_tapper("v6").feed(speech))
    rt = _tapper("v6")
    first = rt.feed(speech[:half])
    rt.set_emotion("pleading")
    switched = _as_array(first + rt.feed(speech[half:]))
    k = half // rt.hop
    largest_usual_step = np.abs(np.diff(steady, axis=0)).max(axis=0)
    step_at_switch = np.abs(switched[k] - switched[k - 1])
    assert (step_at_switch <= 1.5 * largest_usual_step + 1e-9).all()


def test_v6_neutral_is_v5_plus_the_two_phrase_layers():  # noqa: D103
    """Roll, x, y, z match v5 exactly; yaw adds the drift, pitch the final tilt."""
    speech = _speech_like()
    v5 = _as_array(speech_tapper_v5.SwayRollRT(sample_rate=SR).feed(speech))
    v6 = _as_array(speech_tapper_v6.SwayRollRT(sample_rate=SR).feed(speech))
    assert v5.shape == v6.shape
    assert np.abs(v5).max() > 0.01

    np.testing.assert_allclose(v6[:, 2:], v5[:, 2:], rtol=0, atol=1e-12)

    neutral = speech_tapper_v6.PROFILES["neutral"]
    drift = np.abs(v6[:, 1] - v5[:, 1])
    assert 0.0 < drift.max() <= math.radians(neutral.drift_yaw_deg)

    final_tilt = np.abs(v6[:, 0] - v5[:, 0])
    assert final_tilt.max() <= math.radians(neutral.final_tilt_deg) + 1e-12
    # The final tilt only plays once the gate has closed, where v5 is at rest.
    tilted = final_tilt > 1e-12
    assert tilted.any()
    assert not v5[tilted].any()


# ---------------------------------------------------------------------------
# HeadWobbler: version and emotion selection
# ---------------------------------------------------------------------------


@pytest.fixture
def wobbler_env(monkeypatch):
    """Start from a clean environment, whatever the developer's shell has."""
    for name in ("WOBBLER_VERSION", "WOBBLER_EMOTION", "WOBBLER_ENERGY"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _make_wobbler():
    from reachy_mini.motion.head_wobbler import HeadWobbler

    return HeadWobbler(lambda offsets: None, sample_rate=SR)


def test_wobbler_defaults_to_the_official_tapper(wobbler_env):  # noqa: D103
    from reachy_mini.motion import speech_tapper

    wobbler = _make_wobbler()
    assert wobbler.version == "v0"
    assert type(wobbler.sway) is speech_tapper.SwayRollRT
    wobbler.set_emotion("angry")  # no emotion input on v0: silently ignored
    wobbler.reset()
    assert type(wobbler.sway) is speech_tapper.SwayRollRT


@pytest.mark.parametrize("version", VERSIONS)
def test_wobbler_version_env_selects_the_tapper(wobbler_env, version):  # noqa: D103
    wobbler_env.setenv("WOBBLER_VERSION", version)
    wobbler = _make_wobbler()
    assert wobbler.version == version
    assert type(wobbler.sway).__module__.endswith(f"speech_tapper_{version}")


def test_wobbler_unknown_version_warns_and_uses_v0(wobbler_env, caplog):  # noqa: D103
    wobbler_env.setenv("WOBBLER_VERSION", "v99")
    with caplog.at_level(logging.WARNING, logger="reachy_mini.motion.head_wobbler"):
        wobbler = _make_wobbler()
    assert wobbler.version == "v0"
    assert "v99" in caplog.text


def test_wobbler_v6_defaults_to_neutral(wobbler_env):  # noqa: D103
    wobbler_env.setenv("WOBBLER_VERSION", "v6")
    wobbler = _make_wobbler()
    assert wobbler.sway.profile.name == "neutral"
    assert wobbler.sway.energy == 1.0


def test_wobbler_v6_reads_emotion_and_energy_env(wobbler_env):  # noqa: D103
    wobbler_env.setenv("WOBBLER_VERSION", "v6")
    wobbler_env.setenv("WOBBLER_EMOTION", "sassy")
    wobbler_env.setenv("WOBBLER_ENERGY", "1.3")
    wobbler = _make_wobbler()
    assert wobbler.sway.profile.name == "sassy"
    assert wobbler.sway.energy == pytest.approx(1.3)


def test_wobbler_v6_bad_env_values_never_raise(wobbler_env, caplog):  # noqa: D103
    wobbler_env.setenv("WOBBLER_VERSION", "v6")
    wobbler_env.setenv("WOBBLER_EMOTION", "flabbergasted")
    wobbler_env.setenv("WOBBLER_ENERGY", "loud")
    with caplog.at_level(logging.WARNING):
        wobbler = _make_wobbler()
    assert wobbler.sway.profile.name == "neutral"
    assert wobbler.sway.energy == 1.0
    assert "flabbergasted" in caplog.text
    assert "loud" in caplog.text


def test_wobbler_set_emotion_survives_reset(wobbler_env, caplog):  # noqa: D103
    wobbler_env.setenv("WOBBLER_VERSION", "v6")
    wobbler = _make_wobbler()
    wobbler.set_emotion("sad", energy=0.8)
    wobbler.reset()  # recreates the tapper
    assert wobbler.sway.profile.name == "sad"
    assert wobbler.sway.energy == pytest.approx(0.8)

    with caplog.at_level(logging.WARNING, logger="reachy_mini.motion.speech_tapper_v6"):
        wobbler.set_emotion("not-an-emotion")
    assert wobbler.sway.profile.name == "neutral"
    assert wobbler.sway.energy == pytest.approx(0.8)
    assert "not-an-emotion" in caplog.text
