"""Unit tests for speech_tapper and head_wobbler modules."""  # noqa: D100

import time

import numpy as np
import pytest

from reachy_mini.motion.speech_tapper import (
    HOP_MS,
    SR,
    SwayRollRT,
    _loudness_gain,
    _resample_linear,
    _rms_dbfs,
    _to_float32_mono,
)

# ---------------------------------------------------------------------------
# speech_tapper: helper functions
# ---------------------------------------------------------------------------


def test_rms_silence_is_very_negative():  # noqa: D103
    silence = np.zeros(320, dtype=np.float32)
    assert _rms_dbfs(silence) < -100


def test_rms_full_scale_sine_near_zero():  # noqa: D103
    t = np.linspace(0, 1, SR, dtype=np.float32)
    sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    db = _rms_dbfs(sine)
    assert -5 < db < 0  # RMS of sine ≈ -3 dBFS


def test_rms_quiet_signal_is_negative():  # noqa: D103
    t = np.linspace(0, 1, SR, dtype=np.float32)
    quiet = (np.sin(2 * np.pi * 440 * t) * 0.01).astype(np.float32)
    assert _rms_dbfs(quiet) < -35


def test_loudness_below_low_threshold_is_zero():  # noqa: D103
    assert _loudness_gain(-100.0) == 0.0


def test_loudness_above_high_threshold_clamped():  # noqa: D103
    gain = _loudness_gain(0.0)
    assert gain <= 1.0
    assert gain > 0.9


def test_loudness_monotonically_increasing():  # noqa: D103
    dbs = [-50, -40, -30, -20, -10]
    gains = [_loudness_gain(db) for db in dbs]
    for i in range(len(gains) - 1):
        assert gains[i] <= gains[i + 1]


def test_to_float32_mono_int16():  # noqa: D103
    pcm = np.array([0, 16384, -16384], dtype=np.int16)
    result = _to_float32_mono(pcm)
    assert result.dtype == np.float32
    assert abs(result[0]) < 1e-6
    assert 0.4 < result[1] < 0.6


def test_to_float32_mono_stereo():  # noqa: D103
    stereo = np.ones((2, 100), dtype=np.float32)
    mono = _to_float32_mono(stereo)
    assert mono.ndim == 1
    assert len(mono) == 100


def test_to_float32_mono_passthrough():  # noqa: D103
    mono = np.ones(100, dtype=np.float32) * 0.5
    result = _to_float32_mono(mono)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, 0.5)


def test_to_float32_mono_empty():  # noqa: D103
    result = _to_float32_mono(np.array(0))
    assert result.size == 0


def test_resample_same_rate_passthrough():  # noqa: D103
    x = np.ones(100, dtype=np.float32)
    result = _resample_linear(x, 16000, 16000)
    assert len(result) == 100


def test_resample_downsample():  # noqa: D103
    x = np.ones(1600, dtype=np.float32)
    result = _resample_linear(x, 48000, 16000)
    assert abs(len(result) - 533) <= 1


def test_resample_upsample():  # noqa: D103
    x = np.ones(160, dtype=np.float32)
    result = _resample_linear(x, 8000, 16000)
    assert abs(len(result) - 320) <= 1


# ---------------------------------------------------------------------------
# speech_tapper: SwayRollRT
# ---------------------------------------------------------------------------


def test_sway_empty_input():  # noqa: D103
    rt = SwayRollRT()
    assert rt.feed(np.zeros(0, dtype=np.float32), 16000) == []


def test_sway_short_input_no_output():  # noqa: D103
    """Input shorter than one hop produces no output."""
    rt = SwayRollRT()
    short = np.zeros(100, dtype=np.float32)
    assert rt.feed(short, 16000) == []


def test_sway_one_second_produces_hops():  # noqa: D103
    rt = SwayRollRT()
    t = np.linspace(0, 1, SR, dtype=np.float32)
    tone = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    results = rt.feed(tone, SR)
    expected_hops = 1000 // HOP_MS
    assert len(results) == expected_hops


def test_sway_output_keys():  # noqa: D103
    rt = SwayRollRT()
    t = np.linspace(0, 0.1, int(SR * 0.1), dtype=np.float32)
    tone = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    results = rt.feed(tone, SR)
    assert len(results) >= 1
    expected_keys = {"pitch_rad", "yaw_rad", "roll_rad", "x_mm", "y_mm", "z_mm"}
    assert expected_keys <= set(results[0].keys())


def test_sway_silence_produces_near_zero():  # noqa: D103
    rt = SwayRollRT()
    silence = np.zeros(SR, dtype=np.float32)
    results = rt.feed(silence, SR)
    for r in results:
        assert abs(r["pitch_rad"]) < 0.01
        assert abs(r["yaw_rad"]) < 0.01
        assert abs(r["x_mm"]) < 0.1


def test_sway_loud_signal_produces_nonzero():  # noqa: D103
    rt = SwayRollRT()
    t = np.linspace(0, 3, SR * 3, dtype=np.float32)
    tone = (np.sin(2 * np.pi * 300 * t) * 0.8).astype(np.float32)
    results = rt.feed(tone, SR)
    max_yaw = max(abs(r["yaw_rad"]) for r in results)
    assert max_yaw > 0.01


def test_sway_resampling():  # noqa: D103
    rt = SwayRollRT()
    sr_in = 48000
    t = np.linspace(0, 1, sr_in, dtype=np.float32)
    tone = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    results = rt.feed(tone, sr_in)
    expected_hops = 1000 // HOP_MS
    assert abs(len(results) - expected_hops) <= 1


def test_sway_int16_input():  # noqa: D103
    rt = SwayRollRT()
    t = np.linspace(0, 1, SR, dtype=np.float32)
    tone_i16 = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
    results = rt.feed(tone_i16, SR)
    assert len(results) == 1000 // HOP_MS


def test_sway_reset_clears_state():  # noqa: D103
    rt = SwayRollRT()
    t = np.linspace(0, 1, SR, dtype=np.float32)
    tone = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    rt.feed(tone, SR)
    rt.reset()
    assert rt.t == 0.0
    assert rt.vad_on is False
    assert rt.carry.size == 0


def test_sway_deterministic_with_same_seed():  # noqa: D103
    t = np.linspace(0, 1, SR, dtype=np.float32)
    tone = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)

    rt1 = SwayRollRT(rng_seed=42)
    r1 = rt1.feed(tone.copy(), SR)

    rt2 = SwayRollRT(rng_seed=42)
    r2 = rt2.feed(tone.copy(), SR)

    assert len(r1) == len(r2)
    for a, b in zip(r1, r2):
        assert a == pytest.approx(b)


def test_sway_incremental_feeding():  # noqa: D103
    """Feeding small chunks should produce same total hops as one big chunk."""
    rt_batch = SwayRollRT(rng_seed=7)
    t = np.linspace(0, 1, SR, dtype=np.float32)
    tone = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    results_batch = rt_batch.feed(tone, SR)

    rt_inc = SwayRollRT(rng_seed=7)
    results_inc = []
    chunk_size = 1600  # 100ms chunks
    for i in range(0, len(tone), chunk_size):
        results_inc.extend(rt_inc.feed(tone[i : i + chunk_size], SR))

    assert len(results_inc) == len(results_batch)


# ---------------------------------------------------------------------------
# head_wobbler: HeadWobbler
# ---------------------------------------------------------------------------


def test_wobbler_callback_receives_offsets():  # noqa: D103
    from reachy_mini.motion.head_wobbler import HeadWobbler

    received: list[tuple[float, ...]] = []
    wobbler = HeadWobbler(lambda o: received.append(o))
    wobbler.start()

    t = np.linspace(0, 2, SR * 2, dtype=np.float32)
    tone = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    chunk_size = 1600
    for i in range(0, len(tone), chunk_size):
        wobbler.feed_pcm(tone[i : i + chunk_size], SR)

    time.sleep(3)
    wobbler.stop()

    assert len(received) > 0


def test_wobbler_offsets_are_6_tuples():  # noqa: D103
    from reachy_mini.motion.head_wobbler import HeadWobbler

    received: list[tuple[float, ...]] = []
    wobbler = HeadWobbler(lambda o: received.append(o))
    wobbler.start()

    t = np.linspace(0, 1, SR, dtype=np.float32)
    tone = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    wobbler.feed_pcm(tone, SR)

    time.sleep(2)
    wobbler.stop()

    assert len(received) > 0
    for offsets in received:
        assert len(offsets) == 6
        assert all(isinstance(v, float) for v in offsets)


def test_wobbler_stop_resets_to_zero():  # noqa: D103
    from reachy_mini.motion.head_wobbler import HeadWobbler

    received: list[tuple[float, ...]] = []
    wobbler = HeadWobbler(lambda o: received.append(o))
    wobbler.start()
    wobbler.stop()

    assert received[-1] == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_wobbler_reset_drains_queue():  # noqa: D103
    from reachy_mini.motion.head_wobbler import HeadWobbler

    received: list[tuple[float, ...]] = []
    wobbler = HeadWobbler(lambda o: received.append(o))
    wobbler.start()

    tone = (np.sin(np.linspace(0, 1, SR, dtype=np.float32) * 2 * np.pi * 440) * 0.5).astype(np.float32)
    wobbler.feed_pcm(tone, SR)

    wobbler.reset()

    assert wobbler.audio_queue.empty()
    assert received[-1] == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    wobbler.stop()


def test_wobbler_start_is_idempotent():  # noqa: D103
    from reachy_mini.motion.head_wobbler import HeadWobbler

    wobbler = HeadWobbler(lambda o: None)
    wobbler.start()
    wobbler.start()  # should not crash or create duplicate threads
    wobbler.stop()
