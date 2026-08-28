"""Unit tests for the audio analysis helpers used by the hardware loopback test.

No robot and no audio device — but ``load_audio_mono`` decodes through
GStreamer, same as the rest of this module.  These exist so that
``spectral_cosine`` / ``load_audio_mono`` are known-good before a hardware
failure gets blamed on the hardware.
"""

import numpy as np
import pytest

from reachy_mini.media.audio_utils import (
    correlation_peak,
    load_audio_mono,
    save_audio_to_wav,
    spectral_cosine,
)
from reachy_mini.utils.constants import ASSETS_ROOT_PATH

RATE = 16000
SRC_RATE = 44100  # every shipped asset
REFERENCE_PATH = f"{ASSETS_ROOT_PATH}/wake_up.wav"


def _reference() -> np.ndarray:
    """Load the same reference signal the hardware test plays, at capture rate."""
    samples, _ = load_audio_mono(REFERENCE_PATH, samplerate=RATE)
    return samples


def _sweep(duration: float = 0.5, rate: int = RATE) -> np.ndarray:
    """Log sweep 200 Hz -> 6 kHz."""
    t = np.arange(int(duration * rate)) / rate
    f0, f1 = 200.0, 6000.0
    phase = (
        2 * np.pi * f0 * duration / np.log(f1 / f0) * ((f1 / f0) ** (t / duration) - 1)
    )
    return np.sin(phase)


def test_correlation_peak_finds_delayed_copy() -> None:
    """A delayed, attenuated, noisy copy is found at the right offset.

    This is the hardware loopback case: the mic hears the reference late
    (playbin startup), quieter (acoustic path), with noise on top — and the
    matched filter must report both presence and the correct lag.
    """
    rng = np.random.default_rng(0)
    reference = _reference()
    delay = 4000  # samples = 250 ms at 16 kHz
    capture = np.concatenate(
        [np.zeros(delay), 0.05 * reference, np.zeros(2000)]
    ) + 1e-3 * rng.standard_normal(len(reference) + delay + 2000)

    peak, lag_s = correlation_peak(capture, reference, RATE)

    assert peak > 0.5
    assert lag_s == pytest.approx(delay / RATE, abs=0.01)


def test_correlation_peak_rejects_unrelated_noise() -> None:
    """Noise without the reference scores near zero.

    Measured on-robot: ~0.28 with the sound present vs 0.02-0.04 without.
    This is the separation the hardware test's XCORR_MIN gate relies on.
    """
    rng = np.random.default_rng(1)
    reference = _reference()
    noise = rng.standard_normal(len(reference) + 6000)

    peak, _ = correlation_peak(noise, reference, RATE)

    assert peak < 0.1


def test_spectral_cosine_identical_is_one() -> None:
    """A signal against itself scores 1."""
    signal = _reference()
    assert spectral_cosine(signal, signal) == pytest.approx(1.0, abs=1e-6)


def test_spectral_cosine_survives_delay_and_attenuation() -> None:
    """A delayed, quiet, noisy copy still scores high — the loopback case.

    This is the property the hardware test depends on: the mic hears the sound
    late, quieter, and with noise on top, and the metric must not care.
    """
    rng = np.random.default_rng(0)
    signal = _reference()
    degraded = np.concatenate(
        [np.zeros(1000), 0.05 * signal, np.zeros(500)]
    ) + 1e-3 * rng.standard_normal(len(signal) + 1500)

    assert spectral_cosine(degraded, signal) > 0.5


def test_spectral_cosine_rejects_unrelated_noise() -> None:
    """Noise against the reference scores low, so the threshold discriminates.

    This is what makes the hardware test's 0.5 gate meaningful: "recorded
    something" and "recorded the right thing" land on opposite sides of it.
    """
    rng = np.random.default_rng(1)
    signal = _reference()
    noise = rng.standard_normal(len(signal))

    assert spectral_cosine(noise, signal) < 0.3


def test_spectral_cosine_cannot_discriminate_broadband_references() -> None:
    """A log sweep is a *bad* reference for this metric — pinned deliberately.

    ``spectral_cosine`` compares magnitude spectra, so it only separates signal
    from noise when the reference has spectral structure (as speech does).  A
    sweep spreads its energy evenly, which looks much like white noise: the
    score here is ~0.5, i.e. right at the hardware test's pass threshold, for
    two completely unrelated signals.

    So if a sweep is ever introduced as the reference (for latency via
    cross-correlation, say), the *correlation peak* must be the discriminator —
    not this function.
    """
    rng = np.random.default_rng(2)
    sweep = _sweep()
    noise = rng.standard_normal(len(sweep))

    assert spectral_cosine(noise, sweep) > 0.4


@pytest.mark.parametrize("asset", ["wake_up.wav", "go_sleep.wav"])
def test_load_audio_mono_decodes_assets(asset: str) -> None:
    """Shipped assets decode to normalised mono at their own rate.

    wake_up is 16-bit, go_sleep is 24-bit — GStreamer handles the bit depth, so
    both take the same code path here.
    """
    samples, rate = load_audio_mono(f"{ASSETS_ROOT_PATH}/{asset}")

    assert samples.ndim == 1
    assert rate == SRC_RATE
    assert np.abs(samples).max() <= 1.0
    # Real audio, not a run of zeros or a DC offset.
    assert np.abs(samples).max() > 0.01


def test_load_audio_mono_resamples() -> None:
    """Requesting a rate resamples and reports the rate actually delivered."""
    native, native_rate = load_audio_mono(REFERENCE_PATH)
    resampled, rate = load_audio_mono(REFERENCE_PATH, samplerate=RATE)

    assert native_rate == SRC_RATE
    assert rate == RATE
    assert len(resampled) == pytest.approx(len(native) * RATE / SRC_RATE, rel=0.01)
    # Still real audio after the rate change, not silence.
    assert np.abs(resampled).max() > 0.01


def test_spectral_cosine_requires_a_common_sample_rate() -> None:
    """Comparing across sample rates is meaningless — pinned deliberately.

    ``spectral_cosine`` uses a fixed FFT size, so a given frequency lands in a
    different bin at a different rate.  The *same* audio at 44.1 kHz and 16 kHz
    scores near zero, which is a trap worth failing loudly on rather than
    misreading as "the mic recorded the wrong thing".
    """
    native, _ = load_audio_mono(REFERENCE_PATH)
    resampled, _ = load_audio_mono(REFERENCE_PATH, samplerate=RATE)

    assert spectral_cosine(resampled, native) < 0.1
    # ... and matched rates score ~1 on the very same audio.
    assert spectral_cosine(resampled, resampled) == pytest.approx(1.0, abs=1e-6)


def test_save_load_roundtrip(tmp_path) -> None:
    """save_audio_to_wav -> load_audio_mono returns the original signal.

    The hand-rolled ``wave``-based reader this replaced could not do this at all:
    save_audio_to_wav emits IEEE-float WAV, which stdlib ``wave`` rejects.
    """
    signal = (0.5 * _sweep()).astype(np.float32)
    path = str(tmp_path / "sweep.wav")
    save_audio_to_wav(signal, samplerate=RATE, filepath=path)

    loaded, rate = load_audio_mono(path)

    assert rate == RATE
    assert len(loaded) == len(signal)
    assert np.abs(loaded - signal).max() < 1e-6


def test_load_audio_mono_downmixes_to_mono(tmp_path) -> None:
    """Stereo in, mono out — averaged, not interleaved into a longer track."""
    stereo = np.zeros((256, 2), dtype=np.float32)
    stereo[:, 0] = 0.5
    stereo[:, 1] = -0.5
    path = str(tmp_path / "stereo.wav")
    save_audio_to_wav(stereo, samplerate=RATE, filepath=path)

    loaded, _ = load_audio_mono(path)

    assert loaded.ndim == 1
    assert len(loaded) == 256
    assert np.abs(loaded).max() < 1e-6  # +0.5 and -0.5 average to 0


def test_load_audio_mono_rejects_missing_file(tmp_path) -> None:
    """A missing file raises with the cause, not a silent empty array."""
    with pytest.raises(RuntimeError, match="load_audio_mono"):
        load_audio_mono(str(tmp_path / "nope.wav"))
