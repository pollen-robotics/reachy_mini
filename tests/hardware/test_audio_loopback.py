r"""Speaker -> microphone acoustic check, on real hardware.

One test, one sweep: play a generated log sweep on the robot's speaker while
recording its own microphone with board-side echo cancellation disabled, then
gate on three things —

- **peak**: anything was recorded at all (dead mic / dead speaker),
- **burst**: the loudest moment rose well above the room's noise floor
  (speaker actually produced sound; echo cancellation not eating it),
- **curve**: the per-band frequency response matches a stored baseline
  (the speaker tuning — driver, shell, EQ config — hasn't drifted).

The sweep is generated (nothing committed, nothing another feature also plays),
and its capture yields the response curve for free, so the previous separate
count.wav presence test is folded into this one.

The head must be **up**: in the sleep pose it sits over the speaker and muffles
it. The ``head_raised`` fixture enforces that (silently — ``wake_up()`` would
play a chime through the device under test), so this test **moves the robot**
and needs a running daemon.

Requires the XVF3800 audio board on USB from the machine running pytest:
either a **Lite** (board on your laptop) or running **on a Wireless** robot.
Both run from a repo checkout (CI git-clones the repo on the robot), so the
markers come from pyproject.toml as usual::

    # Lite: from the repo checkout
    uv run pytest tests/hardware -v -m "audio and respeaker"

    # Wireless: clone on the robot, then layer pytest over the daemon venv.
    # --no-project stops uv from resolving the cloned pyproject into a fresh
    # env on the CM4; uv is preinstalled at /opt/uv on the robot image.
    ssh pollen@reachy-mini.local "
      git clone --depth 1 -b <branch> \\
          https://github.com/pollen-robotics/reachy_mini.git /tmp/rm &&
      cd /tmp/rm && VIRTUAL_ENV=/venvs/mini_daemon /opt/uv/uv run --no-project \\
          --with pytest pytest tests/hardware -v -m 'audio and respeaker'"

The curve baseline is shared: all Reachy Minis have the same speaker, shell and
mic, so one measured reference (robot "Michel", Wireless, isolated room,
median of 3 sweeps) is baked in as ``BASELINE_DB``. Every run prints the
measured curve — to re-baseline (new hardware revision, or the assumption
proves too tight), copy the printed numbers into the constant.

Bring-up notes (verified on robot "Michel", Wireless):

- ``PP_ECHOONOFF=0`` disables the echo cancellation. The negative control
  (``REACHY_TEST_AEC_OFF=""``) is what proves this test can fail at all — run
  it after touching the fixtures. With a sweep it fails on the CURVE gate
  (4-17 dB of drift), not the burst gate: the adaptive AEC cancels a sweep far
  less well than speech.
- **A quiet room matters.** Ambient noise raises the noise floor and pollutes
  the curve; the figures here are from an isolated room.
- Single sweeps swing up to ~5 dB at 947 Hz (room reflections + the post-EQ
  limiter on a boosted band) — one healthy run in four tripped a single-sweep
  gate. Hence the median of N_SWEEPS and the per-band CURVE_TOL_DB.
- The measured curve does NOT equal the nominal EQ gains: the original EQ was
  calibrated with an external mic, and this path adds the robot's own mic DSP
  and the post-EQ limiter. That is why the baseline is measured, not computed.
- ``lag`` is dominated by playbin startup (~270-460 ms on the CM4); TAIL_S
  keeps the sweep inside the capture window despite it.
- ``REACHY_TEST_ARTIFACTS=<dir>`` writes report artifacts: the full session
  audio (``audio.wav``), every gate's numbers (``curve.json``), and the raw
  windows (``raw.npz``). Render figures offline with
  ``tests/hardware/plot_audio_report.py`` (matplotlib is not on the robot).
- ``REACHY_TEST_MIC_VOL`` overrides the pinned mic level (100 clips;
  70 is the default for that reason).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from reachy_mini.media.audio_control_utils import AudioConfig
from reachy_mini.media.audio_utils import correlation_peak, save_audio_to_wav
from reachy_mini.media.media_manager import MediaManager
from reachy_mini.reachy_mini import ReachyMini
from reachy_mini.tools.speaker_eq_calibration.calibrate import (
    BAND_CENTERS,
    bin_to_bands,
)

# The excitation: a log sweep. Ideal for the matched filter (sharp, unambiguous
# correlation peak) and covers every band the 16 kHz voice path can carry.
RATE = 16000
SWEEP_S = 2.0
SWEEP_F0, SWEEP_F1 = 100.0, 7800.0
SWEEP_AMPLITUDE = 0.7  # mic clips before the sweep does at full scale

# Bands 119-3770 Hz: what the small driver and the 16 kHz path can measure.
# Below, the driver produces nothing; above, the top band exceeds Nyquist.
USABLE_BANDS = slice(2, 8)

# Shared reference curve for the USABLE_BANDS, dB relative to the bands'
# median. Measured on robot "Michel" (Wireless, isolated room, median of 3
# sweeps, 2026-08-31). All Reachy Minis share the same speaker/shell/mic, so
# one baseline serves them all — the first run on a *different* robot is the
# test of that assumption; if a known-good robot fails the curve gate, widen
# CURVE_TOL_DB or re-baseline from the printed curve.
BASELINE_DB = (19.9, 4.1, -1.3, -4.6, -8.6, 1.3)

# Pass/fail knobs, measured on a real Wireless (robot "Michel"), speaker
# pinned to 100 / mic to 70, isolated room. See the module docstring for the
# measurement campaign behind each number.
#
# PEAK_MIN: absolute silence floor. Dead mic / dead speaker.
# BURST_MIN: RMS of the loudest 50 ms block over a transient-resistant noise
#   floor — a pure level gate (dead/muted speaker). Healthy sweeps measured
#   137-267. Note the adaptive AEC cancels a sweep far less well than speech
#   (burst 53 with AEC on), so unlike the earlier count.wav version this gate
#   does NOT catch AEC-still-on — the curve gate does, at 4-17 dB of drift.
# CURVE_TOL_DB: per-band |median measured - baseline|. Single sweeps swing up
#   to ~5 dB at 947 Hz; the median of N_SWEEPS tames that, and a real tuning
#   change (EQ zeroed) reliably blows out several bands by 9-16 dB.
#   Two bands are NOT gated at 4 dB:
#   - 119 Hz (inf = report-only): bass is room-mode dominated — measured
#     drifting monotonically to -7.4 dB over ~an hour as the room changed,
#     while every other band stayed within 3.3 dB. It cannot hold a fixed
#     baseline across rooms or robots.
#   - 3770 Hz (8 dB): spiked +5.8 dB once on a healthy run (intermittent,
#     suspected head-servo whine in that range).
#   Excluding them, measured separation: healthy <= 3.3 dB vs broken >= 9.1 dB.
PEAK_MIN = 1e-3
BURST_MIN = 5.0
CURVE_TOL_DB = (float("inf"), 4.0, 4.0, 4.0, 4.0, 8.0)
N_SWEEPS = 3

# 50 ms at 16 kHz: long enough for a stable RMS, short enough to isolate the
# sweep from the silence around it.
BLOCK_SAMPLES = 800

MIC_READY_TIMEOUT_S = 2.0
NOISE_WINDOW_S = 0.3
# Covers playbin startup latency (fresh pipeline per play_sound call — a few
# hundred ms on a CM4) plus the acoustic tail.
TAIL_S = 1.0


def _make_sweep() -> npt.NDArray[np.float64]:
    """Log sweep with 10 ms raised-cosine fades (no click)."""
    t = np.arange(int(SWEEP_S * RATE)) / RATE
    ratio = SWEEP_F1 / SWEEP_F0
    phase = (
        2 * np.pi * SWEEP_F0 * SWEEP_S / np.log(ratio) * (ratio ** (t / SWEEP_S) - 1)
    )
    sweep = SWEEP_AMPLITUDE * np.sin(phase)
    fade = int(0.01 * RATE)
    window = np.hanning(2 * fade)
    sweep[:fade] *= window[:fade]
    sweep[-fade:] *= window[fade:]
    return sweep


SWEEP = _make_sweep()


def _capture(media: MediaManager, seconds: float) -> npt.NDArray[np.float64]:
    """Poll the mic for ``seconds``, returning an ``(N, channels)`` array."""
    samples: list[npt.NDArray[np.float32]] = []
    deadline = time.time() + seconds
    while time.time() < deadline:
        sample = media.get_audio_sample()
        if sample is not None:
            samples.append(sample)
        else:
            # Yield rather than spin: on the CM4 a busy loop here competes with
            # the GStreamer capture thread we're trying to read from.
            time.sleep(0.002)
    if not samples:
        return np.empty((0, 0), dtype=np.float64)
    return np.concatenate(samples, axis=0).astype(np.float64)


def _noise_floor(noise: npt.NDArray[np.float64]) -> float:
    """Transient-resistant noise level, as a Gaussian-equivalent RMS.

    Plain RMS over the short noise window is dominated by whatever transient
    happened to land in it — a chair, a voice, a fan. The median of |x|
    ignores brief spikes; the 1.2533 factor (sqrt(pi/2)) converts it back to
    the RMS of an equivalent Gaussian so ratios stay meaningful.
    """
    return float(np.median(np.abs(noise)) * 1.2533)


def _loudest_block_rms(track: npt.NDArray[np.float64], block: int) -> float:
    """RMS of the loudest ``block``-sample window of ``track``."""
    usable = len(track) // block * block
    if usable == 0:
        return float(np.sqrt(np.mean(track**2))) if len(track) else 0.0
    blocks = track[:usable].reshape(-1, block)
    return float(np.sqrt((blocks**2).mean(axis=1)).max())


def _band_response(segment: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Capture-vs-sweep dB difference in the 10 EQ bands, level-normalized.

    Dividing by the sweep's own spectrum cancels the excitation shape;
    subtracting the median over the usable bands cancels overall level (volume,
    mic gain, distance), leaving only the curve's *shape*.
    """
    n = 1 << (len(SWEEP) - 1).bit_length()
    cap = np.abs(np.fft.rfft(segment * np.hanning(len(segment)), n))
    ref = np.abs(np.fft.rfft(SWEEP * np.hanning(len(SWEEP)), n))
    freqs = np.fft.rfftfreq(n, 1 / RATE)
    db = 20 * np.log10((cap + 1e-9) / (ref + 1e-9))
    bands = np.asarray(bin_to_bands(freqs, db))
    return bands - np.median(bands[USABLE_BANDS])


@pytest.mark.audio
@pytest.mark.respeaker
def test_speaker_mic_acoustic_path(
    aec_disabled: AudioConfig,
    pinned_volume: None,
    head_raised: None,
    mini: ReachyMini,
    tmp_path: Path,
) -> None:
    """The speaker is audible, and its frequency response matches the baseline."""
    media: MediaManager = mini.media
    rate = media.get_input_audio_samplerate()
    assert rate == RATE, f"capture rate {rate} != {RATE}; sweep constants assume 16 kHz"

    sweep_wav = str(tmp_path / "sweep.wav")
    save_audio_to_wav(SWEEP.astype(np.float32), RATE, sweep_wav)

    # One recording session, N_SWEEPS play/capture cycles. The per-sweep curves
    # are combined by median: the 947 Hz band swings up to ~5 dB between single
    # sweeps (room reflections + the post-EQ limiter on a boosted band), which
    # made a single-sweep gate flaky at any tolerance still worth having.
    sweeps: list[npt.NDArray[np.float64]] = []
    media.start_recording()
    try:
        deadline = time.time() + MIC_READY_TIMEOUT_S
        while media.get_audio_sample() is None:
            if time.time() > deadline:
                pytest.fail(
                    f"Microphone produced no samples within {MIC_READY_TIMEOUT_S}s."
                )
            time.sleep(0.005)

        noise = _capture(media, NOISE_WINDOW_S)
        for _ in range(N_SWEEPS):
            media.play_sound(sweep_wav)
            sweeps.append(_capture(media, SWEEP_S + TAIL_S))
    finally:
        media.stop_recording()

    assert len(noise), "No audio captured for the noise-floor window."
    assert all(len(s) for s in sweeps), "No audio captured while playing."
    played = sweeps[0]

    print(
        f"\nAEC config applied: {list(aec_disabled) or '(none — AEC left enabled)'}"
        f"\ncapture: {len(played)} frames @ {rate} Hz, {played.shape[1]} channels"
    )

    # Level gates per channel, so one dead output path can't hide behind the
    # other. (The XVF3800's USB stream is 2 *processed* channels, not the 4 raw
    # mics — this catches a dead path, not one dead mic in the array.)
    failures: list[str] = []
    peaks: list[float] = []
    bursts: list[float] = []
    for channel in range(played.shape[1]):
        track = played[:, channel]
        peak = float(np.abs(track).max())
        burst = _loudest_block_rms(track, BLOCK_SAMPLES) / (
            _noise_floor(noise[:, channel]) + 1e-12
        )
        peaks.append(peak)
        bursts.append(burst)
        print(f"ch{channel}: peak={peak:.4f} burst={burst:.1f}")

        if peak <= PEAK_MIN:
            failures.append(
                f"ch{channel} is silent (peak={peak:.2e} <= {PEAK_MIN:.0e}): "
                "playback never reached the speaker, or the mic is dead."
            )
        if burst <= BURST_MIN:
            failures.append(
                f"ch{channel} never got loud (burst={burst:.1f} <= {BURST_MIN}): "
                "the loudest moment barely rose above the room's noise floor, so "
                "the speaker is too quiet or its output is being cancelled."
            )

    # Frequency-response curve, on ch0 (the two channels carry the same
    # processed beam). Locate each sweep with the matched filter, compute its
    # band response, then gate the per-band MEDIAN against the baseline.
    curves = []
    xcorrs: list[float] = []
    lags_ms: list[float] = []
    for captured in sweeps:
        track = captured[:, 0]
        xcorr, lag_s = correlation_peak(track, SWEEP, rate)
        start = int(lag_s * rate)
        segment = track[start : start + len(SWEEP)]
        if len(segment) < len(SWEEP):
            segment = np.pad(segment, (0, len(SWEEP) - len(segment)))
        curves.append(_band_response(segment)[USABLE_BANDS])
        xcorrs.append(xcorr)
        lags_ms.append(lag_s * 1000)
        print(f"sweep: xcorr={xcorr:.3f} lag={lag_s * 1000:.0f}ms")

    centers = [int(c) for c in np.asarray(BAND_CENTERS)[USABLE_BANDS]]
    curve = np.median(np.asarray(curves), axis=0)
    drift = curve - np.asarray(BASELINE_DB)
    print("band centers Hz:", " ".join(f"{c:5d}" for c in centers))
    print("curve dB       :", " ".join(f"{v:+5.1f}" for v in curve))
    print("drift vs base  :", " ".join(f"{v:+5.1f}" for v in drift))

    for center, value, tolerance in zip(centers, drift, CURVE_TOL_DB):
        if abs(value) > tolerance:
            failures.append(
                f"speaker response drifted {value:+.1f} dB at {center} Hz "
                f"(tolerance ±{tolerance}): the speaker tuning changed "
                "(driver, shell, EQ config), or this robot/room deviates from "
                "the shared baseline."
            )

    # Report artifacts (REACHY_TEST_ARTIFACTS=<dir>): everything a CI report
    # needs — the full session audio, the numbers behind every gate, and the
    # raw windows. Written before the assert so failures keep their evidence.
    # Plot offline with tests/hardware/plot_audio_report.py (matplotlib is not
    # installed on the robot).
    artifacts = os.environ.get("REACHY_TEST_ARTIFACTS")
    if artifacts:
        out = Path(artifacts)
        out.mkdir(parents=True, exist_ok=True)
        session = np.concatenate([noise, *sweeps], axis=0).astype(np.float32)
        save_audio_to_wav(session, rate, str(out / "audio.wav"))
        (out / "curve.json").write_text(
            json.dumps(
                {
                    "band_centers_hz": centers,
                    "curve_db": [float(v) for v in curve],
                    "baseline_db": list(BASELINE_DB),
                    "drift_db": [float(v) for v in drift],
                    "tolerance_db": [
                        t if np.isfinite(t) else None for t in CURVE_TOL_DB
                    ],
                    "sweep_curves_db": [[float(v) for v in c] for c in curves],
                    "xcorr": xcorrs,
                    "lag_ms": lags_ms,
                    "peak": peaks,
                    "burst": bursts,
                    "gates": {"peak_min": PEAK_MIN, "burst_min": BURST_MIN},
                    "rate_hz": rate,
                    "failures": failures,
                },
                indent=1,
            )
        )
        np.savez(
            out / "raw.npz", noise=noise, played=played, reference=SWEEP, rate=rate
        )
        print(f"artifacts written to {out}")

    assert not failures, "\n".join(failures)
