r"""Speaker -> microphone acoustic loopback, on real hardware.

Plays a known asset on the robot's speaker while recording its own microphone
with board-side echo cancellation disabled, then checks the recording actually
contains the played sound.  Catches a dead speaker, a dead mic, a dead output
channel, and gross breakage of the audio pipeline — none of which any current
test sees.

The head must be **up**: in the sleep pose it sits over the speaker and muffles
it, which reads as a quiet or dead speaker.  The ``head_raised`` fixture enforces
that (it enables torque and moves to the init pose), so the robot can be in any
pose when you start — but it does mean this test **moves the robot** and needs a
running daemon, not just the audio device.

Requires the XVF3800 audio board to be reachable over USB from the machine
running pytest.  That means either a **Lite** (board is on your laptop) or
running this **on a Wireless** robot::

    # Lite: from the repo checkout
    uv run pytest tests/hardware -v -m "audio and respeaker"

    # Wireless: tests/hardware/ is self-contained, so copy it over.
    # uv is preinstalled at /opt/uv on the robot image, and `--with pytest`
    # layers pytest over the daemon venv without installing anything.
    scp -r tests/hardware pollen@reachy-mini.local:/tmp/ht
    ssh pollen@reachy-mini.local \\
      "cd /tmp/ht && VIRTUAL_ENV=/venvs/mini_daemon /opt/uv/uv run --with pytest \\
       pytest -v -m 'audio and respeaker'"

Bring-up notes (verified on robot "Michel", Wireless):

- ``PP_ECHOONOFF=0`` disables the echo cancellation. The negative control
  (``REACHY_TEST_AEC_OFF=""``) is what proves this test can fail at all — run
  it after touching the fixtures, and expect burst ~1 and xcorr < 0.13.
- **A quiet room matters.** Ambient noise raises the noise floor, which deflates
  both gates. In a normally noisy office one healthy run in six dipped below
  the xcorr gate; isolated, healthy runs scored 3-10x clear of it.
- ``lag`` is dominated by playbin startup (~270-460 ms on the CM4); TAIL_S
  exists to keep the sound inside the capture window despite it.
- To re-tune on new hardware: ``REACHY_TEST_DUMP=/tmp/dump.npz`` saves the raw
  windows for offline analysis, ``REACHY_TEST_MIC_VOL`` overrides the pinned
  mic level (100 clips; 70 is the default for that reason).
"""

from __future__ import annotations

import os
import time

import numpy as np
import numpy.typing as npt
import pytest

from reachy_mini.media.audio_control_utils import AudioConfig
from reachy_mini.media.audio_utils import correlation_peak, load_audio_mono
from reachy_mini.media.gstreamer_utils import audio_duration_seconds
from reachy_mini.media.media_manager import MediaManager
from reachy_mini.reachy_mini import ReachyMini
from reachy_mini.utils.constants import ASSETS_ROOT_PATH

# Not wake_up.wav: that one is played by ``wake_up()`` and by the daemon on
# startup, so a stray chime could be what the mic records. count.wav is longer
# (0.66 s vs 0.41 s) and nothing else plays it.
REFERENCE = "count.wav"

# Pass/fail knobs, measured on a real Wireless (robot "Michel") with the
# speaker pinned to 100 / mic to 70, in both a noisy office and an isolated
# quiet room. Three gates, because no single one is sufficient:
#
# PEAK_MIN: absolute silence floor, the same value the virtual-loopback test
#   uses. Catches a dead mic or dead speaker outright. Ambient noise alone
#   clears it, so it discriminates nothing subtler.
# BURST_MIN: RMS of the loudest 50 ms block over the room's noise floor — "did
#   the speaker actually get loud". This is the decisive gate. Measured across
#   8 healthy runs: 19.3-214. With echo cancellation on, 10 runs: 1.0-2.5.
#   Gate 5.0 sits 3.9x under the healthy minimum and 2.0x over the broken
#   maximum.
# XCORR_MIN: matched-filter peak — "was the loud thing the reference".
#   Healthy: 0.228-0.817. Echo-cancelled: 0.024-0.122. Gate 0.15 is roughly the
#   geometric midpoint (1.5x under healthy min, 1.2x over broken max); it was
#   0.2 initially, which sat only 14% under the healthy minimum while leaving
#   64% of headroom over the broken maximum — badly centred, and the wrong way
#   round given BURST_MIN already rejects every broken case seen.
#
# Both of the last two are needed, and this was learned the hard way:
#   - xcorr alone lets a false PASS through. Cross-correlation is scale
#     invariant, and the AEC leaves a faint but perfectly coherent remnant of
#     the playback: one negative control scored 0.122 with its signal 46x too
#     quiet (peak 0.0021) and rms *below* its own noise floor.
#   - a level check alone cannot tell the reference from any other loud noise.
#
# Rejected along the way, both for lack of separation:
#   - spectral_cosine: the acoustic path's coloration put "present" (0.18)
#     below white noise (0.12). Fine on the virtual loopback, useless here.
#   - whole-capture RMS over noise-window RMS: healthy 0.6-5.0 vs broken
#     1.0-9.7, overlapping. The sound is a 0.66 s burst in a ~1.7 s capture, so
#     whole-capture RMS dilutes it ~2.6x and ambient noise dominates. That is
#     exactly what BURST_MIN fixes by isolating the loudest block.
PEAK_MIN = 1e-3
BURST_MIN = 5.0
XCORR_MIN = 0.15

# 50 ms at 16 kHz: long enough for a stable RMS, short enough that a 0.66 s
# burst lands well inside one block.
BLOCK_SAMPLES = 800

MIC_READY_TIMEOUT_S = 2.0
NOISE_WINDOW_S = 0.3
# Covers playbin startup latency (fresh pipeline per play_sound call — can be
# a few hundred ms on a CM4) plus the acoustic tail. Too short clips the sound
# out of the capture window and reads as a low score.
TAIL_S = 1.0


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
    happened to land in it — a chair, a voice, a fan spinning up. Measured on a
    real robot in an office, RMS swung 7x run to run (0.0017-0.0134) and sank
    2 of 6 healthy runs. The median of |x| ignores brief spikes; the 1.2533
    factor (sqrt(pi/2)) converts it back to the RMS of an equivalent Gaussian
    so the ratio against a signal level stays meaningful.
    """
    return float(np.median(np.abs(noise)) * 1.2533)


def _loudest_block_rms(track: npt.NDArray[np.float64], block: int) -> float:
    """RMS of the loudest ``block``-sample window of ``track``.

    The reference is a short burst inside a longer capture, so whole-capture
    RMS dilutes it by the silence around it (~2.6x here) and is dominated by
    ambient noise. Taking the loudest block isolates the sound itself, which is
    what "did the speaker actually produce this" needs to measure.
    """
    usable = len(track) // block * block
    if usable == 0:
        return float(np.sqrt(np.mean(track**2))) if len(track) else 0.0
    blocks = track[:usable].reshape(-1, block)
    return float(np.sqrt((blocks**2).mean(axis=1)).max())


@pytest.mark.audio
@pytest.mark.respeaker
def test_speaker_reaches_microphone(
    aec_disabled: AudioConfig, pinned_volume: None, head_raised: None, mini: ReachyMini
) -> None:
    """The sound played on the speaker is present in the mic recording."""
    media: MediaManager = mini.media
    rate = media.get_input_audio_samplerate()
    reference_path = f"{ASSETS_ROOT_PATH}/{REFERENCE}"
    duration = audio_duration_seconds(reference_path)

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
        media.play_sound(REFERENCE)
        played = _capture(media, duration + TAIL_S)
    finally:
        media.stop_recording()

    assert len(noise), "No audio captured for the noise-floor window."
    assert len(played), "No audio captured while playing."

    reference, _ = load_audio_mono(reference_path, samplerate=rate)

    # Bring-up/debug: dump the raw windows for offline metric analysis.
    dump = os.environ.get("REACHY_TEST_DUMP")
    if dump:
        np.savez(dump, noise=noise, played=played, reference=reference, rate=rate)
        print(f"\ndumped capture to {dump}")

    print(
        f"\nAEC config applied: {list(aec_disabled) or '(none — AEC left enabled)'}"
        f"\ncapture: {len(played)} frames @ {rate} Hz, {played.shape[1]} channels"
    )

    # Per channel, so one dead output path can't hide behind the other. Note the
    # XVF3800's USB stream is 2 *processed* channels, not the 4 raw mics — this
    # catches a dead path, not one dead mic in the array.
    failures: list[str] = []
    for channel in range(played.shape[1]):
        track = played[:, channel]
        peak = float(np.abs(track).max())
        noise_floor = _noise_floor(noise[:, channel])
        burst_rms = _loudest_block_rms(track, BLOCK_SAMPLES)
        burst = burst_rms / (noise_floor + 1e-12)
        xcorr, lag_s = correlation_peak(track, reference, rate)

        print(
            f"ch{channel}: peak={peak:.4f} noise_floor={noise_floor:.5f} "
            f"burst={burst:.1f} xcorr={xcorr:.3f} lag={lag_s * 1000:.0f}ms"
        )

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
        if xcorr <= XCORR_MIN:
            failures.append(
                f"ch{channel} does not contain {REFERENCE} "
                f"(xcorr={xcorr:.3f} <= {XCORR_MIN}): something loud happened, "
                "but it was not what was played."
            )

    assert not failures, "\n".join(failures)
