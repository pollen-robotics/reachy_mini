"""Speaker -> microphone acoustic loopback, on real hardware.

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

Bring-up notes (verified on robot "Michel", Wireless, 2026-08-28):

- ``PP_ECHOONOFF=0`` demonstrably disables the echo cancellation: with AEC left
  on (``REACHY_TEST_AEC_OFF=""``) the played sound all but vanishes from the
  capture (peak 0.02, snr 0.7, xcorr 0.02 — every gate red).
- ``lag`` in the output is dominated by playbin startup (~450 ms on the CM4);
  TAIL_S exists to keep the sound inside the capture window despite it.
- To re-tune on new hardware: ``REACHY_TEST_DUMP=/tmp/dump.npz`` saves the raw
  windows for offline analysis, ``REACHY_TEST_MIC_VOL`` overrides the pinned
  mic level.
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

# Pass/fail knobs. Deliberately coarse: these catch "no signal" and "wrong
# signal", not small acoustic drift. Values measured on a real Wireless
# (robot "Michel", 2026-08-28), speaker pinned to 100 / mic to 70:
#
# PEAK_MIN: absolute silence floor, the same value the virtual-loopback test
#   uses. Catches a dead speaker or mic outright.
# SNR_MIN: capture RMS over the pre-playback noise RMS — relative, so it
#   survives gain changes. Measured 5.9-12.7 with AEC off vs 0.3-0.7 with AEC
#   on (the AEC cancels the robot's own playback almost completely).
# XCORR_MIN: normalized cross-correlation peak of the reference inside the
#   capture (matched filter). Measured ~0.28 present vs 0.02-0.04 for AEC-on
#   or unrelated noise — an 11x separation. Spectral cosine was tried first
#   and rejected: the acoustic path's coloration left present (0.18) below
#   white noise (0.12) territory.
PEAK_MIN = 1e-3
SNR_MIN = 4.0
XCORR_MIN = 0.1

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
        rms = float(np.sqrt(np.mean(track**2)))
        noise_rms = float(np.sqrt(np.mean(noise[:, channel] ** 2)))
        snr = rms / (noise_rms + 1e-12)
        xcorr, lag_s = correlation_peak(track, reference, rate)

        print(
            f"ch{channel}: peak={peak:.4f} rms={rms:.5f} noise_rms={noise_rms:.5f} "
            f"snr={snr:.1f} xcorr={xcorr:.3f} lag={lag_s * 1000:.0f}ms"
        )

        if peak <= PEAK_MIN:
            failures.append(
                f"ch{channel} is silent (peak={peak:.2e} <= {PEAK_MIN:.0e}): "
                "playback never reached the speaker, or the mic is dead."
            )
        if snr <= SNR_MIN:
            failures.append(
                f"ch{channel} barely rose above its own noise floor "
                f"(snr={snr:.1f} <= {SNR_MIN}): speaker output too quiet, "
                "or echo cancellation is still active."
            )
        if xcorr <= XCORR_MIN:
            failures.append(
                f"ch{channel} does not contain {REFERENCE} "
                f"(xcorr={xcorr:.3f} <= {XCORR_MIN}): something was recorded, "
                "but not what was played."
            )

    assert not failures, "\n".join(failures)
