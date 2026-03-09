"""Reachy Mini sound playback example.

Two modes:
  --wav <path>  Play a wav file using the media play_sound API.
  --live        Push a continuous sine tone using push_audio_sample.
"""

# START doc_example

import argparse
import logging
import os
import time

import numpy as np

from reachy_mini import ReachyMini


def play_wav(mini: "ReachyMini", wav_path: str) -> None:
    """Play a wav file using the media play_sound API."""
    wav_path = os.path.abspath(wav_path)
    print(f"Playing {wav_path}...")
    mini.media.play_sound(wav_path)

    wav_duration = os.path.getsize(wav_path) / mini.media.get_output_audio_samplerate()
    time.sleep(wav_duration)
    print("Playback finished.")


def play_live_tone(mini: "ReachyMini", tone_hz: float) -> None:
    """Push a continuous sine tone to the speaker."""
    sample_rate = mini.media.get_output_audio_samplerate()
    chunk_duration = 0.02  # 20 ms chunks
    samples_per_chunk = int(sample_rate * chunk_duration)
    phase = 0.0

    mini.media.start_playing()
    print(f"Playing {tone_hz} Hz tone (Ctrl+C to stop)...")
    try:
        while True:
            t = np.arange(samples_per_chunk, dtype=np.float32) / sample_rate
            mono = 0.5 * np.sin(2.0 * np.pi * tone_hz * t + phase).astype(np.float32)
            phase += 2.0 * np.pi * tone_hz * samples_per_chunk / sample_rate
            mini.media.push_audio_sample(mono)
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nStopping tone.")
    finally:
        mini.media.stop_playing()


def main(backend: str, wav_path: str | None, tone_hz: float) -> None:
    """Run the sound playback example."""
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    with ReachyMini(log_level="DEBUG", media_backend=backend) as mini:
        if wav_path:
            play_wav(mini, wav_path)
        else:
            play_live_tone(mini, tone_hz)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plays audio on Reachy Mini's speaker."
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["default_no_video", "gstreamer_no_video", "webrtc"],
        default="default_no_video",
        help="Media backend to use.",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--wav",
        type=str,
        help="Path to a wav file to play.",
    )
    mode.add_argument(
        "--live",
        action="store_true",
        help="Push a continuous sine tone.",
    )

    parser.add_argument(
        "--tone-hz",
        default=440.0,
        type=float,
        help="Sine wave frequency in Hz (--live mode only).",
    )

    args = parser.parse_args()
    main(backend=args.backend, wav_path=args.wav, tone_hz=args.tone_hz)

# END doc_example
