"""Reachy Mini sound playback example.

Two modes:
  --file <path>  Play a sound file (WAV, OGG, MP3, FLAC, ...) via play_sound.
  --live         Push a continuous sine tone using push_audio_sample.
"""

# START doc_example

import argparse
import os
import time

import numpy as np

from reachy_mini import ReachyMini
from reachy_mini.media.gstreamer_utils import audio_duration_seconds


def play_file(mini: "ReachyMini", file_path: str) -> None:
    """Play a sound file using the media play_sound API."""
    file_path = os.path.abspath(file_path)
    print(f"Playing {file_path}...")
    mini.media.play_sound(file_path)

    time.sleep(audio_duration_seconds(file_path))
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


def main(
    backend: str, file_path: str | None, tone_hz: float, wobbling: bool = False
) -> None:
    """Run the sound playback example."""
    with ReachyMini(log_level="DEBUG", media_backend=backend) as mini:
        if wobbling:
            mini.enable_wobbling()
        if file_path:
            play_file(mini, file_path)
        else:
            play_live_tone(mini, tone_hz)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plays audio on Reachy Mini's speaker."
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["default", "local", "webrtc"],
        default="default",
        help="Media backend to use.",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--file",
        type=str,
        help="Path to a sound file to play (WAV, OGG, MP3, FLAC, ...).",
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
    parser.add_argument(
        "--wobbling",
        action="store_true",
        help="Enable audio-reactive head wobbling.",
    )

    args = parser.parse_args()
    main(
        backend=args.backend,
        file_path=args.file,
        tone_hz=args.tone_hz,
        wobbling=args.wobbling,
    )

# END doc_example
