"""Reachy Mini sound recording example."""

import argparse
import logging
import time

import numpy as np
import soundfile as sf

from reachy_mini import ReachyMini

DURATION = 5  # seconds
OUTPUT_FILE = "recorded_audio.wav"


def main(backend: str) -> None:
    """Record audio for 5 seconds and save to a WAV file."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    with ReachyMini(log_level="INFO", media_backend=backend) as mini:
        print(f"Recording for {DURATION} seconds...")
        audio_samples = []
        t0 = time.time()
        mini.media.start_recording()

        while mini.media.get_audio_sample() is None:
            print("Waiting for audio data...")
            time.sleep(0.1)

        while time.time() - t0 < DURATION:
            sample = mini.media.get_audio_sample()

            if sample is not None:
                print(f"Got audio sample of shape {sample.shape}")
                audio_samples.append(sample)
            else:
                print("No audio data available yet...")
            if backend == "default":
                time.sleep(0.2)
        mini.media.stop_recording()

        print("Recording stopped.")
        remaining_sample = mini.media.get_audio_sample()
        while remaining_sample is not None:
            audio_samples.append(remaining_sample)
            remaining_sample = mini.media.get_audio_sample()

        # Concatenate all samples and save
        if audio_samples:
            audio_data = np.concatenate(audio_samples, axis=0)
            samplerate = mini.media.get_audio_samplerate()
            sf.write(OUTPUT_FILE, audio_data, samplerate)
            print(f"Audio saved to {OUTPUT_FILE}")
        else:
            print("No audio data recorded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Records audio from Reachy Mini's microphone."
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["default", "gstreamer"],
        default="default",
        help="Media backend to use.",
    )

    args = parser.parse_args()
    main(backend=args.backend)
