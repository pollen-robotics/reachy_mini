"""Reachy Mini sound recording example.

The output audio will be saved to 'recorded_audio.wav'.
"""

# START doc_example

import argparse
import logging
import time

import numpy as np
import soundfile as sf

from reachy_mini import ReachyMini

TIMEOUT = 1
DURATION = 5  # seconds
OUTPUT_FILE = "recorded_audio.wav"


def main(backend: str) -> None:
    """Record audio for 5 seconds and save to a WAV file."""
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    with ReachyMini(log_level="INFO", media_backend=backend) as mini:
        audio_samples = []
        mini.media.start_recording()

        # Wait to actually get an audio sample
        print("Waiting for the microphone to be ready...")
        start_time = time.time()
        while (
            mini.media.get_audio_sample() is None and time.time() - start_time < TIMEOUT
        ):
            time.sleep(0.005)

        if time.time() - start_time >= TIMEOUT:
            print(f"Timeout: the microphone did not respond in {TIMEOUT} seconds.")
            return

        print(f"Recording for {DURATION} seconds...")

        start_time = time.time()
        while time.time() - start_time < DURATION:
            sample = mini.media.get_audio_sample()
            if sample is not None:
                audio_samples.append(sample)

        mini.media.stop_recording()

        # Concatenate all samples and save
        if audio_samples:
            audio_data = np.concatenate(audio_samples, axis=0)
            samplerate = mini.media.get_input_audio_samplerate()
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
        choices=["default_no_video", "gstreamer_no_video", "webrtc"],
        default="default_no_video",
        help="Media backend to use.",
    )

    args = parser.parse_args()
    main(backend=args.backend)

# END doc_example
