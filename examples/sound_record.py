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
from reachy_mini.media.media_manager import MediaBackend

DURATION = 5  # seconds
OUTPUT_FILE = "recorded_audio.wav"


def main(backend: str) -> None:
    """Record audio for 5 seconds and save to a WAV file."""
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    with ReachyMini(log_level="INFO", media_backend=backend) as mini:
        print(f"Recording for {DURATION} seconds...")
        audio_samples = []
        t0 = time.time()
        mini.media.start_recording()

        NB_SAMPLES = DURATION * mini.media.get_input_audio_samplerate()
        current_nb_samples = 0
        # make sure that we get the number of samples we want
        # but we also set a timeout
        while current_nb_samples < NB_SAMPLES and time.time() - t0 < DURATION + 2:
            sample = mini.media.get_audio_sample()
            if sample is not None:
                audio_samples.append(sample)
                current_nb_samples += sample.shape[0]

            if mini.media.backend == MediaBackend.DEFAULT_NO_VIDEO:
                # we don't need to poll too fast for sounddevice backend
                time.sleep(0.2)

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
