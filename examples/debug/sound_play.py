"""Reachy Mini sound playback example.

Open a wav and push samples to the speaker. This is a toy example, in real
conditions output from a microphone or a text-to-speech engine would be
 pushed to the speaker instead.
"""

import argparse
import logging
import os
import time

import numpy as np
import scipy
import soundfile as sf

from reachy_mini import ReachyMini
from reachy_mini.utils.constants import ASSETS_ROOT_PATH

INPUT_FILE = os.path.join(ASSETS_ROOT_PATH, "wake_up.wav")


def main(backend: str) -> None:
    """Play a wav file by pushing samples to the audio device."""
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    with ReachyMini(log_level="DEBUG", media_backend=backend) as mini:
        data, samplerate_in = sf.read(INPUT_FILE, dtype="float32")

        if samplerate_in != mini.media.get_output_audio_samplerate():
            data = scipy.signal.resample(
                data,
                int(
                    len(data)
                    * (mini.media.get_output_audio_samplerate() / samplerate_in)
                ),
            )
        if data.ndim > 1:  # convert to mono
            data = np.mean(data, axis=1)

        mini.media.start_playing()
        print("Playing audio...")
        # Push samples in chunks
        chunk_size = 1024
        for i in range(0, len(data), chunk_size):
            chunk = data[i : i + chunk_size]
            mini.media.push_audio_sample(chunk)

        time.sleep(1)  # wait a bit to ensure all samples are played
        mini.media.stop_playing()
        print("Playback finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plays a wav file on Reachy Mini's speaker."
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["default", "gstreamer", "webrtc"],
        default="default",
        help="Media backend to use.",
    )

    args = parser.parse_args()
    main(backend=args.backend)
