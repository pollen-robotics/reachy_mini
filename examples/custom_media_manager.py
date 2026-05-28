"""Demonstrate direct camera and audio access after releasing daemon media.

When you need raw OpenCV camera access or sounddevice audio recording,
the daemon must first release the hardware. This example shows how:

1. Connect with ``media_backend="no_media"`` — this automatically tells
   the daemon to release camera and audio hardware.
2. Use OpenCV to capture frames directly from the camera.
3. Use sounddevice to record audio from the microphone.
4. On exit (context manager ``__exit__``), media is automatically
   re-acquired by the daemon.

Note:
    Requires: pip install opencv-python sounddevice soundfile

"""

# START doc_example

import sys
import time

try:
    import cv2
except ImportError:
    print("Error: OpenCV is required. Install with: pip install opencv-python")
    sys.exit(1)

try:
    import sounddevice as sd
    import soundfile as sf
except ImportError:
    print("Error: sounddevice + soundfile are required.")
    print("Install with: pip install sounddevice soundfile")
    sys.exit(1)

import numpy as np

from reachy_mini import ReachyMini


def main() -> None:
    """Capture a frame with OpenCV and record audio with sounddevice."""
    # media_backend="no_media" automatically releases daemon media hardware
    with ReachyMini(media_backend="no_media") as mini:
        print(f"Connected. media_released={mini.media_released}")

        # --- OpenCV camera capture ---
        print("\nOpening camera with OpenCV...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Could not open camera — is it plugged in?")
        else:
            # Let the camera auto-adjust
            for _ in range(10):
                cap.read()

            ret, frame = cap.read()
            if ret:
                cv2.imwrite("release_media_frame.jpg", frame)
                print(f"Saved frame: release_media_frame.jpg ({frame.shape})")
            else:
                print("Failed to capture frame.")
            cap.release()

        # --- sounddevice audio recording ---
        duration = 2.0  # seconds
        samplerate = 44100
        print(f"\nRecording {duration}s of audio at {samplerate} Hz...")
        try:
            audio = sd.rec(
                int(duration * samplerate),
                samplerate=samplerate,
                channels=1,
                dtype=np.int16,
            )
            sd.wait()
            sf.write("release_media_audio.wav", audio, samplerate)
            print("Saved audio: release_media_audio.wav")
        except Exception as e:
            print(f"Audio recording failed: {e}")

        # The robot is still controllable while media is released
        print("\nWiggling antennas to prove robot control still works...")
        mini.goto_target(antennas=[0.3, -0.3], duration=0.5)
        time.sleep(0.2)
        mini.goto_target(antennas=[0.0, 0.0], duration=0.5)

    # __exit__ automatically calls acquire_media() → daemon reclaims hardware
    print("\nDone. Daemon media re-acquired on exit.")


if __name__ == "__main__":
    main()

# END doc_example
