import argparse
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from reachy_mini import ReachyMini

SAMPLE_RATE = 44100  # samples per second (you can adjust as needed)

parser = argparse.ArgumentParser(description="Record a move with or without sound.")
parser.add_argument(
    "-l", "--library", type=str, required=True, help="Name of the move library."
)
parser.add_argument("-n", "--name", type=str, required=True, help="Name of the move.")
parser.add_argument(
    "--description",
    type=str,
    required=False,
    default="Placeholder description",
    help="Description of the move. You can edit it afterwards if you want.",
)
parser.add_argument(
    "--freq",
    type=int,
    default=100,
    help="Frequency of the recording (in Hz)",
)
parser.add_argument(
    "--audio-device",
    type=str,
    default=None,
    help="Identifier of the audio input device (see --list-audio-devices)",
)
parser.add_argument(
    "--list-audio-devices",
    action="store_true",
    help="List available audio devices and exit",
)
parser.add_argument(
    "--no-sound", action="store_true", help="Record the move without sound."
)
args = parser.parse_args()

if args.list_audio_devices:
    print(sd.query_devices())
    exit(0)


def record(args):
    mini = ReachyMini()
    json_rec_path = os.path.join(args.library, f"{args.name}.json")
    wav_rec_path = os.path.join(args.library, f"{args.name}.wav")

    # Check that the user wants to overwrite.
    if os.path.exists(json_rec_path):
        res = input(
            f"Warning, move {args.name} already exists in library {args.library}. Do you want to overwrite ? (y/N)"
        )
        if not res == "y":
            print("Cancelling recording ... ")
            return
        os.system(f"rm {json_rec_path}")
        if os.path.exists(wav_rec_path):
            os.system(f"rm {wav_rec_path}")

    # This data structure will be populated with the processed recording at the end.
    data: dict = {
        "description": f"{args.description}",
        "time": [],
        "set_target_data": [],
    }

    # --- Schedule a beep sound 1 second after start ---
    def beep():
        duration = 0.2  # Beep duration in seconds
        freq_beep = 440  # Frequency in Hz
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
        beep_sound = 0.5 * np.sin(2 * np.pi * freq_beep * t)
        sd.play(beep_sound, SAMPLE_RATE)
        sd.wait()  # Wait until the beep finishes playing

    time.sleep(1.5)
    beep()

    if not args.no_sound:
        # --- Setup Audio Recording ---
        channels = 1  # mono recording; use 2 for stereo
        audio_frames = []  # will store chunks of recorded audio

        def audio_callback(indata, frames, time_info, status):
            if status:
                print("Audio callback status:", status)
            # Store a copy of the current audio chunk
            audio_frames.append(indata.copy())

        try:
            audio_stream = sd.InputStream(
                device=args.audio_device,
                channels=channels,
                samplerate=SAMPLE_RATE,
                callback=audio_callback,
            )
            audio_stream.start()
            print("Audio recording started using device:", audio_stream.device)
        except Exception as e:
            print("Error starting audio recording:", e)
            print("Available audio devices:")
            print(sd.query_devices())
            return

    try:
        t0 = time.time()
        # Start the recording.
        mini.start_recording()
        print("\nRecording started. Press Ctrl+C here to stop recording.")

        while True:
            # Keep the script alive to listen for Ctrl+C
            time.sleep(0.01)

    except KeyboardInterrupt:
        # Stop recording and retrieve the logged data
        recorded_motion = mini.stop_recording()
        print(f"\nRecording stopped. {len(recorded_motion)} motion frames captured.")
        print(f"Duration of recording: {time.time() - t0:.2f} seconds")

        # Populate the 'data' dictionary from the retrieved 'recorded_motion' list
        for frame in recorded_motion:
            data["time"].append(frame.get("time") - t0)
            # Each "set_target_data" entry will contain the pose data for that frame
            pose_info = {
                "head": frame.get("head"),
                "antennas": frame.get("antennas"),
                "body_yaw": frame.get("body_yaw"),
                "check_collision": frame.get("check_collision"),
            }
            data["set_target_data"].append(pose_info)
        # ---

        if not args.no_sound and audio_stream:
            # Stop the audio stream
            audio_stream.stop()
            audio_stream.close()

        # Save motion data to JSON file
        with open(json_rec_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Robot motion data saved to {json_rec_path}.")

        if not args.no_sound:
            # Save recorded audio to a WAV file.
            if audio_frames:
                audio_data = np.concatenate(audio_frames, axis=0)
                # Use same base name but with .wav extension.

                sf.write(wav_rec_path, audio_data, SAMPLE_RATE)
                print(f"Audio data saved to {wav_rec_path}")
            else:
                print("No audio data was recorded.")


os.makedirs(args.library, exist_ok=True)

record(args)
