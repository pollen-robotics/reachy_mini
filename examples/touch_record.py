#!/usr/bin/env python3

import argparse
import queue
import sys

import sounddevice as sd
import soundfile as sf
import pydub

import matplotlib.pyplot as plt

from tempfile import TemporaryFile
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()

    # --action: required string
    parser.add_argument(
        "--touch-action",
        type=str,
        required=True,
        help="Performed touch action: 'carress', 'tap', etc."
    )

    # --folder: optional string or Path (default: current directory)
    parser.add_argument(
        "--storage-folder",
        type=Path,
        default=Path.cwd()/"touch_action_data",
        help="Storage folder (default: current working directory)"
    )

    # --plot: boolean flag
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Whether to plot the recordings (default: False)"
    )

    return parser.parse_args()

def main():
    """
    Touch recording script for ReSpeaker device.
    Records audio from touch actions (carress, tap, etc.), splits it into segments based on silence between actions (minimum 500ms of silence), and saves them to files.
    """

    args = parse_args()

    # Setup the ReSpeaker device for recording
    device_name = "respeaker"
    device_id = [i for i, device in enumerate(sd.query_devices()) if device_name.lower() in device['name'].lower()]

    if len(device_id) == 0:
        raise ValueError(f"Device '{device_name}' not found.")
    elif len(device_id) >= 1:
        if len(device_id) > 1:
            print(f"Multiple devices found with name '{device_name}': {device_id}. Using the first one.")
        print(f"Using device '{device_name}' with index {device_id[0]} : {sd.query_devices(device_id[0])['name']}")
        device_id = device_id[0]

    device_info = sd.query_devices(device_id, "input")
    sample_rate = int(device_info['default_samplerate'])
    channels = device_info['max_input_channels']

    q = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    # Start the recording for an arbitrary duration (Ctrl+C to stop)
    with TemporaryFile() as tmp:
        try:
            with sf.SoundFile(tmp, mode='x', samplerate=sample_rate,
                            channels=channels, format='WAV') as file:
                with sd.InputStream(samplerate=sample_rate, device=device_id,
                                    channels=channels, callback=callback):
                    print('#' * 80)
                    print('press Ctrl+C to stop the recording')
                    print('#' * 80)
                    while True:
                        file.write(q.get())

        except KeyboardInterrupt:
            print("Recording finished")
        except Exception as e:
            print(f"An error occurred: {e}", file=sys.stderr)
            sys.exit(1)

        tmp.seek(0)

        if args.plot:
            data = sf.read(tmp, dtype='float32')
            plt.plot(data[0])
            plt.show()

        # Split the recording into segments based on silence
        audio_segments = pydub.silence.split_on_silence(
            pydub.AudioSegment.from_wav(tmp),
            min_silence_len=500, 
            silence_thresh=-60,
            keep_silence=50
        )

        # Save the segments to files
        first_segment_index = 1
        if not args.storage_folder.exists():
            args.storage_folder.mkdir(parents=True, exist_ok=True)
        else:
            existing_files_indices = [int(f.stem.split('_')[-1]) for f in args.storage_folder.glob(f"{args.touch_action}_*.wav")]
            if existing_files_indices:
                first_segment_index = max(existing_files_indices) + 1

        for i, segment in enumerate(audio_segments, start=first_segment_index): 
            if args.plot:
                plt.plot(segment.get_array_of_samples())
                plt.show(block=False)     
                plt.pause(0.5)               
                plt.close() 
            segment.export(f"{args.storage_folder}/{args.touch_action}_{i}.wav", format="wav")

if __name__ == '__main__':
    main()