import json  # noqa: D100
import os
from glob import glob
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi


def zero_timestamps(timestamps):
    """In the original recordings, time.time() is recorded.

    This function zeroes out the timestamps by subtracting
    the first timestamp from all timestamps so that time starts at zero
    """
    zeroed_timestamps = []
    first = timestamps[0]
    for ts in timestamps:
        zeroed_timestamps.append(ts - first)

    return zeroed_timestamps


def load_move(path: Path):
    """Load a move from a JSON file.

    Removes the first 1.6 seconds and zeroes the timestamps.
    """
    move = json.load(open(path, "rb"))

    timestamps = zero_timestamps(move["time"])

    # remove the first 1.6seconds
    idx_1_6s = np.searchsorted(timestamps, 1.6)
    timestamps = timestamps[idx_1_6s:]
    move["set_target_data"] = move["set_target_data"][idx_1_6s:]

    timestamps = zero_timestamps(timestamps)

    move["time"] = timestamps

    return move


all_moves_paths = []
all_sounds_paths = []
for move_path in glob("/home/antoine/Pollen/reachy2_emotions/data/recordings/*.json"):
    move_path = Path(move_path)
    all_moves_paths.append(move_path)
    name = move_path.stem
    print(name)
    corresponding_sound = move_path.with_suffix(".wav")
    assert os.path.exists(corresponding_sound), f"Missing sound file for {name}"
    all_sounds_paths.append(corresponding_sound)

os.makedirs("tmp/", exist_ok=True)

for sound_path in all_sounds_paths:
    os.system(f"cp {sound_path} tmp/")

for move_path in all_moves_paths:
    move = load_move(move_path)
    json.dump(move, open(f"tmp/{move_path.stem}.json", "w"))

api = HfApi()

api.create_repo(
    exist_ok=True,
    repo_id="pollen-robotics/reachy-mini-emotions-library",
    repo_type="dataset",
    private=True
)

api.upload_folder(
    folder_path="tmp/",
    repo_id="pollen-robotics/reachy-mini-emotions-library",
    repo_type="dataset",
)
