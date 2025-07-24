import argparse

import joblib
import librosa
import numpy as np
import pandas as pd
import sounddevice as sd
from scipy.stats import skew


def get_mfcc_from_audio(audio, sample_rate, mfcc_number=30):
    ft1 = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=mfcc_number)
    ft2 = librosa.feature.zero_crossing_rate(y=audio)[0]
    ft3 = librosa.feature.spectral_rolloff(y=audio)[0]
    ft4 = librosa.feature.spectral_centroid(y=audio)[0]
    ft5 = librosa.feature.spectral_contrast(y=audio)[0]
    ft6 = librosa.feature.spectral_bandwidth(y=audio)[0]

    ft1_trunc = np.hstack(
        (
            np.mean(ft1, axis=1),
            np.std(ft1, axis=1),
            skew(ft1, axis=1),
            np.max(ft1, axis=1),
            np.median(ft1, axis=1),
            np.min(ft1, axis=1),
        )
    )
    ft2_trunc = np.hstack(
        (
            np.mean(ft2),
            np.std(ft2),
            skew(ft2),
            np.max(ft2),
            np.median(ft2),
            np.min(ft2),
        )
    )
    ft3_trunc = np.hstack(
        (
            np.mean(ft3),
            np.std(ft3),
            skew(ft3),
            np.max(ft3),
            np.median(ft3),
            np.min(ft3),
        )
    )
    ft4_trunc = np.hstack(
        (
            np.mean(ft4),
            np.std(ft4),
            skew(ft4),
            np.max(ft4),
            np.median(ft4),
            np.min(ft4),
        )
    )
    ft5_trunc = np.hstack(
        (
            np.mean(ft5),
            np.std(ft5),
            skew(ft5),
            np.max(ft5),
            np.median(ft5),
            np.min(ft5),
        )
    )
    ft6_trunc = np.hstack(
        (
            np.mean(ft6),
            np.std(ft6),
            skew(ft6),
            np.max(ft6),
            np.median(ft6),
            np.max(ft6),
        )
    )
    return np.hstack((ft1_trunc, ft2_trunc, ft3_trunc, ft4_trunc, ft5_trunc, ft6_trunc))


def main():
    parser = argparse.ArgumentParser(
        description="Live predict touch class from microphone input."
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to the saved model (joblib file)"
    )
    parser.add_argument(
        "--window",
        type=float,
        default=1.0,
        help="Sliding window duration in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=16000, help="Sample rate for audio recording"
    )
    parser.add_argument(
        "--mfcc-number",
        type=int,
        default=30,
        help="Number of MFCCs to use (default: 30)",
    )
    parser.add_argument(
        "--hop",
        type=float,
        default=0.5,
        help="Hop size in seconds between predictions (default: 0.5)",
    )
    args = parser.parse_args()

    print(f"Loading model from {args.model} ...")
    bundle = joblib.load(args.model)
    model = bundle["model"]
    scaler = bundle["scaler"]
    pca = bundle["pca"]
    index_to_label = bundle["index_to_label"]

    # Find respeaker device
    device_name = "respeaker"
    device_id = [
        i
        for i, device in enumerate(sd.query_devices())
        if device_name.lower() in device["name"].lower()
    ]
    if len(device_id) == 0:
        raise ValueError(f"Device '{device_name}' not found.")
    elif len(device_id) >= 1:
        if len(device_id) > 1:
            print(
                f"Multiple devices found with name '{device_name}': {device_id}. Using the first one."
            )
        device_name_str = sd.query_devices(device_id[0])["name"]
        print(
            f"Using device '{device_name}' with index {device_id[0]} : {device_name_str}"
        )
        device_id = device_id[0]

    device_info = sd.query_devices(device_id, "input")
    sample_rate = int(device_info["default_samplerate"])
    channels = 1  # For prediction, we use mono

    print("Live detection started. Press Ctrl+C to stop.")
    import collections
    import time

    window_size = int(args.window * sample_rate)
    hop_size = int(args.hop * sample_rate)
    buffer = collections.deque(maxlen=window_size)

    stream = sd.InputStream(
        samplerate=sample_rate,
        device=device_id,
        channels=channels,
        dtype="float32",
        blocksize=hop_size,
    )
    stream.start()
    try:
        while True:
            audio_chunk, _ = stream.read(hop_size)
            audio_chunk = audio_chunk.flatten()
            buffer.extend(audio_chunk)
            if len(buffer) == window_size:
                audio_np = np.array(buffer)
                # If sample_rate != args.sample_rate, resample
                if sample_rate != args.sample_rate:
                    audio_np = librosa.resample(
                        audio_np, orig_sr=sample_rate, target_sr=args.sample_rate
                    )
                    sr = args.sample_rate
                else:
                    sr = sample_rate
                features = get_mfcc_from_audio(
                    audio_np, sr, mfcc_number=args.mfcc_number
                )
                features = features.reshape(1, -1)
                features_scaled = scaler.transform(features)
                features_pca = pca.transform(features_scaled)
                proba = model.predict_proba(features_pca)[0]
                idx_sorted = np.argsort(proba)[::-1]
                print(f"Predicted class: {index_to_label[idx_sorted[0]]}")
                # print("Top 3 predictions:")
                # for i in idx_sorted[:3]:
                #    print(f"  {index_to_label[i]}: {proba[i]:.3f}")
                # print(
                #    f"All probabilities: {[(index_to_label[i], float(proba[i])) for i in range(len(proba))]}"
                # )
                # print("-" * 40)
            time.sleep(args.hop / 2)
    except KeyboardInterrupt:
        print("Live detection stopped.")
    finally:
        stream.stop()
        stream.close()


if __name__ == "__main__":
    main()
