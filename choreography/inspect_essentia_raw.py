"""
Raw Essentia Analysis Inspector

Runs Essentia analysis on an audio file and dumps the COMPLETELY RAW output
to JSON without any processing or interpretation.

Usage:
    python inspect_essentia_raw.py /path/to/audio.mp3
"""

import sys
import json
import os
from pathlib import Path
import numpy as np
import essentia.standard as es


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def analyze_raw(audio_path: str) -> dict:
    """
    Run raw Essentia analysis with NO processing or interpretation.

    Returns the raw output from all Essentia extractors.
    """
    print(f"\n{'='*80}")
    print(f"RAW ESSENTIA ANALYSIS")
    print(f"{'='*80}")
    print(f"Audio file: {audio_path}\n")

    # Load audio
    print("[1/10] Loading audio...")
    loader = es.MonoLoader(filename=audio_path)
    audio = loader()

    raw_output = {
        'file_path': audio_path,
        'file_name': os.path.basename(audio_path),
        'audio_length_samples': len(audio),
    }

    # Rhythm Extractor
    print("[2/10] Running RhythmExtractor2013...")
    rhythm = es.RhythmExtractor2013()
    bpm, beats, beats_confidence, _, beats_intervals = rhythm(audio)

    raw_output['rhythm'] = {
        'bpm': float(bpm),
        'beats': beats.tolist(),
        'beats_confidence': float(beats_confidence),
        'beats_intervals': beats_intervals.tolist(),
    }

    # Beat Tracker
    print("[3/10] Running BeatTrackerMultiFeature...")
    beat_tracker = es.BeatTrackerMultiFeature()
    beat_times = beat_tracker(audio)

    # BeatTrackerMultiFeature returns a single array or tuple
    if isinstance(beat_times, (list, tuple)):
        beat_times_list = [float(t) if not isinstance(t, (list, np.ndarray)) else convert_to_serializable(t) for t in beat_times]
    else:
        beat_times_list = beat_times.tolist() if hasattr(beat_times, 'tolist') else beat_times

    raw_output['beat_tracker'] = {
        'beat_times': beat_times_list,
    }

    # Danceability
    print("[4/10] Running Danceability...")
    danceability_extractor = es.Danceability()
    danceability, dfa = danceability_extractor(audio)

    raw_output['danceability'] = {
        'danceability': float(danceability),
        'dfa': convert_to_serializable(dfa),
    }

    # Onset Detection
    print("[5/10] Running OnsetRate...")
    onset_rate_extractor = es.OnsetRate()
    onset_rate, onsets = onset_rate_extractor(audio)

    raw_output['onsets'] = {
        'onset_rate': convert_to_serializable(onset_rate),
        'onset_times': convert_to_serializable(onsets),
    }

    # Key Detection
    print("[6/10] Running KeyExtractor...")
    key_extractor = es.KeyExtractor()
    key, scale, key_strength = key_extractor(audio)

    raw_output['key'] = {
        'key': key,
        'scale': scale,
        'key_strength': convert_to_serializable(key_strength),
    }

    # Loudness
    print("[7/10] Running Loudness...")
    loudness_extractor = es.Loudness()
    loudness = loudness_extractor(audio)

    raw_output['loudness'] = {
        'loudness': convert_to_serializable(loudness),
    }

    # Dynamic Complexity
    print("[8/10] Running DynamicComplexity...")
    dynamic_complexity_extractor = es.DynamicComplexity()
    dynamic_complexity = dynamic_complexity_extractor(audio)

    raw_output['dynamic_complexity'] = {
        'dynamic_complexity': convert_to_serializable(dynamic_complexity),
    }

    # SBic Segmentation (requires frames, not raw audio - skipping)
    print("[9/10] Skipping SBic (requires frame input)...")
    raw_output['sbic_segments'] = {
        'note': 'Skipped - requires frame-based input, not raw audio',
    }

    # Spectral features
    print("[10/10] Running spectral analysis...")

    # Windowing and FFT for spectral analysis
    w = es.Windowing(type='hann')
    spectrum = es.Spectrum()

    # Spectral Centroid
    centroid = es.Centroid(range=len(audio))
    spectral_centroid = centroid(spectrum(w(audio)))

    # Spectral Rolloff
    rolloff = es.RollOff()
    spectral_rolloff = rolloff(spectrum(w(audio)))

    # Spectral Flatness
    flatness = es.Flatness()
    spectral_flatness = flatness(spectrum(w(audio)))

    raw_output['spectral'] = {
        'centroid': convert_to_serializable(spectral_centroid),
        'rolloff': convert_to_serializable(spectral_rolloff),
        'flatness': convert_to_serializable(spectral_flatness),
    }

    print("\n✓ Analysis complete!")
    return raw_output


def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_essentia_raw.py /path/to/audio.mp3")
        sys.exit(1)

    audio_path = sys.argv[1]

    if not os.path.exists(audio_path):
        print(f"✗ Error: File not found: {audio_path}")
        sys.exit(1)

    # Run analysis
    raw_data = analyze_raw(audio_path)

    # Convert to serializable
    serializable_data = convert_to_serializable(raw_data)

    # Create output directory
    output_dir = Path(__file__).parent / 'essentia_analysis'
    output_dir.mkdir(exist_ok=True)

    # Generate output filename
    audio_filename = Path(audio_path).stem
    output_path = output_dir / f"{audio_filename}_raw_essentia.json"

    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(serializable_data, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✓ Raw analysis saved to:")
    print(f"  {output_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
