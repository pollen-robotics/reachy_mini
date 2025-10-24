"""
Hybrid Music Segmentation Analyzer

Combines spectral clustering with Essentia's feature analysis to:
1. Detect segment boundaries using agglomerative clustering on spectral features
2. Characterize each segment with energy, spectral, and rhythmic features
3. Infer musical structure labels (intro/verse/chorus/bridge/outro)

This replaces the fake percentage-based segmentation with real analysis.
Uses sklearn + Essentia (no librosa to avoid lzma dependency issues).
"""

import numpy as np
import soundfile as sf
import essentia.standard as es
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import uniform_filter1d
from typing import List, Dict, Any
import json


def analyze_segments(audio_path: str, target_segments: int = None) -> List[Dict[str, Any]]:
    """
    Analyze audio file and return labeled segments with per-segment features.

    Args:
        audio_path: Path to audio file
        target_segments: Target number of segments (default: auto-detect 4-6 segments)

    Returns:
        List of segment dictionaries with:
        - start: Start time in seconds
        - end: End time in seconds
        - label: Inferred structural label (intro/verse/chorus/bridge/outro)
        - energy: Average RMS energy for this segment
        - spectral_centroid: Average spectral centroid (brightness)
        - spectral_rolloff: Average spectral rolloff (frequency content)
        - beats_per_segment: Number of beats in this segment
    """

    print(f"[SegmentAnalyzer] Loading audio: {audio_path}")

    # Load audio with soundfile
    y, sr = sf.read(audio_path)
    # Convert to mono if stereo
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)
    duration = len(y) / sr

    print(f"[SegmentAnalyzer] Duration: {duration:.2f}s, Sample rate: {sr}Hz")

    # ====================
    # 1. EXTRACT FEATURES FOR CLUSTERING
    # ====================
    print("[SegmentAnalyzer] Extracting features for segmentation...")

    frame_size = 2048
    hop_size = 512

    # Windowing and spectral analysis
    window = es.Windowing(type='hann')
    spectrum = es.Spectrum()
    mfcc_extractor = es.MFCC(numberCoefficients=13)

    # Extract frame-by-frame MFCCs for clustering
    mfccs_list = []
    frame_times = []

    for i in range(0, len(y) - frame_size, hop_size):
        frame = y[i:i+frame_size]
        windowed = window(frame)
        spec = spectrum(windowed)

        # MFCC bands and coefficients
        bands, mfccs = mfcc_extractor(spec)
        mfccs_list.append(mfccs)
        frame_times.append(i / sr)

    mfccs_array = np.array(mfccs_list)
    frame_times = np.array(frame_times)

    print(f"[SegmentAnalyzer] Extracted {len(mfccs_array)} frames")

    # ====================
    # 2. BOUNDARY DETECTION VIA CLUSTERING
    # ====================
    print("[SegmentAnalyzer] Detecting segment boundaries...")

    # Normalize features
    scaler = StandardScaler()
    mfccs_normalized = scaler.fit_transform(mfccs_array)

    # Apply smoothing to reduce noise
    smoothed = uniform_filter1d(mfccs_normalized, size=9, axis=0)

    # Determine number of clusters
    if target_segments is None:
        # Auto-detect: aim for 4-6 segments for typical songs
        n_segments = max(3, min(6, int(duration / 10)))  # Roughly 10s per segment
    else:
        n_segments = target_segments

    print(f"[SegmentAnalyzer] Clustering into {n_segments} segments...")

    # Agglomerative clustering
    clustering = AgglomerativeClustering(n_clusters=n_segments, linkage='ward')
    labels = clustering.fit_predict(smoothed)

    # Find boundaries where cluster labels change
    boundaries = [0]
    for i in range(1, len(labels)):
        if labels[i] != labels[i-1]:
            boundaries.append(i)
    boundaries.append(len(labels))

    # Convert frame indices to times
    boundary_times = [frame_times[b] for b in boundaries[:-1]]
    boundary_times.append(duration)

    print(f"[SegmentAnalyzer] Found {len(boundary_times)-1} segments")

    # ====================
    # 3. EXTRACT GLOBAL FEATURES
    # ====================
    print("[SegmentAnalyzer] Extracting rhythm and beat information...")

    # Load with Essentia for rhythm analysis
    loader = es.MonoLoader(filename=audio_path, sampleRate=sr)
    y_essentia = loader()

    rhythm_extractor = es.RhythmExtractor2013()
    bpm, beats, _, _, _ = rhythm_extractor(y_essentia)

    # ====================
    # 4. PER-SEGMENT FEATURE EXTRACTION
    # ====================
    print("[SegmentAnalyzer] Extracting per-segment features...")

    # Frame-based features for segments
    rms_extractor = es.RMS()
    centroid_extractor = es.Centroid(range=sr//2)
    rolloff_extractor = es.RollOff()

    frame_energies = []
    frame_centroids = []
    frame_rolloffs = []
    frame_times_detailed = []

    for i in range(0, len(y_essentia) - frame_size, hop_size):
        frame = y_essentia[i:i+frame_size]
        windowed = window(frame)
        spec = spectrum(windowed)

        frame_energies.append(rms_extractor(frame))
        frame_centroids.append(centroid_extractor(spec))
        frame_rolloffs.append(rolloff_extractor(spec))
        frame_times_detailed.append(i / sr)

    frame_energies = np.array(frame_energies)
    frame_centroids = np.array(frame_centroids)
    frame_rolloffs = np.array(frame_rolloffs)
    frame_times_detailed = np.array(frame_times_detailed)

    # ====================
    # 5. BUILD SEGMENT OBJECTS
    # ====================
    segments = []

    for i in range(len(boundary_times) - 1):
        start = boundary_times[i]
        end = boundary_times[i + 1]

        # Find frames in this segment
        segment_mask = (frame_times_detailed >= start) & (frame_times_detailed < end)

        # Average features for this segment
        segment_energy = np.mean(frame_energies[segment_mask]) if np.any(segment_mask) else 0.0
        segment_centroid = np.mean(frame_centroids[segment_mask]) if np.any(segment_mask) else 0.0
        segment_rolloff = np.mean(frame_rolloffs[segment_mask]) if np.any(segment_mask) else 0.0

        # Count beats in this segment
        beats_in_segment = np.sum((beats >= start) & (beats < end))

        segments.append({
            'start': float(start),
            'end': float(end),
            'duration': float(end - start),
            'energy': float(segment_energy),
            'spectral_centroid': float(segment_centroid),
            'spectral_rolloff': float(segment_rolloff),
            'beats_count': int(beats_in_segment),
        })

    # ====================
    # 6. LABEL INFERENCE
    # ====================
    print("[SegmentAnalyzer] Inferring structural labels...")

    segments = infer_labels(segments, duration)

    print("[SegmentAnalyzer] Segmentation complete")
    return segments


def infer_labels(segments: List[Dict], total_duration: float) -> List[Dict]:
    """
    Infer musical structure labels based on position and energy patterns.

    Heuristics:
    - First segment: Usually intro (especially if low energy)
    - Last segment: Usually outro (especially if low/fading energy)
    - High energy segments in middle: Likely chorus
    - Lower energy segments: Likely verse or bridge
    - Very short segments: Transition or bridge
    """

    if not segments:
        return segments

    # Normalize energies for comparison
    energies = np.array([s['energy'] for s in segments])
    mean_energy = np.mean(energies)
    std_energy = np.std(energies)

    for i, segment in enumerate(segments):
        # Position in track (0.0 = start, 1.0 = end)
        position = segment['start'] / total_duration

        # Energy relative to mean (z-score)
        energy_score = (segment['energy'] - mean_energy) / (std_energy + 1e-6)

        # Default label
        label = 'verse'

        # First segment heuristics
        if i == 0:
            if segment['duration'] < 10.0 or energy_score < -0.5:
                label = 'intro'
            else:
                label = 'verse'

        # Last segment heuristics
        elif i == len(segments) - 1:
            if segment['duration'] < 10.0 or energy_score < 0:
                label = 'outro'
            else:
                label = 'chorus'  # Big finish

        # Middle segments
        else:
            # High energy = chorus
            if energy_score > 0.5:
                label = 'chorus'

            # Very short = bridge/transition
            elif segment['duration'] < 8.0:
                label = 'bridge'

            # Medium energy, early/middle position = verse
            elif position < 0.7:
                label = 'verse'

            # Late in track, moderate energy = bridge or final chorus
            else:
                if energy_score > 0:
                    label = 'chorus'
                else:
                    label = 'bridge'

        segment['label'] = label

    return segments


def save_analysis(segments: List[Dict], output_path: str):
    """Save segment analysis to JSON."""
    with open(output_path, 'w') as f:
        json.dump({
            'segments': segments,
            'total_segments': len(segments),
        }, f, indent=2)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python segment_analyzer.py <audio_file> [target_segments]")
        sys.exit(1)

    audio_path = sys.argv[1]
    target = int(sys.argv[2]) if len(sys.argv) > 2 else None

    segments = analyze_segments(audio_path, target)

    print("\n" + "="*80)
    print("SEGMENTATION RESULTS")
    print("="*80)

    for i, seg in enumerate(segments, 1):
        print(f"\n[Segment {i}] {seg['label'].upper()}")
        print(f"  Time: {seg['start']:.2f}s - {seg['end']:.2f}s ({seg['duration']:.2f}s)")
        print(f"  Energy: {seg['energy']:.4f}")
        print(f"  Spectral Centroid: {seg['spectral_centroid']:.2f} Hz")
        print(f"  Spectral Rolloff: {seg['spectral_rolloff']:.2f} Hz")
        print(f"  Beats: {seg['beats_count']}")

    # Save to file
    from pathlib import Path
    output_path = Path(audio_path).parent / (Path(audio_path).stem + '_segments.json')
    save_analysis(segments, str(output_path))
    print(f"\n✓ Analysis saved to: {output_path}\n")
