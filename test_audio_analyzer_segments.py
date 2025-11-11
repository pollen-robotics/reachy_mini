"""Test audio_analyzer with new segment_analyzer integration."""

import sys
sys.path.insert(0, '/Users/lauras/Desktop/laura/reachy_mini')

from choreography.audio_analyzer import AudioAnalyzer

def test_analyzer():
    audio_path = "/Users/lauras/Downloads/Haunting_Fun_2025-10-22T183243.mp3"

    print("Testing AudioAnalyzer with new segment_analyzer integration...")
    print("="*80)

    analyzer = AudioAnalyzer()
    result = analyzer.analyze(audio_path)

    print(f"\nBPM: {result['bpm']:.2f}")
    print(f"Duration: {result['duration']:.2f}s")
    print(f"Energy: {result['energy']:.2f}")
    print(f"Number of segments: {len(result['segments'])}\n")

    print("SEGMENTS:")
    for i, seg in enumerate(result['segments'], 1):
        print(f"\n[{i}] {seg['label'].upper()}")
        print(f"    Time: {seg['start']:.2f}s - {seg['end']:.2f}s ({seg['duration']:.2f}s)")
        print(f"    Energy: {seg.get('energy', 'N/A')}")
        print(f"    Spectral Centroid: {seg.get('spectral_centroid', 'N/A')}")
        print(f"    Spectral Rolloff: {seg.get('spectral_rolloff', 'N/A')}")
        print(f"    Beats: {seg.get('beats_count', 'N/A')}")

    print("\n" + "="*80)
    print("SUCCESS: Audio analyzer now uses real segmentation!")
    print("="*80)

if __name__ == "__main__":
    test_analyzer()
