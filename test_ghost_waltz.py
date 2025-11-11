"""Test script to analyze Ghost Waltz audio file."""

from choreography import AudioAnalyzer
import json

analyzer = AudioAnalyzer()
analysis = analyzer.analyze('/Users/lauras/Downloads/Ghost_Waltz_2025-10-19T215634-4.wav')

if analysis:
    print('=== AUDIO ANALYSIS RESULTS ===')
    print(f'Duration: {analysis["duration"]:.1f} seconds')
    print(f'BPM: {analysis["bpm"]:.1f}')
    print(f'Energy: {analysis["energy"]:.2f}')
    print(f'Danceability: {analysis["danceability"]:.2f}')
    print('')
    print('Mood:')
    for mood, value in analysis["mood"].items():
        print(f'  {mood}: {value:.2f}')
    print('')
    print(f'Music Structure ({len(analysis["segments"])} segments):')
    for seg in analysis['segments']:
        print(f'  {seg["start"]:.1f}s - {seg["end"]:.1f}s: {seg["label"]} ({seg["end"]-seg["start"]:.1f}s)')
    print('')
    print(f'Beat Count: {len(analysis["beats"])} beats detected')
    print(f'First 10 beat positions: {[round(b, 2) for b in analysis["beats"][:10]]}')
else:
    print('Analysis failed')
