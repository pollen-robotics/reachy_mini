"""Test script for ReAct Choreography Agent."""

import json
import sys
from choreography.react_agent import ReActChoreographer
from choreography.audio_analyzer import AudioAnalyzer

# Use existing test audio
AUDIO_FILE = "/Users/lauras/Downloads/Cemetery_Rave_2025-10-19T220407.wav"

print("=" * 80)
print("REACT CHOREOGRAPHY AGENT TEST")
print("=" * 80)

print(f"\nAudio file: {AUDIO_FILE}")

try:
    # Analyze audio first
    print("\n[1] Analyzing audio with Essentia...")
    analyzer = AudioAnalyzer()
    audio_analysis = analyzer.analyze(AUDIO_FILE)

    if not audio_analysis:
        print("✗ Failed to analyze audio")
        sys.exit(1)

    print(f"✓ BPM: {audio_analysis['bpm']:.1f}")
    print(f"✓ Duration: {audio_analysis['duration']:.1f}s")
    print(f"✓ Energy: {audio_analysis['energy']:.2f}")

    # Create ReAct agent
    print("\n[2] Initializing ReAct agent...")
    agent = ReActChoreographer(audio_analysis)

    print("\n[3] Starting choreography generation...")
    print("     This will use Claude Haiku 4.5 with prompt caching")
    print("     Watch for cache write/read messages in console")
    print()

    # Generate choreography
    choreography = agent.generate()

    if choreography:
        print("\n" + "=" * 80)
        print("SUCCESS - CHOREOGRAPHY GENERATED")
        print("=" * 80)
        print(f"\nBPM: {choreography['bpm']:.1f}")
        print(f"Moves: {len(choreography['sequence'])}")
        print("\nSequence:")
        for i, move in enumerate(choreography['sequence'], 1):
            print(f"  {i}. {move['move']} × {move.get('cycles', 1)} cycle(s)")

        # Save to file
        output_file = "test_react_output.json"
        with open(output_file, 'w') as f:
            json.dump(choreography, f, indent=2)
        print(f"\n✓ Saved to {output_file}")

    else:
        print("\n" + "=" * 80)
        print("FAILED - No choreography generated")
        print("=" * 80)
        sys.exit(1)

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
