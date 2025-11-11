"""Quick test script for choreography system."""

import json
from choreography import AudioAnalyzer, ChoreographyLLM

# Test with a sample audio file (user will need to provide one)
print("Choreography System Test")
print("=" * 50)

# Note: You need an actual audio file to test
# audio_file = "path/to/your/test.mp3"

print("\nTo test the system:")
print("1. Place an audio file (mp3, wav, flac) in this directory")
print("2. Update the audio_file variable above")
print("3. Run this script")
print("\nThe system will:")
print("  - Analyze the audio with Essentia")
print("  - Generate choreography with Claude/Ollama")
print("  - Save recommendation to responses/ directory")

# Uncomment below when you have an audio file:
"""
# Load moves library
with open('moves.json', 'r') as f:
    moves_data = json.load(f)
    moves_library = {
        'dances': sorted(moves_data.get('dances', [])),
        'emotions': sorted(moves_data.get('emotions', []))
    }

# Analyze audio
print("\nAnalyzing audio...")
analyzer = AudioAnalyzer()
analysis = analyzer.analyze(audio_file)

if analysis:
    print(f"✓ BPM: {analysis['bpm']:.1f}")
    print(f"✓ Duration: {analysis['duration']:.1f}s")
    print(f"✓ Energy: {analysis['energy']:.2f}")
    print(f"✓ Segments: {len(analysis['segments'])}")

    # Generate choreography
    print("\nGenerating choreography...")
    llm = ChoreographyLLM(provider="anthropic")
    recommendation = llm.generate_recommendation(
        analysis,
        moves_library,
        save_to="test_recommendation.json"
    )

    if recommendation:
        moves = recommendation.get('choreography', [])
        print(f"✓ Generated {len(moves)} moves")
        print(f"✓ Coverage: {recommendation.get('total_duration_filled', 0):.1f}s")
        print("\nFirst 3 moves:")
        for move in moves[:3]:
            print(f"  - {move['timestamp']:.1f}s: {move['move_name']} ({move['move_type']})")
        print(f"\n✓ Saved to test_recommendation.json")
    else:
        print("✗ Failed to generate choreography")
else:
    print("✗ Failed to analyze audio")
"""

print("\n" + "=" * 50)
print("Test complete!")
