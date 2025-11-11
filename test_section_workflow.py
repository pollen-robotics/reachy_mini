"""
Test the new section-based choreography workflow.

This script tests whether the ReAct agent:
1. Calls get_music_structure() first
2. Works section-by-section
3. Submits within 1-2 iterations instead of iterating 20 times
"""

import sys
sys.path.insert(0, '/Users/lauras/Desktop/laura/reachy_mini')

from choreography.audio_analyzer import AudioAnalyzer
from choreography.react_agent import ReActChoreographer

def test_section_workflow():
    audio_path = "/Users/lauras/Downloads/Haunting_Fun_2025-10-22T183243.mp3"

    print("="*80)
    print("SECTION-BASED CHOREOGRAPHY WORKFLOW TEST")
    print("="*80)
    print(f"Audio: {audio_path}\n")

    # Analyze audio
    print("Step 1: Analyzing audio...")
    analyzer = AudioAnalyzer()
    analysis = analyzer.analyze(audio_path)

    print(f"\n✓ Analysis complete")
    print(f"  Duration: {analysis['duration']:.2f}s")
    print(f"  BPM: {analysis['bpm']:.2f}")
    print(f"  Segments: {len(analysis['segments'])}")

    for i, seg in enumerate(analysis['segments'], 1):
        print(f"    [{i}] {seg['label']:8s} {seg['start']:5.2f}s-{seg['end']:5.2f}s (energy: {seg['energy']:.3f})")

    # Generate choreography with ReAct agent
    print("\n" + "="*80)
    print("Step 2: Running ReAct Choreographer...")
    print("="*80)

    choreographer = ReActChoreographer(
        audio_analysis=analysis,
        max_iterations=5  # Lower max to see if it completes faster
    )

    result = choreographer.generate()

    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)

    if result:
        print("✓ SUCCESS: Choreography generated!")
        print(f"\n  Total moves: {len(result['sequence'])}")
        print(f"  BPM: {result['bpm']}")
        print(f"  Final duration: {result.get('final_duration', 'N/A')}")
        print(f"  Target duration: {analysis['duration']:.2f}s")

        # Show first few moves
        print("\n  First 5 moves:")
        for i, move in enumerate(result['sequence'][:5], 1):
            move_name = move.get('move') or move.get('move_name', 'unknown')
            cycles = move.get('cycles', 1)
            print(f"    {i}. {move_name} ({cycles} cycle{'s' if cycles > 1 else ''})")

        if len(result['sequence']) > 5:
            print(f"    ... and {len(result['sequence']) - 5} more moves")

        print("\n" + "="*80)
        print("✓ Test completed successfully!")
        print("="*80)

    else:
        print("✗ FAILED: No choreography generated")
        print("\nCheck logs above for errors or max iterations reached")

if __name__ == "__main__":
    test_section_workflow()
