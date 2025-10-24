#!/usr/bin/env python3
"""
ReAct Choreographer - Standalone CLI

Generates precise choreography using LLM-based ReAct agent with tools.

Usage:
    python react_choreographer.py --audio audio.wav --output choreo.json
    python react_choreographer.py --analysis analysis.json --output choreo.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
import os
sys.path.insert(0, str(Path(__file__).parent.parent))

from choreography.audio_analyzer import AudioAnalyzer
from choreography.react_agent import ReActChoreographer


def load_audio_analysis(analysis_path: str) -> dict:
    """Load pre-computed audio analysis from JSON."""
    print(f"[CLI] Loading audio analysis from {analysis_path}")
    with open(analysis_path, 'r') as f:
        return json.load(f)


def analyze_audio(audio_path: str) -> dict:
    """Analyze audio file and return analysis dict."""
    print(f"[CLI] Analyzing audio file: {audio_path}")
    analyzer = AudioAnalyzer()
    analysis = analyzer.analyze(audio_path)

    if analysis is None:
        raise ValueError(f"Failed to analyze audio file: {audio_path}")

    return analysis


def save_choreography(choreography: dict, output_path: str):
    """Save choreography to JSON file."""
    print(f"[CLI] Saving choreography to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(choreography, f, indent=2)
    print(f"[CLI] Choreography saved successfully")


def main():
    parser = argparse.ArgumentParser(
        description="Generate choreography using ReAct agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze audio and generate choreography
  python react_choreographer.py --audio cemetery_rave.wav --output choreo.json

  # Use pre-computed analysis
  python react_choreographer.py --analysis analysis.json --output choreo.json

  # Specify max iterations
  python react_choreographer.py --audio audio.wav --output choreo.json --max-iterations 30
        """
    )

    parser.add_argument('--audio', type=str,
                       help='Path to audio file to analyze')
    parser.add_argument('--analysis', type=str,
                       help='Path to pre-computed audio analysis JSON')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for choreography JSON')
    parser.add_argument('--max-iterations', type=int, default=20,
                       help='Maximum ReAct iterations (default: 20)')
    parser.add_argument('--save-analysis', type=str,
                       help='Save audio analysis to this path (optional)')

    args = parser.parse_args()

    # Validate arguments
    if not args.audio and not args.analysis:
        parser.error("Either --audio or --analysis must be provided")

    if args.audio and args.analysis:
        parser.error("Cannot specify both --audio and --analysis")

    try:
        # Get audio analysis
        if args.analysis:
            analysis = load_audio_analysis(args.analysis)
        else:
            analysis = analyze_audio(args.audio)

            # Save analysis if requested
            if args.save_analysis:
                print(f"[CLI] Saving analysis to {args.save_analysis}")
                with open(args.save_analysis, 'w') as f:
                    json.dump(analysis, f, indent=2)

        # Print analysis summary
        print("\n" + "="*60)
        print("AUDIO ANALYSIS SUMMARY")
        print("="*60)
        print(f"Duration: {analysis['duration']:.1f}s")
        print(f"BPM: {analysis['bpm']:.1f}")
        print(f"Energy: {analysis['energy']:.3f}")
        print(f"Danceability: {analysis['danceability']:.3f}")
        vocal_prob = analysis.get('vocal_instrumental', {}).get('vocal_probability', 0.5)
        print(f"Vocal probability: {vocal_prob:.3f}")
        print("="*60 + "\n")

        # Generate choreography
        print("[CLI] Starting ReAct choreographer...")
        agent = ReActChoreographer(analysis, max_iterations=args.max_iterations)
        choreography = agent.generate()

        if choreography is None:
            print("\n[CLI] ERROR: Failed to generate choreography")
            sys.exit(1)

        # Save choreography
        save_choreography(choreography, args.output)

        print(f"\n[CLI] SUCCESS! Choreography generated")
        print(f"  Output: {args.output}")
        print(f"  Moves: {len(choreography['sequence'])}")
        print(f"  BPM: {choreography['bpm']}")

        sys.exit(0)

    except Exception as e:
        print(f"\n[CLI] ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
