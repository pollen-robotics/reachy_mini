"""
Test All-In-One music structure analyzer on Haunting_Fun audio.

This script tests whether All-In-One can detect real music structure
(intro/verse/chorus/bridge/outro) instead of fake percentage-based segmentation.
"""

import json
from pathlib import Path
import allin1

def test_allin1():
    audio_path = "/Users/lauras/Downloads/Haunting_Fun_2025-10-22T183243.mp3"

    print("="*80)
    print("Testing All-In-One Music Structure Analyzer")
    print("="*80)
    print(f"Audio file: {audio_path}\n")

    print("[1/2] Running All-In-One analysis...")
    try:
        # Run All-In-One analysis
        result = allin1.analyze(audio_path)

        print("✓ Analysis complete!\n")

        print("[2/2] Examining results...")
        print(f"\nResult type: {type(result)}")
        print(f"Result keys: {result.keys() if hasattr(result, 'keys') else 'N/A'}\n")

        # Save full result to JSON for inspection
        output_dir = Path("/Users/lauras/Desktop/laura/reachy_mini/choreography/essentia_analysis")
        output_dir.mkdir(exist_ok=True)

        output_path = output_dir / "Haunting_Fun_allin1_output.json"

        # Convert result to serializable format
        serializable_result = {}
        if hasattr(result, 'keys'):
            for key in result.keys():
                value = result[key]
                if hasattr(value, 'tolist'):
                    serializable_result[key] = value.tolist()
                elif hasattr(value, '__dict__'):
                    serializable_result[key] = str(value)
                else:
                    serializable_result[key] = value
        else:
            serializable_result = str(result)

        with open(output_path, 'w') as f:
            json.dump(serializable_result, f, indent=2)

        print(f"✓ Full result saved to: {output_path}\n")

        # Try to print structure information if available
        if hasattr(result, 'keys'):
            if 'segments' in result:
                print("SEGMENTS FOUND:")
                segments = result['segments']
                for i, seg in enumerate(segments):
                    print(f"  {i+1}. {seg}")
                print()

            if 'structure' in result:
                print("STRUCTURE FOUND:")
                print(f"  {result['structure']}\n")

            # Print all top-level keys
            print("ALL RESULT KEYS:")
            for key in result.keys():
                print(f"  - {key}")

        print("\n" + "="*80)
        print("SUCCESS: All-In-One analysis completed")
        print("="*80)

    except Exception as e:
        print(f"\n✗ Error during analysis: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_allin1()
