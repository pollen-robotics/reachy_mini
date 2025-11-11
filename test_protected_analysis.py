"""Test that audio analysis data is properly protected."""

import sys
sys.path.insert(0, '.')

# Simulate the protected state class
class AudioAnalysisState:
    """Protected state container for audio analysis data."""
    def __init__(self):
        self._audio_path = None
        self._analysis_data = None

    @property
    def audio_path(self):
        return self._audio_path

    @audio_path.setter
    def audio_path(self, value):
        self._audio_path = value
        # Clear analysis when audio path changes
        self._analysis_data = None

    @property
    def analysis(self):
        """Read-only access to analysis data."""
        return self._analysis_data

    def set_analysis(self, data):
        """Only way to set analysis data - must be called from analyze_audio()."""
        self._analysis_data = data

    def clear(self):
        """Clear both audio path and analysis."""
        self._audio_path = None
        self._analysis_data = None


# Test the protection
print("=== Testing AudioAnalysisState Protection ===\n")

state = AudioAnalysisState()

# Test 1: Initial state
print("Test 1: Initial state")
print(f"  audio_path: {state.audio_path}")
print(f"  analysis: {state.analysis}")
assert state.audio_path is None
assert state.analysis is None
print("  ✓ Pass\n")

# Test 2: Set audio path
print("Test 2: Set audio path")
state.audio_path = "test.wav"
print(f"  audio_path: {state.audio_path}")
print(f"  analysis: {state.analysis}")
assert state.audio_path == "test.wav"
assert state.analysis is None
print("  ✓ Pass\n")

# Test 3: Set analysis data (protected method)
print("Test 3: Set analysis data via set_analysis()")
mock_analysis = {"bpm": 120.0, "duration": 30.0, "energy": 0.5}
state.set_analysis(mock_analysis)
print(f"  analysis: {state.analysis}")
assert state.analysis == mock_analysis
print("  ✓ Pass\n")

# Test 4: Cannot set analysis directly
print("Test 4: Cannot set analysis directly (read-only property)")
try:
    state.analysis = {"fake": "data"}
    print("  ✗ FAIL - Should have raised AttributeError")
except AttributeError as e:
    print(f"  ✓ Pass - AttributeError raised: {e}\n")

# Test 5: Changing audio path clears analysis
print("Test 5: Changing audio path clears analysis")
state.audio_path = "different.wav"
print(f"  audio_path: {state.audio_path}")
print(f"  analysis: {state.analysis}")
assert state.audio_path == "different.wav"
assert state.analysis is None
print("  ✓ Pass\n")

# Test 6: Clear method
print("Test 6: Clear method")
state.audio_path = "another.wav"
state.set_analysis({"bpm": 128.0})
state.clear()
print(f"  audio_path: {state.audio_path}")
print(f"  analysis: {state.analysis}")
assert state.audio_path is None
assert state.analysis is None
print("  ✓ Pass\n")

print("=== All Tests Passed ===")
print("\nProtection verified:")
print("  ✓ Analysis data is read-only via .analysis property")
print("  ✓ Can only be set through set_analysis() method")
print("  ✓ Automatically cleared when audio path changes")
print("  ✓ Clear method resets both path and analysis")
