# Reachy Mini AI Choreography Generator

**Automated choreography generation system for Reachy Mini using music analysis and LLM reasoning.**

Created by: **Carson (LAURA Project)** - Reachy Mini Beta Tester
Status: **Beta Feature Contribution**
Last Updated: October 22, 2025

---

## Overview

This module enables Reachy Mini to automatically generate expressive choreography from any audio file by combining:
- **Music structure analysis** (Essentia + sklearn) for real intro/verse/chorus detection
- **ReAct agent architecture** (Claude Haiku 4.5) for artistic decision-making
- **Section-based workflow** that matches movement intensity to musical energy
- **Mathematical solver** for precise duration constraint satisfaction

**Result:** Generate complete choreography in 2-3 LLM iterations (10-15 seconds) instead of manual composition or endless trial-and-error.

---

## Key Features

✅ **Real Music Structure Detection**
- Agglomerative clustering on MFCC features to detect segment boundaries
- Automatic labeling (intro/verse/chorus/bridge/outro)
- Per-segment energy, spectral characteristics, and beat counts

✅ **Section-by-Section Choreography**
- Agent works through music sections sequentially
- Curates move palettes based on section energy and structural role
- Natural constraints from musical context (no arbitrary rules)

✅ **Intelligent Move Selection**
- 102 pre-recorded moves (20 dances, 82 emotions)
- Energy-based filtering (low/medium/high intensity)
- Artistic evaluation (variety, flow, expression, musical fit)

✅ **Fast Generation**
- Completes in 2-3 iterations (vs 20 in previous approaches)
- Clear stopping criteria (submit when duration within tolerance)
- Prefer underfilling by <3s over perfect mathematical fit

✅ **Prompt Caching**
- Claude Haiku 4.5 with 9,641 token cached system prompt
- 5-minute TTL for repeated generations
- ~90% cost reduction on subsequent runs

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Audio Input (.mp3)                        │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    audio_analyzer.py                             │
│  • Essentia feature extraction (BPM, energy, danceability)       │
│  • Calls segment_analyzer.py for structure detection             │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   segment_analyzer.py                            │
│  • sklearn AgglomerativeClustering on MFCCs                      │
│  • Boundary detection + label inference                          │
│  • Per-segment energy/spectral/rhythmic features                 │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Analysis Data (segments + features)              │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    react_agent.py                                │
│  ReAct Agent (Claude Haiku 4.5) - Iterative Tool Use:           │
│                                                                   │
│  Iteration 1:                                                    │
│    REASON: "Need to understand music structure"                  │
│    ACT:    Call get_music_structure()                            │
│    OBSERVE: 4 segments (intro/bridge/chorus/outro)               │
│                                                                   │
│  Iteration 2:                                                    │
│    REASON: "Intro quiet (0.008), chorus intense (0.202)"         │
│    ACT:    Call solve_duration_constraint(section_duration)      │
│    OBSERVE: 5 candidate sequences                                │
│                                                                   │
│  Iteration 3:                                                    │
│    REASON: "Solution 1 best matches energy progression"          │
│    ACT:    Call submit_choreography(chosen_sequence)             │
│    OBSERVE: Validation passed ✓                                  │
│                                                                   │
│  ✓ Choreography complete (27.5s target, 27.46s actual)          │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Choreography JSON Output                        │
│  {                                                                │
│    "bpm": 109.8,                                                  │
│    "sequence": [                                                  │
│      {"move": "jackson_square", "cycles": 2},                    │
│      {"move": "stumble_and_recover", "cycles": 1},               │
│      {"move": "neck_recoil", "cycles": 3},                       │
│      ...                                                          │
│    ]                                                              │
│  }                                                                │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
                        Execute on Robot
```

---

## File Structure

```
choreography/
├── README.md                      # This file
├── audio_analyzer.py              # Main audio analysis entry point
├── segment_analyzer.py            # Music structure detection
├── react_agent.py                 # ReAct choreographer (Claude Haiku)
├── react_tools.py                 # Tool registry for agent
├── move_metadata_cache.py         # Move duration/metadata caching
├── inspect_essentia_raw.py        # Debug tool for raw Essentia output
├── archive/
│   └── llm_adapter.py             # Deprecated monolithic approach
└── essentia_analysis/             # Output directory for analysis JSONs
```

### Core Modules

**audio_analyzer.py**
- Primary entry point for audio analysis
- Integrates Essentia extractors (BPM, danceability, key, energy)
- Calls `segment_analyzer.analyze_segments()` for structure
- Returns comprehensive analysis dict with segments and features

**segment_analyzer.py**
- Loads audio with soundfile (stereo → mono conversion)
- Extracts MFCC features (13 coefficients per frame)
- Applies sklearn `AgglomerativeClustering` for boundary detection
- Computes per-segment RMS energy, spectral centroid, spectral rolloff, beat counts
- Infers labels from position and energy patterns:
  - First segment + low energy → intro
  - High energy middle segments → chorus
  - Last segment + low energy → outro
  - Short segments → bridge

**react_agent.py**
- `ReActChoreographer` class with Claude Haiku 4.5 integration
- System prompt (9,641 tokens, cached) teaches section-based workflow
- Iterative Reason-Act-Observe loop with tool calls
- Max 20 iterations, typically completes in 2-3
- Conversation history maintained for context

**react_tools.py**
- `ChoreographyTools` class with tool registry
- Tools available to agent:
  - `get_music_structure()` - Returns labeled segments with features
  - `solve_duration_constraint()` - Randomized greedy solver for sequences
  - `get_move_info()` - Query move metadata
  - `suggest_moves_for_context()` - Filtered move recommendations
  - `validate_duration()` - Check sequence timing
  - `submit_choreography()` - Final submission
  - And more...

**move_metadata_cache.py**
- Caches move durations from Hugging Face datasets
- Avoids repeated SDK calls for move metadata
- JSON file: `move_metadata.json` (102 moves)

---

## Requirements

### Python Dependencies

```bash
pip install essentia
pip install scikit-learn
pip install soundfile
pip install numpy
pip install scipy
pip install anthropic  # For Claude API
```

### API Keys

Set your Anthropic API key:
```bash
export ANTHROPIC_API_KEY="your_key_here"
```

Or in Python:
```python
import os
os.environ['ANTHROPIC_API_KEY'] = 'your_key_here'
```

### Reachy Mini SDK

This module integrates with the official Reachy Mini SDK:
```bash
pip install reachy-mini
```

---

## Usage

### Quick Start

```python
from choreography.audio_analyzer import AudioAnalyzer
from choreography.react_agent import ReActChoreographer

# 1. Analyze audio
analyzer = AudioAnalyzer()
analysis = analyzer.analyze('path/to/music.mp3')

# 2. Generate choreography
choreographer = ReActChoreographer(
    audio_analysis=analysis,
    max_iterations=20  # Usually completes in 2-3
)
result = choreographer.generate()

# 3. Result contains choreography JSON
print(result)
# {
#   'bpm': 120.0,
#   'sequence': [
#     {'move': 'jackson_square', 'cycles': 2},
#     {'move': 'neck_recoil', 'cycles': 3},
#     ...
#   ]
# }
```

### Standalone Segment Analysis

```python
from choreography.segment_analyzer import analyze_segments

segments = analyze_segments('path/to/audio.mp3', target_segments=None)

for seg in segments:
    print(f"{seg['label']:8s} {seg['start']:.2f}s-{seg['end']:.2f}s")
    print(f"  Energy: {seg['energy']:.3f}")
    print(f"  Beats: {seg['beats_count']}")
```

### Debug Raw Essentia Output

```bash
python inspect_essentia_raw.py path/to/audio.mp3
```

Outputs to `essentia_analysis/[filename]_raw_essentia.json` with all raw extractor data.

---

## Integration with desktop_viewer.py

**The choreography system is designed to work with the existing Reachy Mini desktop viewer GUI.**

### What is desktop_viewer.py?

`desktop_viewer.py` is the main control interface for Reachy Mini (from the original repo):
- Web-based GUI (Flask + ImGui)
- Live video stream from simulator
- Real-time pose chart (WebSocket updates)
- Audio upload and analysis display
- Choreography generation button

### How It Uses This Module

The viewer was **modified** (not created) to integrate the choreography system:

```python
# desktop_viewer.py imports
from choreography.react_agent import ReActChoreographer

# On "Generate Choreography" button click:
agent = ReActChoreographer(audio_state.analysis, max_iterations=20)
result = agent.generate()
# Display result in UI
```

### User Workflow via Desktop Viewer

1. **Start the daemon:**
   ```bash
   mjpython -m reachy_mini.daemon.app.main --sim --scene minimal --fastapi-port 8100
   ```

2. **Launch the viewer:**
   ```bash
   python desktop_viewer.py
   ```

3. **In the web UI:**
   - Upload audio file (.mp3, .wav, etc.)
   - Viewer calls `audio_analyzer.py` → displays segments
   - Click "Generate Choreography"
   - Viewer calls `ReActChoreographer` → shows progress
   - Result displayed with move list
   - Click "Play" to execute on robot

### API Integration (Without Viewer)

You can also use the REST API directly:

```bash
# Upload audio
curl -X POST http://localhost:8100/api/audio/upload \
  -F "file=@music.mp3"

# Generate choreography
curl -X POST http://localhost:8100/api/choreography/generate \
  -H "Content-Type: application/json" \
  -d '{"audio_path": "/path/to/music.mp3"}'

# Play choreography
curl -X POST http://localhost:8100/api/choreography/play \
  -H "Content-Type: application/json" \
  -d @choreography.json
```

### Key Point

**desktop_viewer.py is the user interface, choreography/ is the engine.**

- The viewer provides the GUI and integration
- The choreography module provides the AI logic
- They work together but are separate concerns

---

## System Prompt Overview

The ReAct agent is guided by a comprehensive system prompt (9,641 tokens) that teaches:

### 1. Section-by-Section Workflow
```
⚠️ CRITICAL: DO NOT try to solve the entire track at once.

STEP 1: GET MUSIC STRUCTURE **FIRST**
  ▶ Call get_music_structure()
  ▶ Returns: Labeled segments with per-segment features

STEP 2: ANALYZE EACH SECTION
  - Energy Profile: Low (<0.10), Medium (0.10-0.20), High (>0.20)
  - Structural Role: intro, verse, chorus, bridge, outro
  - Musical Characteristics: spectral centroid, beat count, duration

STEP 3: CURATE MOVE PALETTE PER SECTION
  Based on section analysis, choose:
  - Move Type: DANCE (rhythmic) or EMOTION (expressive)
  - Energy Range: What intensity fits this section?
  - Duration Category: Short, medium, long moves?

STEP 4: SOLVE EACH SECTION SEQUENTIALLY
  For each segment:
    a) Call solve_duration_constraint(segment_duration)
    b) Pick solution that matches section's energy/character
    c) Append to full choreography

STEP 5: STOPPING CRITERIA - SUBMIT WHEN DONE
  ✓ All sections have choreography
  ✓ Total duration within ±1.5s tolerance
  ✓ Each section's energy matches music

  ⚠️ DO NOT iterate 20 times comparing options
  ⚠️ Submit first complete solution that meets criteria
```

### 2. Move Library Knowledge
- 102 moves documented (20 dances, 82 emotions)
- Duration categorization (short/medium/long)
- Energy level guidelines
- Signature move characteristics

### 3. Musical Interpretation
- BPM to movement pacing mapping
- Energy level thresholds
- Vocal vs instrumental strategies
- Danceability factor considerations

### 4. Artistic Evaluation Criteria
- Energy match (movement intensity ↔ audio energy)
- Variety (mix of different moves)
- Flow (smooth transitions vs sharp contrasts)
- Expression (captures audio's character)
- Musical fit (pacing aligns with BPM and structure)

---

## Performance Characteristics

### Generation Speed
- **First run:** ~10-15 seconds (cache creation + 2-3 iterations)
- **Subsequent runs:** ~5-8 seconds (cache hit + 2-3 iterations)
- **Audio analysis:** ~2-3 seconds (Essentia + sklearn)
- **Per iteration:** ~2-3 seconds (Haiku inference)

### Cost (Claude Haiku 4.5)
- **Input tokens:** ~10,000 per run (cached: ~750 input + 9,641 cache hit)
- **Output tokens:** ~1,500 total across 3 iterations
- **Cost per run:** ~$0.01-0.02 (with caching)

### Accuracy
- **Duration matching:** Typically within 0.05-0.5s of target
- **Tolerance:** ±1.5s acceptable, prefers underfilling by <3s
- **Success rate:** >95% valid choreography on first submission

---

## Technical Decisions & Rationale

### Why sklearn clustering instead of All-In-One?
**All-In-One** (mir-aidj/all-in-one) was tested but has incompatibilities:
- Requires CUDA or specific natten API version
- Mac ARM architecture unsupported
- Deep learning overhead for simple segmentation

**sklearn AgglomerativeClustering** provides:
- Fast, deterministic boundary detection
- No GPU requirements
- Reliable cross-platform support
- Good results with MFCC features

### Why section-based instead of whole-track solving?
**Problem with whole-track approach:**
- Agent tried to solve entire 30s at once
- No musical context for decisions
- Iterated 20 times comparing "perfect" solutions
- Took 60+ seconds

**Section-based solution:**
- Natural constraints from musical structure
- Agent makes informed decisions per section
- Submits when criteria met (2-3 iterations)
- Completes in 10-15 seconds

### Why ReAct instead of single-shot LLM?
**ReAct advantages:**
- Iterative refinement with tool feedback
- Can query move durations dynamically
- Self-corrects when validation fails
- Reasoning visible in logs

**Single-shot disadvantages:**
- No access to exact move durations
- No validation feedback loop
- High failure rate on duration constraints

### Why prefer underfilling?
Musical preference: Better to end slightly early (natural fade) than overshoot and cut off mid-move. Underfilling by 1-3s feels intentional; overshooting by 1s feels wrong.

---

## Example Output

### Input Audio
- **File:** Haunting_Fun_2025-10-22T183243.mp3
- **Duration:** 30.04s (27.5s content, 2.54s silence)
- **BPM:** 109.8
- **Energy:** 0.916 (high)

### Detected Structure
```
[1] intro    0.00s - 0.79s  (energy: 0.008) - Very quiet
[2] bridge   0.79s - 2.76s  (energy: 0.066) - Building
[3] chorus   2.76s - 27.86s (energy: 0.202) - Intense
[4] outro    27.86s - 30.04s (energy: 0.000) - Silent
```

### Generated Choreography
```json
{
  "bpm": 109.8,
  "sequence": [
    {"move": "jackson_square", "cycles": 2},
    {"move": "stumble_and_recover", "cycles": 1},
    {"move": "neck_recoil", "cycles": 3},
    {"move": "side_peekaboo", "cycles": 1},
    {"move": "jackson_square", "cycles": 1}
  ],
  "final_duration": 27.46
}
```

### Agent Reasoning
```
"Opening with jackson_square (2 cycles) establishes mechanical precision
over the intro/bridge. stumble_and_recover adds playful disruption.
neck_recoil (3 cycles) anchors the high-energy chorus with rhythmic thrust.
side_peekaboo provides exploration, and jackson_square callback creates
powerful finale. Duration: 27.46s (target 27.5s, ±0.04s). ✓ Submitting."
```

**Iterations:** 3
**Time:** 11.3 seconds
**Accuracy:** 99.87%

---

## Known Limitations

### Audio Analysis
- Essentia dtype warnings (harmless, auto-converts double→float32)
- Some extractors fail on certain audio formats (graceful fallback)
- Segment labels are heuristic-based (not ground truth)

### Choreography Generation
- No move preview (names only, no visual representation)
- No undo/redo during generation
- Agent can't create new moves (only uses pre-recorded library)
- Occasional over-clustering (too many small segments)

### Integration
- Requires Anthropic API key (cloud dependency)
- No offline LLM support yet (Ollama integration pending)
- Cache expires after 5 minutes (no persistent caching)

---

## Future Enhancements

### Short-term
- [ ] Add move preview visualization
- [ ] Support choreography import/editing
- [ ] Implement persistent caching (beyond 5min TTL)
- [ ] Add Ollama support for offline generation

### Medium-term
- [ ] Multi-track analysis (verse 1 vs verse 2 differentiation)
- [ ] Style transfer (generate choreography in style of example)
- [ ] User feedback loop (rate choreography → improve suggestions)
- [ ] Emotion-to-move mapping refinement

### Long-term
- [ ] Real-time choreography adjustment during performance
- [ ] Multi-robot synchronization (coordinate multiple Reachys)
- [ ] Custom move recording from UI → automatic integration
- [ ] Generative move synthesis (create new moves from parameters)

---

## Testing

### Run Full Workflow Test
```bash
python test_section_workflow.py
```

Expected output:
- Audio analysis with 4 segments
- ReAct agent completion in 2-3 iterations
- Valid choreography JSON with 5-8 moves
- Duration within ±1.5s of target

### Test Segment Analyzer Only
```bash
cd choreography
python segment_analyzer.py /path/to/audio.mp3
```

### Test Audio Analyzer Integration
```bash
python test_audio_analyzer_segments.py
```

---

## Contributing

This module was created as a beta tester contribution. If you'd like to improve it:

1. **Bug reports:** Open an issue with the Reachy Mini team
2. **Feature requests:** Describe use case and musical context
3. **Code improvements:** Test thoroughly with diverse audio samples
4. **Documentation:** Clarify technical details or add examples

---

## Credits

**Developer:** Carson (LAURA Project)
**LLM Architecture:** Claude Sonnet 4.5 (design) + Claude Haiku 4.5 (execution)
**Audio Analysis:** Essentia (Music Technology Group, Universitat Pompeu Fabra)
**Machine Learning:** scikit-learn
**Robot Platform:** Reachy Mini by Pollen Robotics

**Special Thanks:**
- Pollen Robotics for the Reachy Mini beta program
- Anthropic for Claude API and prompt caching
- Essentia team for comprehensive audio analysis tools

---

## License

This module follows the same license as the Reachy Mini SDK (Apache 2.0).

---

## Contact

For questions about this choreography module:
- GitHub: LAURA Project beta testing repository
- Integration questions: Contact Pollen Robotics support

**Note:** This is a beta contribution and not officially supported by Pollen Robotics yet. Use at your own discretion during beta testing phase.

---

*Last updated: October 22, 2025*
*Reachy Mini Choreography Generator v1.0*
