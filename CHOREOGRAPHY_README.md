# Choreography Builder for Reachy Mini

**Status:** MVP Complete (Phase 1-3)
**Created:** 2025-10-19

## What's Been Built

A choreography creation system integrated into the desktop viewer that uses AI to generate robot dance sequences from music analysis.

### Core Features (✅ Implemented)

1. **Audio Analysis (Essentia)**
   - Import audio files (.mp3, .wav, .flac, etc.)
   - Extract BPM, energy, mood, danceability
   - Detect music structure (intro, verse, chorus, etc.)

2. **AI Choreography Generation**
   - LLM adapter pattern (supports Anthropic Claude, Ollama, HuggingFace)
   - Generates move sequences that fill entire audio duration
   - Matches move energy to music features
   - Includes descriptions and reasoning for each move

3. **Desktop Viewer Integration**
   - New "Choreography Builder" panel in UI
   - Audio import with file picker
   - Background thread processing (non-blocking)
   - Real-time status feedback
   - Move list display with hover tooltips

4. **Export System**
   - Saves recommendations to `responses/choreography_recommendations/`
   - Exports final choreography to `choreographies/`
   - JSON format compatible with daemon

## File Structure

```
reachy_mini/
├── choreography/
│   ├── __init__.py              # Module exports
│   ├── audio_analyzer.py        # Essentia audio analysis
│   └── llm_adapter.py           # Multi-provider LLM inference
│
├── responses/
│   └── choreography_recommendations/  # LLM outputs saved here
│
├── choreographies/                     # Final exports saved here
│
├── training_data/                      # Future: user edit logs
│   └── sessions/
│
├── desktop_viewer.py            # Main viewer (MODIFIED)
└── test_choreography.py         # Test script
```

## How to Use

### 1. Start Desktop Viewer

```bash
source /Users/lauras/Desktop/laura/venv/bin/activate
python3 desktop_viewer.py --scene minimal
```

### 2. Import Audio

1. Expand "Choreography Builder" panel
2. Click "Import Audio..."
3. Select your music file

### 3. Analyze

1. Click "Analyze Audio"
2. Wait 2-5 seconds
3. Review BPM, energy, sections

### 4. Generate Choreography

1. Select LLM provider (Anthropic or Ollama)
2. Click "Generate Choreography"
3. Wait 3-10 seconds
4. Review recommended moves

### 5. Export

1. Click "Export Final Choreography"
2. Find output in `choreographies/` directory

## Configuration

### LLM Providers

**Anthropic Claude (Default):**
- Requires: `ANTHROPIC_API_KEY` environment variable
- Model: `claude-3-5-haiku-20241022`
- Cost: ~$0.002 per choreography

**Ollama (Local):**
- Requires: Ollama running (`ollama serve`)
- Model: Default `llama3.2`
- Free, offline

**HuggingFace (Future):**
- For custom fine-tuned model from Clem
- Not yet implemented

## Output Format

### Recommendation JSON

```json
{
  "recommendation_id": "rec_20251019_143022",
  "audio_file": "song.mp3",
  "audio_duration": 180.0,
  "bpm": 128.5,
  "generated_by": "anthropic/claude-3-5-haiku-20241022",
  "timestamp": "2025-10-19T14:30:22",
  "choreography": [
    {
      "index": 0,
      "timestamp": 0.0,
      "section": "intro",
      "move_type": "emotion",
      "move_name": "welcoming1",
      "duration": 3.0,
      "description": "Friendly greeting gesture",
      "reasoning": "Sets welcoming tone"
    }
  ],
  "total_duration_filled": 180.0,
  "coverage": 1.0
}
```

### Final Choreography JSON

```json
{
  "description": "Choreography for song.mp3",
  "audio_file": "song.mp3",
  "bpm": 128.5,
  "duration": 180.0,
  "created_from_recommendation": "rec_20251019_143022",
  "edited_by_user": true,
  "choreography": [ /* same format */ ]
}
```

## Dependencies

**Python Packages:**
- `essentia-tensorflow` - Audio analysis
- `anthropic` - Claude API (if using Anthropic)
- `requests` - Ollama API (if using Ollama)

**Environment:**
- `ANTHROPIC_API_KEY` - Required for Claude

## What's Next (Not Yet Implemented)

### Phase 4: Move Editing (Planned)
- Delete moves
- Replace moves from dropdown
- Reorder moves (up/down buttons)
- Adjust duration/cycles
- Insert idle positions
- Real-time duration validation

### Phase 5: Enhanced UI (Planned)
- Timeline visualization
- Beat markers overlay
- Segment boundaries
- Scrollable move table
- Search/filter moves

### Phase 6: Training Data Collection (Planned)
- Log user edits
- Track replacements and deletions
- Save for model fine-tuning
- A/B testing different providers

## Testing

Run the test script:

```bash
python3 test_choreography.py
```

(You'll need to add an audio file and uncomment the test code)

## Troubleshooting

**"No module named 'essentia'":**
```bash
pip install essentia-tensorflow
```

**"ANTHROPIC_API_KEY not found":**
```bash
export ANTHROPIC_API_KEY="your-key-here"
```

**Ollama connection error:**
```bash
# Start Ollama server
ollama serve

# In another terminal
ollama pull llama3.2
```

**Audio analysis fails:**
- Check audio file format (mp3, wav, flac supported)
- Verify file isn't corrupted
- Try a different audio file

**Generation takes too long:**
- Claude: Usually 3-10 seconds (depends on API)
- Ollama: Can take 30-60 seconds (depends on hardware)
- Try a shorter audio file for testing

## Known Limitations

- Move editing not yet implemented (Phase 4)
- Only shows first 5 moves in UI (full list in JSON)
- No undo/redo functionality
- No preview playback in MuJoCo
- No timeline visualization

## Technical Notes

### LLM Adapter Pattern

The system uses the same adapter pattern as `main_loop.py`, allowing easy switching between providers:

```python
llm = ChoreographyLLM(provider="anthropic")  # or "ollama"
recommendation = llm.generate_recommendation(analysis, moves)
```

This makes it trivial to:
- Switch providers via UI
- Add new providers (HuggingFace, etc.)
- A/B test different models
- Use custom fine-tuned models

### Essentia Audio Features

Currently extracting:
- BPM (tempo)
- Beat positions
- Energy level
- Danceability
- Mood estimates
- Music structure segments

Future: Could add more features like key detection, onset density, spectral features for even better choreography matching.

---

**Last Updated:** 2025-10-19
**Next Steps:** Test with real audio, implement move editing UI
