# Reachy Mini Beta Tester Contribution - Choreography System

**Contributor:** Carson (LAURA Project)
**Date:** October 2025
**Status:** Beta Feature - Community Contribution

---

## Overview

This contribution adds an AI-powered choreography generation system and enhanced desktop viewer interface to Reachy Mini. The system allows automatic generation of expressive robot choreography from any audio file using music analysis and LLM reasoning.

## What's Included

### 1. AI Choreography Generator (`choreography/`)
- **Automatic music structure detection** using Essentia and sklearn
- **ReAct agent architecture** (Claude Haiku 4.5) for artistic decision-making
- **Section-based workflow** that matches movement intensity to musical energy
- **Fast generation:** 2-3 LLM iterations (10-15 seconds total)
- **102 pre-recorded moves** (20 dances, 82 emotions) with intelligent selection

See `choreography/README.md` for detailed documentation.

### 2. Desktop Viewer (`desktop_viewer.py`)
- **Live 3D simulator visualization** via MuJoCo
- **Real-time pose monitoring** with WebSocket updates
- **Audio analysis integration** with segment visualization
- **Choreography generation UI** with one-click creation
- **Flask-based web interface** with ImGui controls

### 3. Web-Based Move Controller (`move_controller.html`)
- **Manual 6-DOF head control** with safety visualization
- **Pre-recorded move library browser** (101 moves)
- **Choreography builder** with BPM configuration
- **Professional Pollen Robotics branding**

---

## Quick Start

### Prerequisites

```bash
# Install additional dependencies for choreography system
pip install essentia scikit-learn soundfile anthropic

# Set Anthropic API key
export ANTHROPIC_API_KEY="your_key_here"
```

### Launch the Desktop Viewer

```bash
# Terminal 1: Start the daemon with simulator
mjpython -m reachy_mini.daemon.app.main --sim --scene minimal --fastapi-port 8100

# Terminal 2: Launch desktop viewer
python desktop_viewer.py
```

Then open your browser to `http://localhost:5000`

### Generate Choreography

1. Upload an audio file (.mp3, .wav, etc.) via the web interface
2. View automatic music structure analysis
3. Click "Generate Choreography"
4. AI creates a custom routine in 10-15 seconds
5. Review the move sequence and play on robot

---

## Known Issues & Apologies

### Menu System Needs Work

**Right-side menu is currently static:**
- Menu does not collapse or hide
- When not in fullscreen mode, the menu takes up approximately half the viewer width
- This significantly reduces the visible area for the 3D simulator view
- No toggle/minimize functionality implemented yet

**Impact:**
- Desktop users must use fullscreen mode for optimal viewing experience
- Smaller screens may find the interface cramped
- Menu cannot be repositioned or resized

**Planned Improvements:**
- Add collapsible menu with toggle button
- Implement responsive layout that adapts to screen size
- Add fullscreen mode detection to auto-collapse menu
- Consider floating/draggable menu panel option

### Other Known Limitations

- **Choreography import:** Export works, but cannot re-import/edit saved choreographies
- **Move preview:** No visual preview of moves (name-only display)
- **Mobile responsiveness:** Desktop-only interface (no touch/mobile support)
- **WebSocket reconnection:** Requires page refresh if connection drops
- **Undo/redo:** No undo functionality in choreography builder

---

## Technical Architecture

### Choreography Generation Pipeline

```
Audio File (.mp3)
    ↓
Essentia Analysis (BPM, energy, structure)
    ↓
sklearn Segmentation (intro/verse/chorus detection)
    ↓
ReAct Agent (Claude Haiku 4.5)
    ↓
Move Selection (energy-matched from 102-move library)
    ↓
Duration Solver (mathematical constraint satisfaction)
    ↓
Choreography JSON (ready to play)
```

### Desktop Viewer Components

- **Backend:** Flask server with ImGui integration
- **Frontend:** HTML/CSS/JavaScript with Chart.js visualization
- **Video Stream:** MJPEG from MuJoCo simulator (via daemon API)
- **Real-time Data:** WebSocket connection for pose updates
- **AI Integration:** Calls choreography system for generation

### Modified Daemon Files

The following daemon files were modified to support the choreography system:

- `src/reachy_mini/daemon/app/main.py` - Added routes
- `src/reachy_mini/daemon/app/routers/move.py` - Enhanced move playback
- `src/reachy_mini/daemon/backend/abstract.py` - Backend improvements
- `src/reachy_mini/daemon/backend/mujoco/backend.py` - MuJoCo integration enhancements
- `src/reachy_mini/daemon/daemon.py` - Core daemon updates

---

## File Structure

```
reachy_mini/
├── choreography/                      # AI choreography generation system
│   ├── README.md                      # Detailed choreography docs
│   ├── audio_analyzer.py              # Essentia integration
│   ├── segment_analyzer.py            # Music structure detection
│   ├── react_agent.py                 # ReAct choreographer
│   ├── react_tools.py                 # Agent tool registry
│   └── move_metadata_cache.py         # Move duration caching
│
├── desktop_viewer.py                  # Main desktop viewer application
├── move_controller.html               # Web-based move controller
├── moves.json                         # Move library definitions
└── BETA_CONTRIBUTION_README.md        # This file
```

---

## Usage Examples

### Generate Choreography via Python

```python
from choreography.audio_analyzer import AudioAnalyzer
from choreography.react_agent import ReActChoreographer

# Analyze audio
analyzer = AudioAnalyzer()
analysis = analyzer.analyze('music.mp3')

# Generate choreography
choreographer = ReActChoreographer(audio_analysis=analysis)
result = choreographer.generate()

# Result contains BPM and move sequence
print(result)
```

### Load and Play Choreography

```python
from reachy_mini import ReachyMini
import json

# Load choreography
with open('choreography.json') as f:
    choreo = json.load(f)

# Play on robot
reachy = ReachyMini()
reachy.play_choreography(choreo)
```

---

## Performance Characteristics

- **Analysis time:** 2-3 seconds (Essentia + sklearn)
- **Generation time:** 10-15 seconds first run, 5-8 seconds cached
- **Iteration count:** Typically 2-3 iterations (max 20)
- **Success rate:** >95% valid choreography on first submission
- **Cost per generation:** ~$0.01-0.02 (with Claude prompt caching)

---

## Dependencies

### New Python Packages

```
essentia>=2.1b6
scikit-learn>=1.3.0
soundfile>=0.12.1
anthropic>=0.34.0
flask>=2.3.0
imgui[full]>=2.0.0
```

### Existing Reachy Mini Dependencies

All standard Reachy Mini dependencies remain unchanged.

---

## Testing

### Test Full Choreography Workflow

```bash
python test_section_workflow.py
```

### Test Audio Analysis Only

```bash
cd choreography
python segment_analyzer.py /path/to/audio.mp3
```

### Test Desktop Viewer

```bash
# Start daemon first
mjpython -m reachy_mini.daemon.app.main --sim --scene minimal --fastapi-port 8100

# Run viewer
python desktop_viewer.py
```

---

## Future Improvements (Community Wishlist)

### High Priority
- [ ] Fix menu system (collapsible, responsive)
- [ ] Add choreography import/edit functionality
- [ ] Implement move preview visualization
- [ ] Add undo/redo support

### Medium Priority
- [ ] Mobile/tablet responsive layout
- [ ] WebSocket auto-reconnection
- [ ] Offline LLM support (Ollama integration)
- [ ] Multi-track choreography (verse 1 vs verse 2)

### Long Term
- [ ] Real-time choreography adjustment during performance
- [ ] Multi-robot synchronization
- [ ] Custom move recording from UI
- [ ] Style transfer (generate in style of example)

---

## Integration Notes for Pollen Robotics Team

### Potential Merge Considerations

This contribution modifies core daemon files. Suggested integration approach:

1. **Review daemon changes** - Some modifications may conflict with ongoing development
2. **Consider feature flag** - Allow choreography system to be enabled/disabled
3. **Dependencies** - Essentia and Anthropic add significant dependencies
4. **API key requirement** - Choreography requires Anthropic API key (not free)

### Alternative Integration Path

If full merge is not desired, this could be packaged as:
- **Standalone plugin/extension** separate from core repo
- **Beta tester tools** in a separate directory
- **Example/contrib** folder for community contributions

---

## Credits

**Developer:** Carson (LAURA Project Beta Tester)
**LLM Architecture:** Claude Sonnet 4.5 (design) + Claude Haiku 4.5 (execution)
**Audio Analysis:** Essentia (Music Technology Group, Universitat Pompeu Fabra)
**Machine Learning:** scikit-learn
**Robot Platform:** Reachy Mini by Pollen Robotics

**Special Thanks:**
- Pollen Robotics team for the beta testing opportunity
- Anthropic for Claude API and prompt caching
- Essentia team for comprehensive audio analysis tools

---

## License

This contribution follows the same license as the Reachy Mini SDK (Apache 2.0).

---

## Contact

For questions about this contribution:
- **GitHub:** LAURA-agent/reachy_mini (forked repo)
- **Integration questions:** Contact Pollen Robotics beta program
- **Technical issues:** Open issue in beta tester repository

**Note:** This is a beta community contribution and not officially supported by Pollen Robotics. Use at your own discretion during beta testing phase.

---

*Submitted with gratitude for the opportunity to contribute to the Reachy Mini ecosystem.*
