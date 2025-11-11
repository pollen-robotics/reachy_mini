# Reachy Mini Choreography Builder

**Created:** October 16, 2025 (Initial Version)
**Updated:** October 18, 2025 (Major Expansion - v1.0.0)
**Purpose:** Comprehensive web-based control interface for Reachy Mini with manual control, move testing, and choreography creation

## Overview

The Choreography Builder is a full-featured browser-based control panel for Reachy Mini that provides:

- **Manual Position Control** - Precise 6-DOF head and antenna control with visual safety zones
- **Pre-Recorded Move Library** - One-click access to 101 moves (20 dances, 81 emotions)
- **Choreography Builder** - Create, test, and export custom choreography sequences
- **Live Monitoring** - Real-time 3D simulator view and pose data visualization
- **Professional Branding** - Official Pollen Robotics color palette throughout

No Python coding required - everything is accessible through an intuitive visual interface.

## Key Features

### 🎮 Manual Position Control
- 6 degrees of freedom (X, Y, Z position + Yaw, Pitch, Roll rotation)
- Dual control modes (sliders + number inputs)
- Visual safety zones (blue = safe, pink = danger)
- Antenna control with normal and inverse binding
- Quick look direction presets
- Configurable movement duration

### 🎭 Pre-Recorded Moves
- **101 total moves** dynamically loaded from JSON:
  - 20 dance moves (side_to_side_sway, jackson_square, dizzy_spin, etc.)
  - 81 emotion moves (amazed1, anxiety1, confused1, frustrated1, etc.)
- Radio button selection
- Execute and Stop controls
- Real-time status feedback

### 🎵 Choreography Creation
- Build custom routines from move library
- Configure BPM (beats per minute) for timing
- Set cycles and amplitude per move
- Export to official Reachy Mini JSON format
- Ready to play via Python SDK or REST API

### 📹 Live Monitoring
- 3D simulator video feed (MJPEG stream)
- Real-time pose chart (WebSocket powered)
- 8-axis visualization with color coding
- Connection status indicators

## File Structure

```
/Users/lauras/Desktop/laura/reachy_mini/
├── move_controller.html                  # Main choreography builder interface
├── moves.json                            # Move library definitions (101 moves)
├── CHOREOGRAPHY_BUILDER_README.md        # User guide for beta testers
├── DEVELOPMENT.md                        # Technical documentation for developers
├── CHANGELOG.md                          # Version history and design decisions
└── examples/                             # Example choreography JSON files
    ├── README.md                         # Guide to example choreographies
    ├── simple_greeting.json              # Basic 3-move greeting sequence
    ├── dance_party.json                  # 8-move high-energy dance routine
    ├── emotional_journey.json            # 9-move emotional storytelling
    ├── subtle_conversation.json          # 8-move conversational behaviors
    └── energetic_performance.json        # 9-move maximum intensity performance
```

## Prerequisites

**The Reachy Mini daemon must be running:**

```bash
cd /Users/lauras/Desktop/laura/reachy_mini
source /Users/lauras/Desktop/laura/venv/bin/activate
mjpython -m reachy_mini.daemon.app.main --sim --scene minimal --fastapi-port 8100
```

## Usage

### 1. Launch the Interface

```bash
open /Users/lauras/Desktop/laura/reachy_mini/move_controller.html
```

Or drag the file into any web browser.

### 2. Select a Move

- **Left panel:** Dance moves (choreographed movements, rhythmic)
- **Right panel:** Emotion moves (expressive poses, character-driven)

Click the radio button next to the desired move.

### 3. Execute

Click the **"Execute Move"** button. The move will:
- Start immediately on the robot
- Display status feedback
- Return a UUID for tracking

### 4. Stop (Optional)

Click **"Stop"** to cancel a running move mid-execution.

## Technical Details

### Move Library (moves.json)

The interface dynamically loads all moves from `moves.json`, which contains two arrays:

```json
{
  "dances": ["move1", "move2", ...],
  "emotions": ["emotion1", "emotion2", ...]
}
```

- Moves are automatically sorted alphabetically when loaded
- Counts update dynamically based on array length
- No HTML editing required to add/remove moves
- Easy to maintain and version control

### API Endpoints Used

**Execute move:**
```
POST /api/move/play/recorded-move-dataset/{dataset}/{move_name}
```

**Stop move:**
```
POST /api/move/stop
Body: {"uuid": "..."}
```

### Move Libraries

**Dance Library:**
```
pollen-robotics/reachy-mini-dances-library
```
Contains: stumble_and_recover, chin_lead, head_tilt_roll, jackson_square, pendulum_swing, side_glance_flick, grid_snap, simple_nod, side_to_side_sway, polyrhythm_combo, interwoven_spirals, uh_huh_tilt, chicken_peck, yeah_nod, headbanger_combo, side_peekaboo, dizzy_spin, neck_recoil, groovy_sway_and_roll, sharp_side_tilt

**Emotion Library:**
```
pollen-robotics/reachy-mini-emotions-library
```
Contains: amazed1, anxiety1, attentive1, attentive2, boredom1, boredom2, calming1, cheerful1, come1, confused1, contempt1, curious1, dance1, dance2, dance3, disgusted1, displeased1, displeased2, downcast1, dying1, electric1, enthusiastic1, enthusiastic2, exhausted1, fear1, frustrated1, furious1, go_away1, grateful1, helpful1, helpful2, impatient1, impatient2, incomprehensible2, indifferent1, inquiring1, inquiring2, inquiring3, irritated1, irritated2, laughing1, laughing2, lonely1, lost1, loving1, no1, no_excited1, no_sad1, oops1, oops2, proud1, proud2, proud3, rage1, relief1, relief2, reprimand1, reprimand2, reprimand3, resigned1, sad1, sad2, scared1, serenity1, shy1, sleep1, success1, success2, surprised1, surprised2, thoughtful1, thoughtful2, tired1, uncertain1, uncomfortable1, understanding1, understanding2, welcoming1, welcoming2, yes1, yes_sad1

### Response Format

**Successful execution returns:**
```json
{
  "uuid": "550e8400-e29b-41d4-a716-446655440000"
}
```

This UUID can be used to stop the move or check its status.

## Customization

### Change Server Port

If the daemon is running on a different port, update the server URL in the interface:

```
Default: http://localhost:8100
Change to: http://localhost:YOUR_PORT
```

### Add Custom Moves

To add custom recorded moves:

1. Record new moves using the SDK
2. Upload to a Hugging Face dataset
3. Edit `moves.json` and add the move name to either the `dances` or `emotions` array
4. Refresh the web interface - moves will automatically load and sort alphabetically

**Example:**
```json
{
  "dances": [
    "my_new_dance",
    "another_cool_move",
    ...
  ],
  "emotions": [
    "happy_wave",
    ...
  ]
}
```

No need to edit the HTML - all move management is now done through `moves.json`!

## Color Palette (Pollen Robotics Branding)

The interface uses the official Pollen Robotics color palette from their logo:

- **Navy Blue** (#2B4C7E) - Primary brand color, headers, primary buttons
- **Light Blue** (#3bb0d1) - Safe zones, info messages, secondary actions
- **Light Green** (#3dde99) - Success states, connected indicators
- **Yellow Gold** (#ffc261) - Highlights, accents, info states
- **Pink** (#ff6170) - Danger zones, errors, warnings
- **White** (#FFFFFF) - Text, backgrounds

All colors are managed via CSS custom properties for easy theming.

## Slider Safety Zones

Visual feedback using color gradients:

- **Position sliders (X, Y, Z):** Blue center → pink extremes
- **Rotation sliders:** Pink extremes ← blue center → pink extremes (symmetric)
- **Antenna sliders:** Blue safe zone (-3 to 1) → pink danger zone (1 to 3)
- **Right antenna:** Visually reversed using CSS `scaleX(-1)` transform (no value negation)

## Binding Mechanisms

### Yaw Binding
- Synchronizes left and right yaw values
- Delta-based: maintains relative offset
- Checkbox: "Bind yaw"

### Antenna Bindings
- **Normal binding:** Both antennae move to same value (symmetrical)
- **Inverse binding:** Antennae mirror each other (opposite values)
- **Mutual exclusion:** Only one binding mode active at a time
- Checkboxes: "Bind ant" and "Inv ant"

## Technical Implementation

### Architecture
- Single HTML file (~1700 lines) - zero build step required
- Pure vanilla JavaScript (ES6+) - no framework dependencies
- CSS Custom Properties for theming
- External dependency: Chart.js 3.9.1 (CDN) for visualization

### API Integration
- **Manual control:** POST /api/joints/target
- **Move execution:** POST /api/move/play/recorded-move-dataset/{dataset}/{move}
- **Move stop:** POST /api/move/stop
- **Video stream:** GET /api/camera/stream.mjpg
- **Real-time state:** WebSocket ws://localhost:8100/api/state/stream

### Choreography Export
- Client-side JSON generation using Blob API
- Official Reachy Mini choreography format
- Downloads as `choreography_{bpm}bpm.json`
- Ready to play via Python SDK or REST API

## Known Limitations (v1.0.0)

- No mobile responsiveness (desktop browsers only)
- No WebSocket auto-reconnection (requires page refresh)
- No undo/redo for choreography editing
- No move preview (names only, no visual)
- No choreography import (export only)
- No preset saving for manual positions
- No keyboard shortcuts
- No move search/filter
- No move reordering in routine (must clear and rebuild)

## Future Enhancements (Planned)

### v1.1
- [ ] Keyboard shortcuts (Space = execute, Esc = stop)
- [ ] Move search/filter functionality
- [ ] Undo/redo for choreography builder
- [ ] Routine move reordering (drag-and-drop)
- [ ] Preset saving/loading for manual positions
- [ ] WebSocket auto-reconnection
- [ ] Choreography import functionality

### v2.0
- [ ] Mobile responsive layout
- [ ] Touch gesture support
- [ ] Choreography timeline view
- [ ] Move preview on hover
- [ ] Custom move recording from UI
- [ ] TypeScript migration
- [ ] Automated testing

## Troubleshooting

**"Connection error" message:**
- Verify daemon is running: `curl http://localhost:8100/api/daemon/status`
- Check server URL matches daemon port
- Ensure no firewall blocking localhost

**Move doesn't execute:**
- Check browser console for errors (F12)
- Verify move name exists in library
- Restart daemon if unresponsive

**Simulator not responding:**
- Moves may be queued - wait for completion
- Use Stop button to cancel
- Restart daemon if frozen

## Documentation Files

All documentation is included in the repository for easy distribution:

- **CHOREOGRAPHY_BUILDER_README.md** - Comprehensive user guide for beta testers
  - Quick start guide with prerequisites
  - Detailed interface walkthrough
  - Choreography creation tutorial
  - Troubleshooting section
  - Tips & best practices

- **DEVELOPMENT.md** - Technical documentation for developers
  - Architecture overview
  - Color palette system details
  - Component breakdown
  - API integration guide
  - Code style guide
  - Extending the interface

- **CHANGELOG.md** - Version history and design decisions
  - Complete feature list
  - Design rationale
  - Technical challenges & solutions
  - Known limitations
  - Future roadmap

- **examples/README.md** - Guide to example choreographies
  - BPM, amplitude, and cycles guides
  - Choreography design tips
  - Testing procedures
  - Move library reference

## Example Choreographies

Five example choreographies demonstrate different moods and use cases:

1. **simple_greeting.json** (100 BPM, 3 moves)
   - Welcoming sequence for demos and onboarding
   - Moderate tempo, standard amplitudes

2. **dance_party.json** (140 BPM, 8 moves)
   - High-energy dance performance
   - Fast tempo, varied amplitudes (1.0-1.5)

3. **emotional_journey.json** (90 BPM, 9 moves)
   - Emotional storytelling with clear arc
   - Slow tempo, progression from confusion to success

4. **subtle_conversation.json** (80 BPM, 8 moves)
   - Natural conversation behaviors
   - Slow tempo, low amplitudes (0.6-0.9)

5. **energetic_performance.json** (160 BPM, 9 moves)
   - Maximum expressiveness showcase
   - Very fast tempo, high amplitudes (1.4-1.8)

## Related Reachy Mini Files

- `/examples/recorded_moves_example.py` - Python version of move playback
- `/examples/choreographies/` - Official choreography examples
- `/docs/rest-api.md` - Full REST API documentation
- `/docs/python-sdk.md` - Python SDK guide
- `/src/reachy_mini/motion/recorded_move.py` - Move loading implementation

## Development History

### October 16, 2025 - Initial Version
- Basic move controller with radio button selection
- Dynamic move loading from JSON
- Execute and stop controls

### October 18, 2025 - Major Expansion (v1.0.0)
- Added manual position control (6-DOF)
- Implemented antenna sliders with visual reversal
- Added binding mechanisms (yaw, antenna normal, antenna inverse)
- Implemented Pollen Robotics color palette throughout
- Added live monitoring (video feed, pose chart)
- Created choreography builder with BPM and export
- Optimized layout (side-by-side chart/look pad, taller video)
- Created comprehensive documentation suite
- Added 5 example choreography JSON files

### Design Challenges Solved

1. **Antenna Visual Reversal**
   - Challenge: Right antenna needed opposite visual direction
   - Solution: CSS `scaleX(-1)` for visual flip, no JavaScript value negation
   - Key insight: Transform flips everything (including gradients) automatically

2. **Color Palette Distribution**
   - Challenge: Too much of one color, not utilizing full palette
   - Solution: Individual button color classes, semantic color coding
   - Result: Vibrant, balanced, professionally branded interface

3. **Binding Logic**
   - Challenge: Multiple binding modes (yaw, antenna normal, antenna inverse)
   - Solution: Centralized logic in input handlers, mutual exclusion for conflicting modes
   - Result: Clean, predictable binding behavior

4. **Dynamic Move Loading**
   - Challenge: 101 moves in HTML was unmaintainable
   - Solution: External JSON with dynamic JavaScript generation
   - Benefit: Add/remove moves by editing one file, automatic sorting

---

## Submission to Pollen Robotics

This choreography builder was created by **Carson (LAURA Project Beta Tester)** for the Reachy Mini beta testing program.

**Package Contents for Submission:**
- `move_controller.html` - Main interface
- `moves.json` - Move library
- `CHOREOGRAPHY_BUILDER_README.md` - User guide
- `DEVELOPMENT.md` - Developer documentation
- `CHANGELOG.md` - Version history
- `examples/` directory with 5 choreography JSON files and README

**Target Audience:**
- Beta testers (use README for getting started)
- Developers (use DEVELOPMENT.md for technical details)
- Pollen Robotics team (use CHANGELOG for design decisions)

**License:** Apache 2.0 (aligns with Reachy Mini SDK)

---

*This interface was created to provide an intuitive, comprehensive control panel for Reachy Mini that eliminates the need for Python scripting while offering professional-grade features for choreography creation and robot control.*
