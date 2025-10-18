# Reachy Mini Choreography Builder

**Version:** 1.0.0
**Created:** October 2025
**Author:** Carson (LAURA Project Beta Tester)

A comprehensive web-based interface for controlling Reachy Mini, testing moves, and creating choreographies with an intuitive visual interface.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Interface Guide](#interface-guide)
- [Creating Choreographies](#creating-choreographies)
- [Troubleshooting](#troubleshooting)
- [Tips & Best Practices](#tips--best-practices)
- [Technical Details](#technical-details)
- [Contributing](#contributing)

---

## Overview

The Choreography Builder is a browser-based control panel that provides:

- **Manual Control**: Precise head position and antenna control with real-time feedback
- **Move Testing**: One-click access to 101 pre-recorded moves (20 dances, 81 emotions)
- **Choreography Creation**: Build, test, and export custom choreography sequences
- **Live Monitoring**: Real-time 3D simulator view and pose data visualization
- **Visual Feedback**: Color-coded sliders with safety zones using Pollen Robotics branding

All controls use the official [Pollen Robotics color palette](https://www.pollen-robotics.com):
- 🔵 **Navy Blue** (#2B4C7E) - Primary brand color
- 🔵 **Light Blue** (#3bb0d1) - Safe zones, info
- 💚 **Light Green** (#3dde99) - Success, connected
- 💛 **Yellow Gold** (#ffc261) - Accents, highlights
- 🩷 **Pink** (#ff6170) - Danger zones, errors
- ⚪ **White** (#FFFFFF) - Text, backgrounds

---

## Features

### 1. Manual Position Control
- **6 degrees of freedom**: X, Y, Z position + Yaw, Pitch, Roll rotation
- **Dual control modes**: Sliders for quick adjustment, number inputs for precision
- **Visual safety zones**: Color gradients indicate safe (blue) and danger (pink) ranges
- **Synchronized controls**: Optional yaw binding, antenna binding (normal/inverse)
- **Quick look directions**: One-click pad for instant head positioning

### 2. Pre-Recorded Move Library
- **101 total moves** dynamically loaded from `moves.json`:
  - **20 Dance Moves**: `side_to_side_sway`, `jackson_square`, `dizzy_spin`, etc.
  - **81 Emotion Moves**: `amazed1`, `cheerful1`, `frustrated1`, `loving1`, etc.
- **Radio selection**: Choose one move at a time
- **Instant execution**: Click "Execute Move" to play immediately
- **Stop control**: Cancel moves mid-execution

### 3. Choreography Builder
- **Drag-and-drop sequencing**: Build complex routines from individual moves
- **BPM configuration**: Set tempo (40-200 BPM) for choreography timing
- **Per-move parameters**: Configure cycles (1-10) and amplitude (0.1-2.0)
- **Visual routine editor**: See your sequence at a glance
- **JSON export**: Export choreographies in official Reachy Mini format
- **Clear/reset**: Start fresh anytime

### 4. Live Monitoring
- **3D Simulator Feed**: Real-time MJPEG stream from MuJoCo simulator
- **Live Pose Chart**: WebSocket-powered Chart.js visualization showing:
  - X, Y, Z position (mm)
  - Yaw, Pitch, Roll rotation (degrees)
  - Left & Right antenna position
- **Connection status**: Visual indicators for video and WebSocket status
- **Color-coded axes**: Each dimension has distinct color for easy reading

### 5. Quick Presets
- **Look Up**: Quick upward head tilt
- **Look Down**: Quick downward head tilt
- **Look Left**: Quick left turn
- **Look Right**: Quick right turn
- **Reset**: Return to neutral position

---

## Quick Start

### Prerequisites

1. **Reachy Mini daemon must be running**:

```bash
cd /Users/lauras/Desktop/laura/reachy_mini
source /Users/lauras/Desktop/laura/venv/bin/activate

# For simulation (macOS):
mjpython -m reachy_mini.daemon.app.main --sim --scene minimal --fastapi-port 8100

# For real robot:
python -m reachy_mini.daemon.app.main --fastapi-port 8100
```

2. **Verify daemon is running**:

```bash
curl http://localhost:8100/api/daemon/status
```

Expected response: `{"status":"running"}`

### Launch the Interface

**Option 1 - Open in browser:**
```bash
open /Users/lauras/Desktop/laura/reachy_mini/move_controller.html
```

**Option 2 - Drag and drop:**
Drag `move_controller.html` into any modern web browser (Chrome, Firefox, Safari, Edge).

**Option 3 - HTTP Server (recommended for best compatibility):**
```bash
cd /Users/lauras/Desktop/laura/reachy_mini
python -m http.server 8080
# Then open: http://localhost:8080/move_controller.html
```

### Verify Connection

Once opened, you should see:
- ✅ **Video Status**: "Connected - Live feed from simulator" (green)
- ✅ **Chart Status**: "Connected - Live data stream active" (green)
- ✅ **Status Messages**: "Ready" or "Connected successfully"

---

## Interface Guide

The interface is organized into two columns:

### Left Column: Monitoring & Quick Controls

#### 📹 3D Simulator View
- **Live video feed** from MuJoCo simulator (500px tall)
- **Connection status** indicator (green when connected)
- **Updates continuously** while daemon is running

#### 📊 Controls & Data (Side-by-Side)

**👁️ Quick Look Direction:**
- **9-button directional pad** for instant head positioning
- **Center button**: Reset to neutral
- **Edge buttons**: Look in that direction (up, down, left, right, diagonals)

**📈 Live Head Pose Chart:**
- **Real-time bar chart** showing all 8 axes:
  - X (pink), Y (yellow), Z (light blue)
  - Yaw (light green), Pitch (pink), Roll (yellow)
  - Ant L (light blue), Ant R (light green)
- **WebSocket-powered**: Updates multiple times per second
- **Color-coded**: Each axis has unique color from Pollen palette

### Right Column: Control Panels

#### 🎮 Manual Position Control

**Position Sliders (X, Y, Z):**
- **X**: Left/Right movement (-40mm to 40mm)
- **Y**: Forward/Back movement (-40mm to 40mm)
- **Z**: Up/Down movement (-40mm to 40mm)
- **Color gradient**: Blue (safe) to pink (extreme)

**Rotation Sliders (Yaw, Pitch, Roll):**
- **Yaw**: Left/Right rotation (-60° to 60°)
- **Pitch**: Up/Down tilt (-60° to 60°)
- **Roll**: Side-to-side tilt (-60° to 60°)
- **Yaw binding**: Optional synchronized yaw control (L/R move together with delta)

**Antenna Sliders:**
- **Left Antenna**: -3 to 3 range
- **Right Antenna**: -3 to 3 range (visually reversed)
- **Color gradient**: Blue (safe, -3 to 1) to pink (danger, 1 to 3)
- **Bind Ant**: Both antennae move to same value
- **Inv Ant**: Antennae mirror each other (inverse values)
- **Mutual exclusion**: Only one binding mode active at a time

**Options:**
- ☑️ **Degrees**: Use degrees instead of radians for rotation
- **Duration**: Movement duration in seconds (default 0.2s)

**Execute Button (Navy):**
- Sends current position/rotation to daemon
- Moves robot smoothly over specified duration

#### 🎭 Pre-Recorded Moves

**Left Panel - Dance Moves (20):**
- Alphabetically sorted choreographed movements
- Radio button selection
- Examples: `dizzy_spin`, `groovy_sway_and_roll`, `jackson_square`

**Right Panel - Emotion Moves (81):**
- Alphabetically sorted expressive poses
- Radio button selection
- Examples: `cheerful1`, `loving1`, `surprised2`, `thoughtful1`

**Controls:**
- **Execute Move (Light Blue)**: Play selected move immediately
- **Stop (Pink)**: Cancel running move mid-execution
- **Status indicator**: Shows execution status and UUID

#### 🎵 Choreography Builder

**Routine Section:**
- **Current routine displayed** as ordered list
- **Each entry shows**: Move name, cycles, amplitude
- **Clear button (Pink)**: Remove all moves from routine

**Add to Routine:**
1. Select move from Pre-Recorded Moves
2. Set **Cycles** (1-10, default 4): Number of repetitions
3. Set **Amplitude** (0.1-2.0, default 1.0): Movement intensity
4. Click **"Add to Routine" (Yellow)**
5. Move appears in routine list

**BPM Configuration:**
- **Input field**: Set beats per minute (40-200, default 120)
- **Affects timing**: Higher BPM = faster choreography

**Export Choreography:**
- Click **"💾 Export JSON" (Yellow)**
- Downloads `choreography_{bpm}bpm.json`
- **Format**: Official Reachy Mini choreography specification
- **Ready to use**: Can be played back via Python SDK or REST API

---

## Creating Choreographies

### Basic Workflow

1. **Set your BPM** (e.g., 120 for moderate tempo)
2. **Select a move** from Dance or Emotion library
3. **Configure parameters**:
   - **Cycles**: How many times to repeat (default 4)
   - **Amplitude**: How intense (1.0 = normal, 1.5 = more expressive)
4. **Click "Add to Routine"**
5. **Repeat** for each move in your sequence
6. **Review** your routine in the list
7. **Export** as JSON when ready

### Example: Simple Greeting Sequence

```
1. Select "welcoming1" emotion
   - Cycles: 2
   - Amplitude: 1.2
   - Add to Routine

2. Select "cheerful1" emotion
   - Cycles: 3
   - Amplitude: 1.0
   - Add to Routine

3. Select "yes1" emotion
   - Cycles: 2
   - Amplitude: 1.0
   - Add to Routine

4. Set BPM to 100
5. Click "Export JSON"
```

### Exported JSON Format

```json
{
    "bpm": 100,
    "sequence": [
        {
            "move": "welcoming1",
            "cycles": 2,
            "amplitude": 1.2
        },
        {
            "move": "cheerful1",
            "cycles": 3,
            "amplitude": 1.0
        },
        {
            "move": "yes1",
            "cycles": 2,
            "amplitude": 1.0
        }
    ]
}
```

### Playing Back Choreographies

**Via Python SDK:**
```python
from reachy_mini import ReachyMini

with ReachyMini() as reachy:
    reachy.play_choreography('choreography_100bpm.json')
```

**Via REST API:**
```bash
curl -X POST http://localhost:8100/api/choreography/play \
  -H "Content-Type: application/json" \
  -d @choreography_100bpm.json
```

---

## Troubleshooting

### Connection Issues

**Symptom**: "Connection error" or "Disconnected" status

**Solutions**:
1. Verify daemon is running:
   ```bash
   curl http://localhost:8100/api/daemon/status
   ```
2. Check server URL in interface matches daemon port (default: `http://localhost:8100`)
3. Ensure no firewall blocking localhost connections
4. Try restarting the daemon
5. Check browser console (F12) for detailed errors

### Video Feed Issues

**Symptom**: Video shows "Disconnected" or no image

**Solutions**:
1. Verify simulator is running (`--sim` flag)
2. Check MJPEG stream endpoint: `http://localhost:8100/api/camera/stream.mjpg`
3. Try refreshing the browser page
4. macOS users: Ensure using `mjpython` not regular `python`

### WebSocket Chart Issues

**Symptom**: "Connecting WebSocket..." never changes to "Connected"

**Solutions**:
1. Verify daemon is running
2. Check WebSocket endpoint: `ws://localhost:8100/api/state/stream`
3. Some browsers block mixed content (HTTP/WS) - ensure both are non-secure
4. Check browser console for WebSocket errors
5. Try a different browser (Chrome recommended)

### Move Execution Issues

**Symptom**: Clicking "Execute Move" does nothing

**Solutions**:
1. Ensure a move is selected (radio button checked)
2. Check browser console for API errors
3. Verify daemon is responsive:
   ```bash
   curl http://localhost:8100/api/daemon/status
   ```
4. Try stopping any running moves first
5. Restart daemon if frozen

### Manual Control Issues

**Symptom**: Sliders move but robot doesn't respond

**Solutions**:
1. Check "Execute" button was clicked (sliders don't auto-send)
2. Verify duration is reasonable (0.1-5.0 seconds)
3. Check if robot is already in motion (wait for completion)
4. Verify position values are within safe ranges
5. Try "Reset" button to return to neutral

### Export Issues

**Symptom**: Export button does nothing or downloads empty file

**Solutions**:
1. Ensure routine has at least one move
2. Check browser's download settings (popups allowed)
3. Verify browser supports Blob API (all modern browsers do)
4. Try a different browser
5. Check browser console for JavaScript errors

---

## Tips & Best Practices

### Manual Control

- **Start small**: Test with small movements (5-10mm, 5-10°) before larger ones
- **Use duration wisely**: Slower movements (1-2s) are smoother than fast (0.1s)
- **Watch the gradients**: Pink zones indicate extreme positions - use cautiously
- **Bind yaw carefully**: Delta-based binding can accumulate - reset periodically
- **Antenna safety**: Values approaching ±3 may stress motors - stay near 0-1 range

### Move Testing

- **Preview in list**: Move names are descriptive (e.g., `frustrated1` vs `cheerful1`)
- **Test individually**: Try moves alone before adding to choreographies
- **Stop if needed**: Don't wait for completion if move looks wrong
- **Note favorites**: Keep a list of moves that work well for your use case

### Choreography Building

- **Theme consistency**: Group similar moves (all dances or all emotions)
- **Vary amplitude**: Mix 0.8-1.2 for natural variation, avoid extremes
- **Cycles matter**: More cycles = longer performance time
- **BPM affects timing**: Higher BPM means moves execute faster
- **Test before exporting**: Add a few moves, export, test, then build full routine
- **Save multiple versions**: Export variations with different BPM/parameters

### Performance Optimization

- **Close unused tabs**: Video/WebSocket streams use bandwidth
- **Use HTTP server**: File:// protocol has limitations, `python -m http.server` is better
- **Modern browser**: Chrome 90+ or Firefox 88+ recommended
- **Stable connection**: Wired Ethernet preferred over WiFi for daemon communication

### Color Coding Reference

- 🔵 **Blue slider zones**: Safe operating range
- 🩷 **Pink slider zones**: Extreme positions, use carefully
- 💚 **Green status**: Connected, success, ready
- 🩷 **Pink status**: Error, disconnected, warning
- 💛 **Yellow status**: Info, processing, neutral

---

## Technical Details

### Files Structure

```
/Users/lauras/Desktop/laura/reachy_mini/
├── move_controller.html          # Main interface (1700+ lines)
├── moves.json                     # Move library (101 moves)
├── CHOREOGRAPHY_BUILDER_README.md # This file
├── DEVELOPMENT.md                 # Technical documentation
└── CHANGELOG.md                   # Version history
```

### API Endpoints Used

**Manual Control:**
```
POST /api/joints/target
Body: {
  "head": [x, y, z, yaw, pitch, roll],
  "ant_l": float,
  "ant_r": float,
  "duration": float
}
```

**Execute Move:**
```
POST /api/move/play/recorded-move-dataset/{dataset}/{move_name}
Returns: {"uuid": "..."}
```

**Stop Move:**
```
POST /api/move/stop
Body: {"uuid": "..."}
```

**Video Stream:**
```
GET /api/camera/stream.mjpg
Returns: MJPEG stream
```

**State WebSocket:**
```
WS /api/state/stream
Sends JSON every 100ms with full robot state
```

### Move Libraries

**Dance Library:**
```
pollen-robotics/reachy-mini-dances-library
```
20 moves: `stumble_and_recover`, `chin_lead`, `head_tilt_roll`, `jackson_square`, `pendulum_swing`, `side_glance_flick`, `grid_snap`, `simple_nod`, `side_to_side_sway`, `polyrhythm_combo`, `interwoven_spirals`, `uh_huh_tilt`, `chicken_peck`, `yeah_nod`, `headbanger_combo`, `side_peekaboo`, `dizzy_spin`, `neck_recoil`, `groovy_sway_and_roll`, `sharp_side_tilt`

**Emotion Library:**
```
pollen-robotics/reachy-mini-emotions-library
```
81 moves: `amazed1`, `anxiety1`, `attentive1`, `attentive2`, `boredom1`, `boredom2`, `calming1`, `cheerful1`, `come1`, `confused1`, `contempt1`, `curious1`, `dance1`, `dance2`, `dance3`, `disgusted1`, `displeased1`, `displeased2`, `downcast1`, `dying1`, `electric1`, `enthusiastic1`, `enthusiastic2`, `exhausted1`, `fear1`, `frustrated1`, `furious1`, `go_away1`, `grateful1`, `helpful1`, `helpful2`, `impatient1`, `impatient2`, `incomprehensible2`, `indifferent1`, `inquiring1`, `inquiring2`, `inquiring3`, `irritated1`, `irritated2`, `laughing1`, `laughing2`, `lonely1`, `lost1`, `loving1`, `no1`, `no_excited1`, `no_sad1`, `oops1`, `oops2`, `proud1`, `proud2`, `proud3`, `rage1`, `relief1`, `relief2`, `reprimand1`, `reprimand2`, `reprimand3`, `resigned1`, `sad1`, `sad2`, `scared1`, `serenity1`, `shy1`, `sleep1`, `success1`, `success2`, `surprised1`, `surprised2`, `thoughtful1`, `thoughtful2`, `tired1`, `uncertain1`, `uncomfortable1`, `understanding1`, `understanding2`, `welcoming1`, `welcoming2`, `yes1`, `yes_sad1`

### Browser Compatibility

**Fully Supported:**
- Chrome 90+ ✅
- Firefox 88+ ✅
- Safari 14+ ✅
- Edge 90+ ✅

**Required Features:**
- CSS Grid & Flexbox
- CSS Custom Properties (variables)
- Fetch API
- WebSocket API
- Blob API
- Chart.js 3.x

**Not Supported:**
- Internet Explorer (any version) ❌
- Very old mobile browsers ❌

### Dependencies

**Loaded from CDN:**
- Chart.js 3.9.1 - Live pose visualization

**No Build Required:**
- Pure HTML/CSS/JavaScript
- No npm, webpack, or compilation needed
- Edit HTML file directly to customize

### Customization

**Change Server URL:**
```javascript
// Line ~50 in move_controller.html
const serverUrl = 'http://localhost:8100';
```

**Change Port:**
Update daemon launch command:
```bash
mjpython -m reachy_mini.daemon.app.main --sim --fastapi-port YOUR_PORT
```

**Add Custom Moves:**
1. Record moves using Reachy Mini SDK
2. Upload to Hugging Face dataset
3. Edit `moves.json`:
   ```json
   {
     "dances": ["existing_moves", "your_new_dance"],
     "emotions": ["existing_moves", "your_new_emotion"]
   }
   ```
4. Refresh browser - moves auto-load and sort

**Modify Colors:**
Edit CSS custom properties in `:root`:
```css
:root {
    --pink: #ff6170;
    --yellow-gold: #ffc261;
    --light-blue: #3bb0d1;
    --light-green: #3dde99;
    --navy-blue: #2B4C7E;
    --white: #FFFFFF;
}
```

---

## Contributing

This tool was created by **Carson** (LAURA Project Beta Tester) for the Reachy Mini beta testing program.

### Reporting Issues

Found a bug or have a feature request?

1. Check existing issues: [Reachy Mini GitHub Issues](https://github.com/pollen-robotics/reachy_mini/issues)
2. Create new issue with:
   - Browser and OS version
   - Daemon version and launch command
   - Steps to reproduce
   - Expected vs actual behavior
   - Browser console errors (F12)

### Contributing Code

1. Fork the repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Test thoroughly in simulation and (if possible) on real robot
4. Ensure code follows existing style (see DEVELOPMENT.md)
5. Submit pull request with:
   - Clear description of changes
   - Screenshots/videos if UI changes
   - Testing performed

### Acknowledgments

- **Pollen Robotics Team** - Reachy Mini hardware and SDK
- **Hugging Face Team** - Move libraries and infrastructure
- **Chart.js Team** - Live visualization library
- **Beta Testing Community** - Feedback and testing

---

## License

This choreography builder interface is provided as-is for the Reachy Mini beta testing program.

Reachy Mini software is licensed under Apache 2.0 License.

For Pollen Robotics licensing details, see: [Reachy Mini License](https://github.com/pollen-robotics/reachy_mini/blob/main/LICENSE)

---

## Additional Resources

- **Reachy Mini Documentation**: https://docs.pollen-robotics.com/reachy-mini/
- **Python SDK Guide**: https://github.com/pollen-robotics/reachy_mini/blob/main/docs/python-sdk.md
- **REST API Reference**: https://github.com/pollen-robotics/reachy_mini/blob/main/docs/rest-api.md
- **Choreography Format**: https://docs.pollen-robotics.com/sdk/first-moves/orchestration/choreography/
- **Pollen Robotics Website**: https://www.pollen-robotics.com
- **Community Forum**: https://forum.pollen-robotics.com

---

**Questions or feedback?** Reach out to the Pollen Robotics team or beta testing community!

**Happy choreographing! 🎭🤖**
