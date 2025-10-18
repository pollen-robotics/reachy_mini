# Reachy Mini Choreography Builder - Developer Documentation

**Version:** 1.0.0
**Last Updated:** October 2025
**Maintainer:** Carson (LAURA Project)

This document provides technical details for developers who want to understand, modify, or extend the Choreography Builder interface.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Technology Stack](#technology-stack)
- [Color Palette System](#color-palette-system)
- [Component Breakdown](#component-breakdown)
- [API Integration](#api-integration)
- [State Management](#state-management)
- [Slider System](#slider-system)
- [Binding Mechanisms](#binding-mechanisms)
- [Choreography Export](#choreography-export)
- [WebSocket Integration](#websocket-integration)
- [Chart Visualization](#chart-visualization)
- [Error Handling](#error-handling)
- [Performance Considerations](#performance-considerations)
- [Extending the Interface](#extending-the-interface)
- [Code Style Guide](#code-style-guide)

---

## Architecture Overview

### Single-File Application

The Choreography Builder is intentionally designed as a **single HTML file** (`move_controller.html`) containing:
- HTML structure
- CSS styling (inline `<style>` tag)
- JavaScript logic (inline `<script>` tag)

**Why Single-File?**
- Zero build step required
- Easy to distribute and version
- Works with `file://` protocol (though HTTP server recommended)
- Simple deployment (just copy one file + moves.json)
- No dependency management beyond CDN loads

### Component Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Browser Window                      │
├──────────────────────┬──────────────────────────────┤
│   Left Column        │      Right Column            │
│   (Monitoring)       │      (Controls)              │
├──────────────────────┼──────────────────────────────┤
│ ┌─────────────────┐  │  ┌─────────────────────────┐ │
│ │ Video Feed      │  │  │ Manual Position Control │ │
│ │ (MJPEG Stream)  │  │  │ - Position Sliders      │ │
│ └─────────────────┘  │  │ - Rotation Sliders      │ │
│                      │  │ - Antenna Sliders       │ │
│ ┌────────┬────────┐  │  │ - Binding Options       │ │
│ │ Look   │ Chart  │  │  └─────────────────────────┘ │
│ │ Pad    │ (Live) │  │                              │
│ └────────┴────────┘  │  ┌─────────────────────────┐ │
│                      │  │ Pre-Recorded Moves      │ │
│                      │  │ - Dances (20)           │ │
│                      │  │ - Emotions (81)         │ │
│                      │  └─────────────────────────┘ │
│                      │                              │
│                      │  ┌─────────────────────────┐ │
│                      │  │ Choreography Builder    │ │
│                      │  │ - Routine List          │ │
│                      │  │ - BPM Config            │ │
│                      │  │ - Export Function       │ │
│                      │  └─────────────────────────┘ │
└──────────────────────┴──────────────────────────────┘
```

### Data Flow

```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   UI Input   │─────▶│  JavaScript  │─────▶│ REST API     │
│  (Sliders,   │      │  (Validate & │      │ POST/GET     │
│   Buttons)   │      │   Format)    │      │              │
└──────────────┘      └──────────────┘      └──────┬───────┘
                                                    │
                                                    ▼
                      ┌──────────────┐      ┌──────────────┐
                      │  UI Update   │◀─────│ Reachy Mini  │
                      │  (Status,    │      │ Daemon       │
                      │   Feedback)  │      │ (FastAPI)    │
                      └──────────────┘      └──────────────┘

┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  WebSocket   │◀─────│  Daemon      │─────▶│  Chart.js    │
│  Stream      │      │  State       │      │  Rendering   │
│  (ws://)     │      │  Broadcast   │      │              │
└──────────────┘      └──────────────┘      └──────────────┘

┌──────────────┐      ┌──────────────┐
│  MJPEG       │◀─────│  MuJoCo      │
│  Video       │      │  Simulator   │
│  (<img src>) │      │  Camera      │
└──────────────┘      └──────────────┘
```

---

## Technology Stack

### Core Technologies

- **HTML5**: Structure and semantic markup
- **CSS3**: Styling with modern features
  - CSS Grid for layout
  - Flexbox for component alignment
  - Custom Properties (CSS Variables) for theming
  - Gradients for visual feedback
  - Transform for visual effects (scaleX flip)
- **JavaScript (ES6+)**: Logic and API communication
  - Fetch API for HTTP requests
  - WebSocket API for real-time data
  - Async/await for asynchronous operations
  - DOM manipulation
  - Event listeners

### External Dependencies

**Chart.js 3.9.1** (CDN):
```html
<script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
```
- Purpose: Real-time pose data visualization
- License: MIT
- Docs: https://www.chartjs.org/docs/3.9.1/

### Backend Dependencies

**Reachy Mini Daemon** (Python FastAPI):
- Required for all functionality
- Provides REST API and WebSocket endpoints
- Handles robot/simulator communication
- Documentation: https://github.com/pollen-robotics/reachy_mini

---

## Color Palette System

### CSS Custom Properties

The entire color scheme is managed via CSS variables defined in `:root`:

```css
:root {
    /* Primary Pollen Robotics Palette */
    --pink: #ff6170;          /* Danger, errors, warnings */
    --yellow-gold: #ffc261;   /* Highlights, info, accents */
    --light-blue: #3bb0d1;    /* Safe zones, primary actions */
    --light-green: #3dde99;   /* Success, connected states */
    --navy-blue: #2B4C7E;     /* Primary brand, headers */
    --white: #FFFFFF;         /* Text, backgrounds */

    /* Derived Darker Shades (for gradients/hover) */
    --navy-blue-dark: #1E3A5F;
    --navy-blue-darker: #153251;
    --light-blue-dark: #2a9dbd;
    --light-green-dark: #2bc985;
    --pink-dark: #e5475c;
}
```

### Color Usage Map

| UI Element | Color | CSS Variable | Purpose |
|------------|-------|--------------|---------|
| Headers (h2) | Navy Blue | `var(--navy-blue)` | Branding consistency |
| Section borders | Varies | All palette colors | Visual variety |
| Safe slider zones | Light Blue | `var(--light-blue)` | Indicates safe range |
| Danger slider zones | Pink | `var(--pink)` | Warns extreme values |
| Success status | Light Green | `var(--light-green)` | Connected, ready states |
| Error status | Pink | `var(--pink)` | Disconnected, errors |
| Info status | Yellow Gold | `var(--yellow-gold)` | Processing, neutral |
| Primary buttons | Navy Blue | `var(--navy-blue)` | Execute, Stop |
| Secondary buttons | Yellow Gold | `var(--yellow-gold)` | Export, Add |
| Action buttons | Light Blue | `var(--light-blue)` | Quick actions |
| Clear buttons | Pink | `var(--pink)` | Destructive actions |
| Chart X axis | Pink | `var(--pink)` | Distinct per axis |
| Chart Y axis | Yellow Gold | `var(--yellow-gold)` | Distinct per axis |
| Chart Z axis | Light Blue | `var(--light-blue)` | Distinct per axis |
| Chart Yaw | Light Green | `var(--light-green)` | Distinct per axis |
| Chart Pitch | Pink | `var(--pink)` | Distinct per axis |
| Chart Roll | Yellow Gold | `var(--yellow-gold)` | Distinct per axis |
| Chart Ant L | Light Blue | `var(--light-blue)` | Distinct per axis |
| Chart Ant R | Light Green | `var(--light-green)` | Distinct per axis |

### Button Color Classes

Individual button classes provide reusable styling:

```css
.btn-navy {
    background: linear-gradient(135deg, var(--navy-blue) 0%, var(--navy-blue-dark) 100%);
    color: var(--white);
    font-weight: 500;
    box-shadow: 0 2px 6px rgba(43, 76, 126, 0.3);
}

.btn-yellow {
    background: linear-gradient(135deg, var(--yellow-gold) 0%, #f0b554 100%);
    color: #2c3e50;
    font-weight: 500;
    box-shadow: 0 2px 6px rgba(255, 194, 97, 0.4);
}

.btn-blue {
    background: linear-gradient(135deg, var(--light-blue) 0%, var(--light-blue-dark) 100%);
    color: var(--white);
    font-weight: 500;
    box-shadow: 0 2px 6px rgba(59, 176, 209, 0.3);
}

.btn-pink {
    background: linear-gradient(135deg, var(--pink) 0%, var(--pink-dark) 100%);
    color: var(--white);
    font-weight: 500;
    box-shadow: 0 2px 6px rgba(255, 97, 112, 0.3);
}

.btn-green {
    background: linear-gradient(135deg, var(--light-green) 0%, var(--light-green-dark) 100%);
    color: var(--white);
    font-weight: 500;
    box-shadow: 0 2px 6px rgba(61, 222, 153, 0.3);
}
```

**Hover states** automatically darken via additional CSS:
```css
.btn-navy:hover { background: var(--navy-blue-dark); }
.btn-yellow:hover { background: #f0b554; }
/* etc... */
```

---

## Component Breakdown

### 1. Manual Position Control

**HTML Structure:**
```html
<div class="slider-row">
    <label title="Tooltip">Label:</label>
    <input type="range" id="sliderID" min="-40" max="40" step="0.2" value="0">
    <input type="number" step="0.01" id="inputID" value="0" min="-40" max="40">
</div>
```

**JavaScript Initialization:**
```javascript
function syncSliderAndInput(sliderId, inputId, isReversed = false) {
    const slider = document.getElementById(sliderId);
    const input = document.getElementById(inputId);

    slider.addEventListener('input', () => {
        let value = parseFloat(slider.value);
        input.value = value;
        // Binding logic here
    });

    input.addEventListener('input', () => {
        let value = parseFloat(input.value) || 0;
        const min = parseFloat(slider.min);
        const max = parseFloat(slider.max);
        value = Math.max(min, Math.min(max, value));
        slider.value = value;
        // Binding logic here
    });
}
```

**Key Features:**
- Two-way binding between slider and number input
- Value clamping to min/max ranges
- Optional binding to paired axes (yaw, antennas)
- Visual feedback via gradients

### 2. Pre-Recorded Moves

**Dynamic Loading from JSON:**
```javascript
fetch('moves.json')
    .then(response => response.json())
    .then(data => {
        const dances = data.dances.sort();
        const emotions = data.emotions.sort();

        // Populate dance list
        const danceList = document.getElementById('danceList');
        dances.forEach(move => {
            const label = document.createElement('label');
            label.style.display = 'block';
            label.innerHTML = `
                <input type="radio" name="prerecordedMove" value="dance::${move}">
                ${move}
            `;
            danceList.appendChild(label);
        });

        // Similar for emotions...
    });
```

**Radio Button Selection:**
- Only one move selectable at a time
- Value format: `"dance::move_name"` or `"emotion::move_name"`
- Extracted on execution to determine dataset

**Execution Logic:**
```javascript
document.getElementById('executeMoveBtn').onclick = async () => {
    const selected = document.querySelector('input[name="prerecordedMove"]:checked');
    if (!selected) {
        alert('Please select a move first');
        return;
    }

    const [dataset, moveName] = selected.value.split('::');
    const datasetPath = dataset === 'dance'
        ? 'pollen-robotics/reachy-mini-dances-library'
        : 'pollen-robotics/reachy-mini-emotions-library';

    const url = `${serverUrl}/api/move/play/recorded-move-dataset/${datasetPath}/${moveName}`;
    const response = await fetch(url, { method: 'POST' });
    const data = await response.json();

    currentMoveUUID = data.uuid;
    showStatus('moveStatus', `Executing: ${moveName} (UUID: ${data.uuid})`, 'success');
};
```

### 3. Choreography Builder

**Routine Data Structure:**
```javascript
let routine = []; // Array of move objects

// Example routine entry:
{
    move: "cheerful1",
    cycles: 4,
    amplitude: 1.0
}
```

**Adding to Routine:**
```javascript
document.getElementById('addToRoutineBtn').onclick = () => {
    const selected = document.querySelector('input[name="prerecordedMove"]:checked');
    if (!selected) {
        alert('Select a move first');
        return;
    }

    const [, moveName] = selected.value.split('::');
    const cycles = parseInt(document.getElementById('routineCycles').value) || 4;
    const amplitude = parseFloat(document.getElementById('routineAmplitude').value) || 1.0;

    routine.push({ move: moveName, cycles, amplitude });
    updateRoutineDisplay();
};
```

**Routine Display:**
```javascript
function updateRoutineDisplay() {
    const list = document.getElementById('routineList');
    list.innerHTML = '';

    routine.forEach((item, idx) => {
        const li = document.createElement('li');
        li.textContent = `${item.move} (×${item.cycles}, amp: ${item.amplitude})`;
        list.appendChild(li);
    });

    document.getElementById('routineCount').textContent = routine.length;
}
```

### 4. Video Feed

**MJPEG Stream Loading:**
```html
<img id="videoFeed" src="" alt="Simulator feed">
```

```javascript
const videoFeed = document.getElementById('videoFeed');
const videoStatus = document.getElementById('videoStatus');

videoFeed.src = `${serverUrl}/api/camera/stream.mjpg`;

videoFeed.onload = () => {
    videoStatus.textContent = 'Connected - Live feed from simulator';
    videoStatus.className = 'video-status connected';
};

videoFeed.onerror = () => {
    videoStatus.textContent = 'Disconnected - Check daemon status';
    videoStatus.className = 'video-status disconnected';
};
```

**Why MJPEG?**
- Simple implementation (just an `<img>` tag)
- Browser handles frame decoding automatically
- Low latency for local connections
- No JavaScript streaming logic needed

### 5. Look Pad (Quick Directions)

**9-Button Grid:**
```html
<div class="look-pad">
    <button class="look-btn btn-green" onclick="quickLook(-10, 10)">↖</button>
    <button class="look-btn btn-blue" onclick="quickLook(0, 10)">↑</button>
    <button class="look-btn btn-yellow" onclick="quickLook(10, 10)">↗</button>

    <button class="look-btn btn-pink" onclick="quickLook(-10, 0)">←</button>
    <button class="look-btn btn-navy" onclick="quickLook(0, 0)">●</button>
    <button class="look-btn btn-pink" onclick="quickLook(10, 0)">→</button>

    <button class="look-btn btn-green" onclick="quickLook(-10, -10)">↙</button>
    <button class="look-btn btn-blue" onclick="quickLook(0, -10)">↓</button>
    <button class="look-btn btn-yellow" onclick="quickLook(10, -10)">↘</button>
</div>
```

**Quick Look Function:**
```javascript
async function quickLook(yaw, pitch) {
    const url = `${serverUrl}/api/joints/target`;
    const body = {
        head: [0, 0, 0, yaw, pitch, 0],
        duration: 0.5
    };

    await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    });
}
```

---

## API Integration

### Server URL Configuration

```javascript
const serverUrl = 'http://localhost:8100';
```

**Change for different port:**
```javascript
const serverUrl = 'http://localhost:YOUR_PORT';
```

**Remote daemon:**
```javascript
const serverUrl = 'http://192.168.1.100:8100';
```

### Manual Control API

**Endpoint:** `POST /api/joints/target`

**Request Body:**
```json
{
    "head": [x, y, z, yaw, pitch, roll],
    "ant_l": float,
    "ant_r": float,
    "duration": float
}
```

**Example:**
```javascript
const body = {
    head: [10.5, -5.2, 15.0, 30, -15, 5],
    ant_l: 0.5,
    ant_r: -0.5,
    duration: 1.5
};

fetch(`${serverUrl}/api/joints/target`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
});
```

**Response:**
```json
{
    "status": "ok"
}
```

### Move Execution API

**Endpoint:** `POST /api/move/play/recorded-move-dataset/{dataset}/{move_name}`

**Example:**
```javascript
const dataset = 'pollen-robotics/reachy-mini-dances-library';
const move = 'dizzy_spin';

const response = await fetch(
    `${serverUrl}/api/move/play/recorded-move-dataset/${dataset}/${move}`,
    { method: 'POST' }
);

const data = await response.json();
// Returns: {"uuid": "550e8400-e29b-41d4-a716-446655440000"}
```

### Move Stop API

**Endpoint:** `POST /api/move/stop`

**Request Body:**
```json
{
    "uuid": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Example:**
```javascript
await fetch(`${serverUrl}/api/move/stop`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ uuid: currentMoveUUID })
});
```

### Video Stream

**Endpoint:** `GET /api/camera/stream.mjpg`

**Usage:**
```html
<img src="http://localhost:8100/api/camera/stream.mjpg">
```

No JavaScript needed - browser handles MJPEG automatically.

---

## State Management

### Global State Variables

```javascript
let routine = [];              // Choreography sequence
let currentMoveUUID = null;    // Currently executing move UUID
let ws = null;                 // WebSocket connection
let poseChart = null;          // Chart.js instance
```

**Why Global?**
- Simple state management for single-page app
- No framework overhead
- Easy debugging (accessible in console)
- Sufficient for this use case

### No Framework Decision

**Reasoning:**
- Small codebase (~1700 lines total)
- No complex state transitions
- Minimal component nesting
- Performance is excellent without framework
- Zero build step requirement
- Easy for beginners to understand and modify

**Tradeoffs:**
- Manual DOM manipulation
- No virtual DOM diffing
- Potential for code duplication
- Less structure than React/Vue

**Verdict:** For this tool, vanilla JavaScript is optimal.

---

## Slider System

### Gradient Implementation

**Position Sliders (X, Y, Z):**
```css
.slider-row input[type="range"]#gotoXSlider,
.slider-row input[type="range"]#gotoYSlider,
.slider-row input[type="range"]#gotoZSlider {
    background: linear-gradient(90deg,
        var(--light-blue) 0%,
        var(--light-blue) 50%,
        var(--pink) 100%
    );
}
```

**Rotation Sliders (Yaw, Pitch, Roll):**
```css
.slider-row input[type="range"]#gotoYawSlider,
.slider-row input[type="range"]#gotoPitchSlider,
.slider-row input[type="range"]#gotoRollSlider {
    background: linear-gradient(90deg,
        var(--pink) 0%,
        var(--light-blue) 30%,
        var(--light-blue) 70%,
        var(--pink) 100%
    );
}
```

**Antenna Sliders (Special Case):**
```css
/* Left antenna: danger on right side (approaching +3) */
.slider-row input[type="range"]#gotoAntLSlider {
    background: linear-gradient(90deg,
        var(--light-blue) 0%,
        var(--light-blue) 67%,
        var(--pink) 100%
    );
}

/* Right antenna: same gradient, flipped by scaleX(-1) */
.slider-row input[type="range"]#gotoAntRSlider {
    background: linear-gradient(90deg,
        var(--light-blue) 0%,
        var(--light-blue) 67%,
        var(--pink) 100%
    );
}
```

**Gradient Math:**
- Antenna range: -3 to 3 (total span: 6 units)
- Safe zone: -3 to 1 (4 units = 67% of range)
- Danger zone: 1 to 3 (2 units = 33% of range)
- Blend point: 67% corresponds to value 1

### Visual Reversal (Right Antenna)

**CSS Transform:**
```css
.reversed-slider {
    transform: scaleX(-1);
}
```

**What This Does:**
- Flips the slider horizontally
- Thumb moves left when value increases
- Gradient automatically flips too
- **Values remain unchanged** (no JavaScript negation)

**Critical Insight:**
- DO NOT negate values in JavaScript
- scaleX(-1) is purely visual
- Both sliders have min="-3" max="3"
- Only difference is CSS class `reversed-slider`

**Example:**
```html
<!-- Normal slider -->
<input type="range" id="gotoAntLSlider" min="-3" max="3" value="0">

<!-- Visually reversed slider -->
<input type="range" id="gotoAntRSlider" min="-3" max="3" value="0" class="reversed-slider">
```

When right slider is at value `2`:
- Visual position: Leftmost side of slider (flipped by scaleX)
- Actual value: `2` (not `-2`)
- Gradient: Blue on left, pink on right (flipped visually)

---

## Binding Mechanisms

### Yaw Binding (Delta-Based)

**Purpose:** Keep left and right yaw synchronized while allowing independent movement.

**Logic:**
```javascript
if (document.getElementById('bindYaw').checked) {
    if (inputId === 'gotoYawL') {
        const delta = value - parseFloat(document.getElementById('gotoYawR').value);
        document.getElementById('gotoYawR').value = value;
        document.getElementById('gotoYawRSlider').value = value;
    } else if (inputId === 'gotoYawR') {
        const delta = value - parseFloat(document.getElementById('gotoYawL').value);
        document.getElementById('gotoYawL').value = value;
        document.getElementById('gotoYawLSlider').value = value;
    }
}
```

**Behavior:**
- User adjusts one yaw slider
- Other yaw slider jumps to same value
- Delta is calculated (currently unused, but available for future)
- Both yaws now move together

### Antenna Binding (Normal)

**Purpose:** Both antennae move to the same value.

**Logic:**
```javascript
if (document.getElementById('bindAnt').checked) {
    if (inputId === 'gotoAntL') {
        document.getElementById('gotoAntR').value = value;
        document.getElementById('gotoAntRSlider').value = value;
    } else if (inputId === 'gotoAntR') {
        document.getElementById('gotoAntL').value = value;
        document.getElementById('gotoAntLSlider').value = value;
    }
}
```

**Behavior:**
- User adjusts left antenna to `1.5`
- Right antenna updates to `1.5`
- Both sliders show same absolute value
- Visual position differs due to scaleX(-1) on right

### Antenna Inverse Binding

**Purpose:** Antennae mirror each other (opposite values).

**Logic:**
```javascript
if (document.getElementById('invAnt').checked) {
    if (inputId === 'gotoAntL') {
        const invValue = -value;
        document.getElementById('gotoAntR').value = invValue;
        document.getElementById('gotoAntRSlider').value = invValue;
    } else if (inputId === 'gotoAntR') {
        const invValue = -value;
        document.getElementById('gotoAntL').value = invValue;
        document.getElementById('gotoAntLSlider').value = invValue;
    }
}
```

**Behavior:**
- User adjusts left antenna to `1.5`
- Right antenna updates to `-1.5`
- Creates mirrored/symmetrical antenna poses

### Mutual Exclusion

**Purpose:** Prevent both normal and inverse binding simultaneously.

**Logic:**
```javascript
document.getElementById('bindAnt').addEventListener('change', (e) => {
    if (e.target.checked) {
        document.getElementById('invAnt').checked = false;
    }
});

document.getElementById('invAnt').addEventListener('change', (e) => {
    if (e.target.checked) {
        document.getElementById('bindAnt').checked = false;
    }
});
```

**Behavior:**
- Checking "Bind ant" unchecks "Inv ant"
- Checking "Inv ant" unchecks "Bind ant"
- Only one can be active at a time
- Both can be unchecked (independent control)

---

## Choreography Export

### Export Function

```javascript
document.getElementById('exportChoreographyBtn').onclick = () => {
    if (routine.length === 0) {
        alert('Routine is empty. Add moves before exporting.');
        return;
    }

    const bpm = parseInt(document.getElementById('choreographyBPM').value) || 120;

    const choreography = {
        bpm: bpm,
        sequence: routine.map(item => ({
            move: item.move,
            cycles: item.cycles || 4,
            amplitude: item.amplitude || 1.0
        }))
    };

    const jsonStr = JSON.stringify(choreography, null, 4);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `choreography_${bpm}bpm.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    showStatus('choreographyStatus', `Exported choreography (${routine.length} moves, ${bpm} BPM)`, 'success');
};
```

### JSON Format

**Official Reachy Mini choreography specification:**
```json
{
    "bpm": 120,
    "sequence": [
        {
            "move": "move_name",
            "cycles": 4,
            "amplitude": 1.0
        },
        {
            "move": "another_move",
            "cycles": 2,
            "amplitude": 1.5
        }
    ]
}
```

**Parameters:**
- `bpm` (int): Beats per minute, controls timing (40-200 recommended)
- `sequence` (array): Ordered list of moves
  - `move` (string): Exact move name from library
  - `cycles` (int): Number of repetitions (1-10)
  - `amplitude` (float): Movement intensity (0.1-2.0, default 1.0)

### Blob Download Mechanism

**Why Blob API?**
- No server-side processing needed
- Works entirely in browser
- Creates downloadable file from JavaScript
- Clean UX (standard browser download)

**Step-by-Step:**
1. Convert routine array to JSON string with formatting
2. Create Blob from string with MIME type `application/json`
3. Generate temporary object URL from Blob
4. Create hidden `<a>` element with download attribute
5. Programmatically click the link
6. Clean up: remove element and revoke URL

**Browser Compatibility:**
- All modern browsers support Blob API
- IE 10+ (but we don't target IE anyway)

---

## WebSocket Integration

### Connection Establishment

```javascript
const ws = new WebSocket(`ws://localhost:8100/api/state/stream`);

ws.onopen = () => {
    console.log('WebSocket connected');
    document.getElementById('chartStatus').textContent = 'Connected - Live data stream active';
    document.getElementById('chartStatus').className = 'chart-status connected';
};

ws.onerror = (error) => {
    console.error('WebSocket error:', error);
    document.getElementById('chartStatus').textContent = 'Connection failed - Check daemon';
    document.getElementById('chartStatus').className = 'chart-status disconnected';
};

ws.onclose = () => {
    console.log('WebSocket closed');
    document.getElementById('chartStatus').textContent = 'Disconnected - Reconnecting...';
    document.getElementById('chartStatus').className = 'chart-status disconnected';
};
```

### Message Handling

```javascript
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);

    // Extract pose data
    const head = data.head || [0, 0, 0, 0, 0, 0];
    const [x, y, z, yaw, pitch, roll] = head;
    const antL = data.ant_l || 0;
    const antR = data.ant_r || 0;

    // Update chart
    updateChart(x, y, z, yaw, pitch, roll, antL, antR);
};
```

**Message Format from Daemon:**
```json
{
    "head": [x, y, z, yaw, pitch, roll],
    "ant_l": float,
    "ant_r": float,
    "timestamp": "2025-10-18T12:34:56.789Z"
}
```

**Update Frequency:**
- Daemon broadcasts ~10 messages per second
- JavaScript receives and processes all messages
- Chart updates on every message (throttled by browser)

### Reconnection Logic

**Current Implementation:**
- No automatic reconnection
- User must refresh page

**Future Enhancement:**
```javascript
function connectWebSocket() {
    ws = new WebSocket(`ws://localhost:8100/api/state/stream`);

    ws.onopen = () => { /* ... */ };
    ws.onerror = () => { /* ... */ };
    ws.onclose = () => {
        console.log('WebSocket closed, reconnecting in 3s...');
        setTimeout(connectWebSocket, 3000);
    };
    ws.onmessage = (event) => { /* ... */ };
}
```

---

## Chart Visualization

### Chart.js Setup

```javascript
const ctx = document.getElementById('poseChart').getContext('2d');

poseChart = new Chart(ctx, {
    type: 'bar',
    data: {
        labels: ['X', 'Y', 'Z', 'Yaw', 'Pitch', 'Roll', 'Ant L', 'Ant R'],
        datasets: [{
            label: 'Current Pose',
            data: [0, 0, 0, 0, 0, 0, 0, 0],
            backgroundColor: [
                'rgba(255, 97, 112, 0.7)',    // Pink - X
                'rgba(255, 194, 97, 0.7)',    // Yellow - Y
                'rgba(59, 176, 209, 0.7)',    // Light Blue - Z
                'rgba(61, 222, 153, 0.7)',    // Light Green - Yaw
                'rgba(255, 97, 112, 0.7)',    // Pink - Pitch
                'rgba(255, 194, 97, 0.7)',    // Yellow - Roll
                'rgba(59, 176, 209, 0.7)',    // Light Blue - Ant L
                'rgba(61, 222, 153, 0.7)'     // Light Green - Ant R
            ],
            borderColor: [
                'rgba(255, 97, 112, 1)',
                'rgba(255, 194, 97, 1)',
                'rgba(59, 176, 209, 1)',
                'rgba(61, 222, 153, 1)',
                'rgba(255, 97, 112, 1)',
                'rgba(255, 194, 97, 1)',
                'rgba(59, 176, 209, 1)',
                'rgba(61, 222, 153, 1)'
            ],
            borderWidth: 2
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
            duration: 100  // Fast updates for real-time feel
        },
        scales: {
            y: {
                beginAtZero: true,
                title: {
                    display: true,
                    text: 'Value'
                }
            }
        },
        plugins: {
            legend: {
                display: false
            },
            tooltip: {
                enabled: true
            }
        }
    }
});
```

### Chart Update Function

```javascript
function updateChart(x, y, z, yaw, pitch, roll, antL, antR) {
    if (!poseChart) return;

    poseChart.data.datasets[0].data = [
        x.toFixed(2),
        y.toFixed(2),
        z.toFixed(2),
        yaw.toFixed(2),
        pitch.toFixed(2),
        roll.toFixed(2),
        antL.toFixed(2),
        antR.toFixed(2)
    ];

    poseChart.update('none');  // 'none' mode = no animation, instant update
}
```

**Performance Optimization:**
- `animation.duration: 100` - Minimal animation time
- `update('none')` - Skip animation entirely for real-time updates
- Fixed precision (2 decimals) prevents label flickering

---

## Error Handling

### Status Message System

```javascript
function showStatus(elementId, message, type) {
    const element = document.getElementById(elementId);
    element.textContent = message;
    element.className = `status ${type}`;

    // Auto-clear after 5 seconds (except persistent states)
    if (type !== 'connected' && type !== 'disconnected') {
        setTimeout(() => {
            element.textContent = 'Ready';
            element.className = 'status';
        }, 5000);
    }
}
```

**Status Types:**
- `success` - Green, operation succeeded
- `error` - Pink, operation failed
- `info` - Yellow, processing or info
- `connected` - Green, persistent connection
- `disconnected` - Pink, persistent disconnection

### API Error Handling

```javascript
async function executeManualControl() {
    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
        showStatus('manualStatus', 'Movement executed successfully', 'success');

    } catch (error) {
        console.error('Error executing movement:', error);
        showStatus('manualStatus', `Error: ${error.message}`, 'error');
    }
}
```

**Error Categories:**
- Network errors (fetch failed)
- HTTP errors (4xx, 5xx responses)
- JSON parsing errors
- Daemon not running
- Invalid parameters

### User Input Validation

```javascript
// Clamp values to range
let value = parseFloat(input.value) || 0;
const min = parseFloat(slider.min);
const max = parseFloat(slider.max);
value = Math.max(min, Math.min(max, value));
input.value = value;
slider.value = value;
```

**Validation Points:**
- Number inputs clamped to min/max
- BPM limited to 40-200
- Cycles limited to 1-10
- Amplitude limited to 0.1-2.0
- Empty move selection caught before execution

---

## Performance Considerations

### Optimization Techniques

1. **Event Debouncing** (not currently implemented, but recommended):
```javascript
let debounceTimer;
slider.addEventListener('input', () => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
        // Process slider change
    }, 50);
});
```

2. **Chart Update Throttling**:
```javascript
poseChart.update('none');  // Skip animation
```

3. **WebSocket Message Handling**:
- No processing bottleneck (simple JSON parse + chart update)
- Browser naturally throttles render at 60 FPS

4. **MJPEG Stream**:
- Browser-native decoding (hardware accelerated)
- No JavaScript processing overhead

### Memory Management

**Current State:**
- Minimal memory footprint
- No memory leaks detected
- Routine array grows linearly (acceptable)

**Potential Issues:**
- Very long routines (1000+ moves) may slow down
- Long-running sessions with continuous WebSocket updates
- Chart.js retains some data internally

**Mitigation:**
- Clear routine when done
- Refresh page for multi-hour sessions
- Chart data is replaced, not appended

### Browser Compatibility

**Tested:**
- Chrome 90+ ✅ (Recommended)
- Firefox 88+ ✅
- Safari 14+ ✅ (macOS)
- Edge 90+ ✅

**Not Supported:**
- Internet Explorer (any version) ❌
- Very old mobile browsers ❌

**Required Features:**
- CSS Grid
- CSS Custom Properties
- Fetch API
- WebSocket API
- Blob API
- ES6+ JavaScript (const, let, arrow functions, template literals)

---

## Extending the Interface

### Adding New Manual Controls

**Example: Adding a "Speed" slider**

1. **Add HTML:**
```html
<div class="slider-row">
    <label title="Movement speed multiplier">Speed:</label>
    <input type="range" id="gotoSpeedSlider" min="0.1" max="2.0" step="0.1" value="1.0">
    <input type="number" step="0.1" id="gotoSpeed" value="1.0" min="0.1" max="2.0">
</div>
```

2. **Add CSS (if custom styling needed):**
```css
.slider-row input[type="range"]#gotoSpeedSlider {
    background: linear-gradient(90deg, var(--light-blue) 0%, var(--yellow-gold) 100%);
}
```

3. **Initialize in JavaScript:**
```javascript
syncSliderAndInput('gotoSpeedSlider', 'gotoSpeed');
```

4. **Use in execution:**
```javascript
const speed = parseFloat(document.getElementById('gotoSpeed').value);
const adjustedDuration = duration / speed;
```

### Adding New Move Libraries

**Steps:**

1. **Record moves** using Reachy Mini SDK
2. **Upload to Hugging Face** dataset
3. **Edit `moves.json`:**
```json
{
  "dances": [...],
  "emotions": [...],
  "custom": ["my_move1", "my_move2"]
}
```

4. **Update JavaScript loading:**
```javascript
const custom = data.custom.sort();
const customList = document.getElementById('customList');
custom.forEach(move => {
    const label = document.createElement('label');
    label.innerHTML = `
        <input type="radio" name="prerecordedMove" value="custom::${move}">
        ${move}
    `;
    customList.appendChild(label);
});
```

5. **Handle in execution:**
```javascript
const [category, moveName] = selected.value.split('::');
const datasetPath = category === 'custom'
    ? 'your-username/your-dataset-name'
    : category === 'dance'
        ? 'pollen-robotics/reachy-mini-dances-library'
        : 'pollen-robotics/reachy-mini-emotions-library';
```

### Adding Custom Export Formats

**Example: Export to Python script**

```javascript
document.getElementById('exportPythonBtn').onclick = () => {
    const bpm = parseInt(document.getElementById('choreographyBPM').value) || 120;

    let pythonScript = `from reachy_mini import ReachyMini\n\n`;
    pythonScript += `with ReachyMini() as reachy:\n`;
    pythonScript += `    # BPM: ${bpm}\n`;

    routine.forEach(item => {
        pythonScript += `    reachy.play_move("${item.move}", cycles=${item.cycles}, amplitude=${item.amplitude})\n`;
    });

    const blob = new Blob([pythonScript], { type: 'text/x-python' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `choreography_${bpm}bpm.py`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
};
```

### Adding Keyboard Shortcuts

**Example: Space to execute, Escape to stop**

```javascript
document.addEventListener('keydown', (e) => {
    if (e.code === 'Space') {
        e.preventDefault();
        document.getElementById('executeMoveBtn').click();
    } else if (e.code === 'Escape') {
        e.preventDefault();
        document.getElementById('stopMoveBtn').click();
    }
});
```

---

## Code Style Guide

### HTML

- **Semantic elements** where possible
- **IDs for JavaScript targets** (unique)
- **Classes for styling** (reusable)
- **Inline styles** only for dynamic values
- **Title attributes** for tooltips

### CSS

- **CSS Custom Properties** for all colors
- **Mobile-first** approach (not yet implemented, but recommended)
- **BEM naming** for complex components (optional)
- **Consistent spacing** (2-space indentation)
- **Gradients** for visual feedback

**Example:**
```css
.section {
    background: white;
    border: 3px solid var(--light-blue);
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
```

### JavaScript

- **Const by default**, let when reassignment needed
- **Arrow functions** for callbacks
- **Async/await** for promises
- **Template literals** for string interpolation
- **Descriptive variable names**
- **Comments for complex logic**

**Example:**
```javascript
const executeMove = async (moveName, dataset) => {
    try {
        const url = `${serverUrl}/api/move/play/${dataset}/${moveName}`;
        const response = await fetch(url, { method: 'POST' });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();
        return data.uuid;

    } catch (error) {
        console.error('Move execution failed:', error);
        throw error;
    }
};
```

### Naming Conventions

- **Elements:** camelCase (e.g., `gotoYawSlider`)
- **Functions:** camelCase (e.g., `syncSliderAndInput`)
- **Constants:** camelCase (e.g., `serverUrl`)
- **CSS classes:** kebab-case (e.g., `slider-row`)
- **CSS variables:** kebab-case (e.g., `--navy-blue`)

### Comments

**Inline comments** for complex logic:
```javascript
// Delta-based binding: both yaws move together but maintain relative offset
const delta = value - parseFloat(document.getElementById('gotoYawR').value);
```

**Block comments** for section headers:
```javascript
/*
 * Manual Position Control
 * Handles slider/input synchronization and execution
 */
```

**TODOs** for future improvements:
```javascript
// TODO: Add debouncing to prevent excessive API calls
// TODO: Implement WebSocket reconnection logic
```

---

## Future Enhancements

### Planned Features

1. **Mobile Responsive Layout**
   - Touch-optimized sliders
   - Collapsible sections
   - Vertical stacking on small screens

2. **Choreography Sequencer**
   - Visual timeline view
   - Drag-and-drop reordering
   - Move preview on hover

3. **Preset Management**
   - Save favorite positions
   - Load presets with one click
   - Export/import preset files

4. **Advanced Bindings**
   - Inverse rotation binding
   - Scaled movements (antenna moves 2x head)
   - Custom binding rules

5. **Performance Mode**
   - Reduced animation
   - Lower WebSocket update rate
   - Simplified UI for low-power devices

6. **Accessibility**
   - Keyboard navigation
   - Screen reader support
   - High contrast mode
   - Focus indicators

### Technical Debt

- No automated testing (consider Jest)
- No TypeScript (consider migration for type safety)
- No modularization (consider splitting into separate JS files)
- No state management library (currently unnecessary)
- No CSS preprocessing (consider SCSS for maintainability)

---

## Debugging Tips

### Browser Console

**Enable verbose logging:**
```javascript
const DEBUG = true;

function log(...args) {
    if (DEBUG) console.log('[ChoreographyBuilder]', ...args);
}
```

### Common Issues

**Sliders not responding:**
1. Check browser console for errors
2. Verify `syncSliderAndInput()` was called for each slider
3. Inspect element to confirm IDs match JavaScript

**API calls failing:**
1. Check Network tab in DevTools
2. Verify daemon is running: `curl http://localhost:8100/api/daemon/status`
3. Check CORS (shouldn't be issue for localhost)
4. Verify request payload structure

**WebSocket not connecting:**
1. Check Console for WebSocket errors
2. Verify URL: `ws://localhost:8100/api/state/stream` (not `wss://`)
3. Check daemon supports WebSocket (should by default)
4. Try different browser

**Chart not updating:**
1. Verify WebSocket is connected
2. Check `poseChart` is initialized
3. Inspect WebSocket messages in Network tab
4. Verify `updateChart()` is being called

### Performance Profiling

**Chrome DevTools:**
1. Open Performance tab
2. Click Record
3. Interact with interface
4. Stop recording
5. Analyze flame graph for bottlenecks

**Expected:**
- Chart updates: <5ms per frame
- Slider input: <1ms per event
- WebSocket message: <2ms per message

---

## License & Attribution

**Interface Author:** Carson (LAURA Project Beta Tester)

**Pollen Robotics SDK:** Apache 2.0 License

**Chart.js:** MIT License

**Reachy Mini:** https://www.pollen-robotics.com/reachy-mini/

---

**Questions or contributions?** Contact the Pollen Robotics team or LAURA project maintainers.
