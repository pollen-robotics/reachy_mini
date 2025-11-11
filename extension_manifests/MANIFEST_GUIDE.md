# Extension Manifest Guide

Created 4 example manifests for your existing apps:

## 1. Choreography Builder (`choreography_builder_manifest.json`)

**Key Features:**
- File upload widget for music
- Mood/intensity/complexity controls
- Multi-step workflow (upload → analyze → generate → play)
- Progress bar for generation
- Export button
- Timeline display in viewport

**Unique Controls:**
- `file_upload` - Upload MP3/WAV files
- `progress_bar` - Shows generation progress
- `button.download` - Export choreography JSON

**Display:** Timeline editor in viewport

---

## 2. Manual Move Controller (`move_controller_manifest.json`)

**Key Features:**
- 6 sliders for head position (X/Y/Z)
- 3 sliders for head rotation (yaw/pitch/roll)
- 2 antenna sliders
- Quick preset buttons (9-button grid for look directions)
- Binding controls (sync yaw, bind antennas)
- Duration slider for movement speed

**Unique Controls:**
- `color_zones` - Safe (blue) vs warning (pink) ranges
- `symmetric_gradient` - Rotation sliders (danger at extremes)
- `visual_reverse` - Right antenna appears flipped
- `button_grid` - 3x3 grid of arrow buttons
- `exclusive_with` - Can't enable both antenna bindings

**Display:** None (uses main 3D viewer)

---

## 3. Conversation App (`conversation_app_manifest.json`)

**Key Features:**
- Backend selection (OpenAI vs Gemini)
- Voice selection (when using OpenAI)
- Feature toggles (vision, face tracking, expressive moves)
- Personality presets + custom instructions
- Connection status indicator
- Conversation history display

**Unique Controls:**
- `text_area` - Multi-line custom instructions
- `status_indicator` - Connected/disconnected/connecting states
- `button.confirm` - Confirmation dialog before clearing history
- `environment` - API keys from env variables
- `visible_when` - Show controls conditionally

**Display:** Conversation log + camera feed in viewport

---

## 4. Dance Dance Reachy (`dance_dance_reachy_manifest.json`)

**Key Features:**
- Start/stop/calibrate buttons
- Sensitivity sliders (hip sway, arm movement)
- Real-time pose display (hip sway, arm angles with color coding)
- Camera source dropdown
- Skeleton overlay toggle
- FPS counter + pose detection indicator

**Unique Controls:**
- `color_range` - Green/yellow/red based on value
- `status_indicator` - Boolean states with icons (✓/✗)
- `disabled_when` - Camera selector disabled while running

**Display:** Webcam with pose overlay in viewport

---

## Common Patterns

### All Manifests Have:

1. **Basic Info**
   ```json
   "extension": {
     "name": "...",
     "api_base_url": "http://localhost:XXXX"
   }
   ```

2. **Sidebar Panel**
   ```json
   "sidebar_panel": {
     "title": "...",
     "icon": "emoji",
     "controls": [...]
   }
   ```

3. **Status Polling**
   ```json
   "status_polling": {
     "endpoint": "/status",
     "interval_ms": 100-500,
     "fields": {...}
   }
   ```

4. **Lifecycle**
   ```json
   "lifecycle": {
     "on_start": "python ...",
     "on_stop": "/shutdown",
     "healthcheck": "/health"
   }
   ```

---

## Control Types Reference

### Buttons
```json
{
  "type": "button",
  "label": "Start",
  "endpoint": "/start",
  "method": "POST",
  "color": "green|red|blue|gray",
  "enabled_when": "condition",
  "disabled_when": "condition",
  "confirm": "Confirmation message?"
}
```

### Sliders
```json
{
  "type": "slider",
  "label": "Intensity",
  "endpoint": "/config/intensity",
  "min": 0.0,
  "max": 2.0,
  "default": 1.0,
  "step": 0.1,
  "unit": "x",
  "color_zones": {
    "safe": [-10, 10],
    "warning": [-40, 40]
  },
  "symmetric_gradient": true,
  "visual_reverse": false
}
```

### Dropdowns
```json
{
  "type": "dropdown",
  "label": "Mode",
  "endpoint": "/config/mode",
  "options": [
    {"label": "Option 1", "value": "opt1"},
    {"label": "Option 2", "value": "opt2"}
  ],
  "default": "opt1"
}
```

### Toggles
```json
{
  "type": "toggle",
  "label": "Enable Feature",
  "endpoint": "/config/feature",
  "default": true,
  "description": "What this does",
  "exclusive_with": "other_toggle_id"
}
```

### Status Text
```json
{
  "type": "status_text",
  "label": "FPS",
  "endpoint": "/status",
  "field": "fps",
  "format": "{:.1f}",
  "color_range": {
    "green": [25, 100],
    "yellow": [15, 25],
    "red": [0, 15]
  }
}
```

### Status Indicator
```json
{
  "type": "status_indicator",
  "label": "Connection",
  "endpoint": "/status",
  "field": "connected",
  "states": {
    "connected": {"color": "green", "icon": "●"},
    "disconnected": {"color": "red", "icon": "○"}
  }
}
```

### File Upload
```json
{
  "type": "file_upload",
  "label": "Upload",
  "endpoint": "/upload",
  "accept": ".mp3,.wav",
  "max_size_mb": 50
}
```

### Text Area
```json
{
  "type": "text_area",
  "label": "Instructions",
  "endpoint": "/config/instructions",
  "placeholder": "Enter text...",
  "rows": 4
}
```

### Progress Bar
```json
{
  "type": "progress_bar",
  "label": "Generation",
  "endpoint": "/status",
  "field": "progress_percent",
  "visible_when": "generating"
}
```

### Button Grid
```json
{
  "type": "button_grid",
  "buttons": [
    {"label": "↑", "endpoint": "/up", "method": "POST"},
    {"label": "↓", "endpoint": "/down", "method": "POST"}
  ],
  "columns": 3
}
```

---

## Viewport Control

### Auto-Claim (Conversation, DDR)
```json
"viewport_priority": {
  "auto_claim": true,
  "release_on_stop": true,
  "conflicts_with": ["mujoco_viewer"]
}
```
**Behavior:** Takes over viewport when started, releases when stopped.

### Manual Claim (Choreography)
```json
"viewport_priority": {
  "auto_claim": false,
  "manual_claim": true
}
```
**Behavior:** User clicks "Show Timeline" button to display.

### No Claim (Move Controller)
```json
"viewport_priority": {
  "auto_claim": false,
  "prefers": "mujoco_viewer"
}
```
**Behavior:** Works best with 3D viewer, doesn't need its own display.

---

## Conditional Display

### `enabled_when` / `disabled_when`
```json
{
  "type": "button",
  "label": "Stop",
  "enabled_when": "running"
}
```

**How it works:** Control center polls `/status`, checks if `running: true`, then enables button.

### `visible_when`
```json
{
  "type": "slider",
  "visible_when": "backend_openai"
}
```

**How it works:** Control checks if `backend_openai: true` in status, shows/hides control.

### `exclusive_with`
```json
{
  "type": "toggle",
  "id": "bind_normal",
  "exclusive_with": "bind_inverse"
}
```

**How it works:** Enabling one automatically disables the other.

---

## Installation Flow

1. **User enters GitHub URL**
   ```
   https://github.com/ElGrorg/dance-dance-reachy
   ```

2. **Control center clones repo**
   ```bash
   git clone https://github.com/ElGrorg/dance-dance-reachy
   ```

3. **Looks for manifest**
   ```
   dance-dance-reachy/extension_manifest.json
   ```

4. **Runs install commands**
   ```bash
   pip install -r requirements.txt
   python -c 'from ultralytics import YOLO; YOLO("yolov8n-pose.pt")'
   ```

5. **Adds to extensions list**
   - Shows in sidebar
   - Ready to start

---

## How Control Center Uses Manifests

### On Extension Start:

1. Run `lifecycle.on_start.command`
   ```bash
   python main.py --api-mode --port 5050
   ```

2. Wait for healthcheck
   ```
   GET http://localhost:5050/health
   ```

3. Generate sidebar UI from `controls[]`

4. Start status polling
   ```
   GET http://localhost:5050/status every 100ms
   ```

5. Update control states based on status fields

### When User Clicks Button:

1. Check `enabled_when` condition
2. Send request to endpoint
   ```
   POST http://localhost:5050/start
   ```
3. Update UI based on response

### When User Moves Slider:

1. Send new value to endpoint
   ```
   POST http://localhost:5050/config/intensity
   Body: {"value": 1.5}
   ```

### On Extension Stop:

1. Send stop request
   ```
   POST http://localhost:5050/shutdown
   ```

2. Kill subprocess if needed

3. Release viewport if claimed

---

## Key Differences Between Extensions

| Feature | Choreography | Move Controller | Conversation | DDR |
|---------|-------------|-----------------|--------------|-----|
| **Display** | Timeline | None | History | Webcam |
| **Viewport** | Manual | Prefers 3D | Auto-claim | Auto-claim |
| **Port** | 5100 | 5200 | 5300 | 5050 |
| **Workflow** | Multi-step | Realtime | Session-based | Realtime |
| **Status Poll** | 500ms | 100ms | 500ms | 100ms |

---

*This system lets beta testers install extensions without coding - just drop in a GitHub URL and the manifest describes everything.*
