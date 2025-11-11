# Reachy Mini Control Center

A web-based control panel for Reachy Mini with extension support, inspired by Automatic1111's Stable Diffusion WebUI architecture.

## Features

- **Web-based UI** - Access from any browser, no desktop app required
- **Video streaming** - Live MJPEG stream from daemon camera
- **Manual control** - Sliders for head position/rotation and antenna control
- **Move library** - Play dances and emotions from Hugging Face libraries
- **Extension system** - Auto-discover and load community extensions
- **Simple like A1111** - Tab-based navigation, familiar patterns

## Prerequisites

1. **Reachy Mini daemon must be running:**
   ```bash
   cd /Users/lauras/Desktop/laura/reachy_mini
   source /Users/lauras/Desktop/laura/venv/bin/activate
   mjpython -m reachy_mini.daemon.app.main --sim --scene minimal --fastapi-port 8100
   ```

2. **Python 3.10+** with Gradio installed

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch control center:**
   ```bash
   python control_center.py
   ```

3. **Open browser:**
   - Automatically opens at `http://localhost:7860`
   - Or manually navigate to that URL

## Usage

### Manual Control Tab

Control robot position and rotation using sliders:

- **Head Position:** X, Y, Z sliders (in mm)
- **Head Rotation:** Yaw, Pitch, Roll sliders (in degrees)
- **Antennas:** Left and right antenna positions
- **Duration:** Movement duration slider
- **Buttons:**
  - "Move to Position" - Execute movement
  - "Emergency Stop" - Stop all movement

### Move Library Tab

Play pre-recorded moves:

- **Dances** - 20 choreographed dance moves
- **Emotions** - 81 expressive emotion moves
- Select move from radio buttons
- Click "Execute Move"

### Settings Tab

- **Connection Settings** - Test daemon connection
- **Extensions** - View installed extensions

## Extensions

Extensions are auto-discovered from the `extensions/` directory.

### Installing Extensions

1. **Clone extension to `extensions/` directory:**
   ```bash
   cd extensions/
   git clone https://github.com/user/extension-name
   ```

2. **Extension must have `manifest.json`** in root directory

3. **Restart control center** to detect new extension

4. **Extension will appear as new tab** in UI

### Creating Extensions

See `extension_manifests/` directory for examples:

- `dance_dance_reachy_manifest.json` - Pose mirroring extension
- `choreography_builder_manifest.json` - AI choreography generation
- `conversation_app_manifest.json` - AI conversation mode
- `move_controller_manifest.json` - Advanced manual control

### Extension Structure

```
extensions/my-extension/
├── manifest.json          # Required: UI declaration
├── main.py               # Optional: Extension app with REST API
├── requirements.txt      # Optional: Python dependencies
└── templates/
    └── display.html      # Optional: HTML display at /display endpoint
```

### Manifest Format

```json
{
  "extension": {
    "name": "My Extension",
    "api_base_url": "http://localhost:5050"
  },
  "display": {
    "enabled": true,
    "url": "/display",
    "title": "Extension Display"
  },
  "sidebar_panel": {
    "title": "Controls",
    "controls": [
      {
        "type": "button",
        "label": "Start",
        "endpoint": "/start",
        "method": "POST"
      }
    ]
  },
  "lifecycle": {
    "on_start": "python main.py --port 5050",
    "on_stop": "/shutdown"
  }
}
```

## Command-Line Options

```bash
python control_center.py --help
```

Options:
- `--daemon-url URL` - Daemon URL (default: http://localhost:8100)
- `--port PORT` - Control center port (default: 7860)
- `--share` - Create public share link via Gradio
- `--debug` - Enable debug logging

## Architecture

```
control_center.py           # Main Gradio app
├── core/
│   ├── daemon_client.py    # REST API wrapper
│   ├── extension_manager.py # Extension discovery
│   └── ui_builder.py       # Manifest → Gradio components
├── extensions/             # User-installed extensions
└── extensions-builtin/     # Built-in extensions
```

## Troubleshooting

### "Connection failed" error

- Verify daemon is running: `curl http://localhost:8100/api/daemon/status`
- Check daemon URL in Settings tab
- Ensure no firewall blocking localhost

### Video stream not showing

- Check MJPEG stream directly: `http://localhost:8100/api/camera/stream.mjpg`
- Verify daemon has camera access
- Try refreshing browser

### Extension not appearing

- Check `manifest.json` is valid JSON
- Look at console logs for validation errors
- Restart control center after installing extension

## Development

### Adding Built-in Tabs

Edit `control_center.py` and add new tab in `create_ui()`:

```python
with gr.Tab("My Feature"):
    self._build_my_feature_tab()
```

### Modifying UI Components

Edit `core/ui_builder.py` to add new control types or modify existing ones.

### Custom Styling

Edit `static/style.css` for custom CSS (not yet integrated - TODO).

## Comparison with Desktop Viewer

| Feature | Desktop Viewer | Control Center |
|---------|---------------|----------------|
| **UI** | ImGui (desktop app) | Gradio (web UI) |
| **Access** | Local only | Browser, any device |
| **Extensions** | Hardcoded | Auto-discovered |
| **Community** | Developer-only | Beta tester friendly |
| **Code** | ~2000 lines | ~1000 lines |

## License

Apache 2.0 (same as Reachy Mini SDK)

---

## Quick Start

```bash
# Terminal 1: Start daemon
cd /Users/lauras/Desktop/laura/reachy_mini
source /Users/lauras/Desktop/laura/venv/bin/activate
mjpython -m reachy_mini.daemon.app.main --sim --scene minimal --fastapi-port 8100

# Terminal 2: Start control center
cd /Users/lauras/Desktop/laura/reachy_mini
python control_center.py

# Browser opens automatically at http://localhost:7860
```

---

*Built for the Reachy Mini beta testing community. Simple, extensible, and familiar.*
