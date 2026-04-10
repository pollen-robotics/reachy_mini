# Building & Publishing Apps

Reachy Mini has an app ecosystem powered by Hugging Face Spaces. You can build Python apps, publish them, and any Reachy Mini owner can install them in one click from the dashboard.

<iframe
  width="560"
  height="315"
  src="https://www.youtube.com/embed/h2lyqR2eMyM"
  frameborder="0"
  allowfullscreen
/>

> For a step-by-step tutorial with screenshots, see the blog post: [Make and Publish Your Reachy Mini Apps](https://huggingface.co/blog/pollen-robotics/make-and-publish-your-reachy-mini-apps).

---

## Using AI Agents

If you use an AI coding agent (Claude Code, Cursor, Copilot, etc.), it can build apps for you. Point it to the project's [AGENTS.md](https://github.com/pollen-robotics/reachy_mini/blob/main/AGENTS.md):

> *I'd like to create a Reachy Mini app. Start by reading https://github.com/pollen-robotics/reachy_mini/blob/main/AGENTS.md*

The repository includes a `skills/` directory with detailed guides that AI agents use to build apps correctly:

| Skill | What it covers |
|-------|---------------|
| `create-app.md` | App creation workflow and templates |
| `ai-integration.md` | Building LLM-powered apps |
| `control-loops.md` | Real-time reactive apps (tracking, games) |
| `motion-philosophy.md` | Choosing between `goto_target` and `set_target` |
| `interaction-patterns.md` | Antennas as buttons, head as controller |
| `symbolic-motion.md` | Defining motion mathematically (dances, rhythms) |

---

## How Apps Work

The Reachy Mini daemon manages your app's entire lifecycle:

1. You send a "start app" request (from the dashboard or REST API).
2. The daemon launches your app as a **Python subprocess** (`python -u -m your_app.main`).
3. Your app receives a connected `ReachyMini` instance and a `stop_event`.
4. When stopped, the daemon sends `SIGINT` to your process, which triggers graceful shutdown.
5. After the app exits, the daemon returns the robot to its default position.

**Key constraints:**
- Only **one app** can run at a time.
- Your app runs inside the daemon's subprocess — it does not manage its own hardware connections.
- On the Wireless robot, your app runs in a shared virtual environment at `/venvs/apps_venv/`.

---

## Creating an App

Always use the CLI tool to create apps — it generates the correct structure, metadata, and entry points:

```bash
# Install reachy-mini if not already done
uv pip install reachy-mini

# Create and publish in one step (recommended)
reachy-mini-app-assistant create my_app_name /path/to/destination --publish

# Or create locally first
reachy-mini-app-assistant create my_app_name /path/to/destination
```

> **Never create app folders manually.** The assistant handles boilerplate, Hugging Face tags, entry points, and the correct package structure. Manual creation leads to subtle issues that are hard to debug.

### Choose a Template

| Template | Command | Use when |
|----------|---------|----------|
| **Default** | `reachy-mini-app-assistant create my_app .` | Most apps. Minimal working structure. |
| **Conversation** | `reachy-mini-app-assistant create --template conversation my_app .` | LLM integration, speech, making the robot talk. Includes audio pipeline, LLM tools, movement fusion and all the plumbing. |

### Generated Structure

```
my_app/
├── index.html              # Hugging Face Space landing page
├── style.css               # Landing page styles
├── pyproject.toml          # Package config with entry points
├── README.md               # Must contain reachy_mini_python_app tag
└── my_app/
    ├── __init__.py
    ├── main.py             # Your app logic
    └── static/             # Optional web UI
        ├── index.html
        ├── style.css
        └── main.js
```

---

## The ReachyMiniApp Contract

Your app is a class that extends `ReachyMiniApp` and implements a `run()` method. Here's the minimal structure:

```python
import threading
import time

import numpy as np

from reachy_mini import ReachyMini, ReachyMiniApp
from reachy_mini.utils import create_head_pose


class MyApp(ReachyMiniApp):
    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event):
        t0 = time.time()

        while not stop_event.is_set():
            t = time.time() - t0

            # Move the head
            yaw = 30.0 * np.sin(2.0 * np.pi * 0.2 * t)
            head_pose = create_head_pose(yaw=yaw, degrees=True)

            # Move the antennas
            a = np.deg2rad(25.0 * np.sin(2.0 * np.pi * 0.5 * t))
            antennas = np.array([a, -a])

            reachy_mini.set_target(head=head_pose, antennas=antennas)
            time.sleep(0.02)


if __name__ == "__main__":
    app = MyApp()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
```

### Key points

- **`run(reachy_mini, stop_event)`**: The only method you must implement. The `ReachyMini` instance is already connected and ready. Poll `stop_event` in your main loop to exit gracefully.
- **`wrapped_run()`**: Called from the `__main__` block. It handles connecting to the robot, starting optional services, and calling your `run()` method.
- **`stop()`**: Sets the `stop_event`. The daemon calls this via `SIGINT` when stopping your app.
- **`__main__` block**: Required. The daemon runs your app as a module (`python -m my_app.main`), so this block is the actual entry point.

### The `pyproject.toml` Entry Point

The daemon discovers your app through a standard Python entry point. The assistant generates this for you:

```toml
[project.entry-points."reachy_mini_apps"]
my_app = "my_app.main:MyApp"
```

The group name is `reachy_mini_apps` (with underscores). The value points to your class (`module.main:ClassName`).

Any additional dependency for your project should be added in this file.

---

## Optional: Web UI for Your App

If you want a settings page or any web interface for your app, set `custom_app_url` on your class:

```python
class MyApp(ReachyMiniApp):
    custom_app_url: str | None = "http://0.0.0.0:8042"

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event):
        # Define FastAPI routes on self.settings_app
        @self.settings_app.post("/my_endpoint")
        def my_endpoint():
            return {"status": "ok"}

        # Your main loop...
```

When `custom_app_url` is set, the app automatically starts a FastAPI web server that serves files from the `static/` directory inside your package. The dashboard shows a settings icon to open this UI. The page can be accessed from `http://localhost:8042` with a lite or `http://reachy-mini.local:8042` with a wireless.

Set `custom_app_url = None` if your app doesn't need a web UI.

> For a working example with a toggle and sound playback, see the [template app](https://huggingface.co/spaces/pollen-robotics/reachy_mini_template_app).

---

## Testing Your App

### 1. Validate structure

```bash
reachy-mini-app-assistant check /path/to/my_app
```

This checks that your app has the correct structure, entry points, and metadata.

### 2. Run directly

You can run your app's `main.py` directly for quick iteration (make sure the daemon is running):

```bash
python -m my_app.main
```

### 3. Test through the dashboard

Install your app locally and test it through the dashboard, as users would:

```bash
# Install in development mode
uv pip install -e /path/to/my_app

# Start the daemon
reachy-mini-daemon          # Lite
reachy-mini-daemon --sim    # Simulation
```

Then open <a href="http://127.0.0.1:8000/">http://127.0.0.1:8000/</a> — your app appears in the installed list.

---

## Publishing to Hugging Face

### 1. Log in to Hugging Face

```bash
uv pip install --upgrade huggingface_hub
hf auth login
```

Use a token with **Write** permissions.

### 2. Publish

If you used `--publish` when creating the app, it's already a Hugging Face Space with a Git remote. Just push:

```bash
git add . && git commit -m "my changes" && git push
```

If you created without `--publish`, you can publish later:

```bash
reachy-mini-app-assistant publish /path/to/my_app
```

### 3. Discoverability

For your app to appear in the Reachy Mini app store, its `README.md` must contain the `reachy_mini_python_app` tag in the YAML frontmatter:

```yaml
---
tags:
  - reachy_mini_python_app
---
```

The assistant adds this automatically. If you create the README manually, don't forget it.

---

## Installing Apps

### From the dashboard

Open the Reachy Mini dashboard and click **Install** on any community app. This is the easiest way.

### Via the REST API

```bash
# Install from Hugging Face
curl -X POST http://localhost:8000/api/apps/install \
  -H "Content-Type: application/json" \
  -d '{"url": "https://huggingface.co/spaces/<user>/<app_name>"}'

# Start an app
curl -X POST http://localhost:8000/api/apps/start-app/<app_name>

# Stop the current app
curl -X POST http://localhost:8000/api/apps/stop-current-app

# List installed apps
curl http://localhost:8000/api/apps/list
```

Replace `localhost` with `reachy-mini.local` or the robot's IP address for the Wireless version.

### Offline / manual deployment for a Wireless unit

If you don't have internet access on the robot (e.g., at a conference), you can install your app directly:

```bash
# Copy and install your app on the robot
scp -r /path/to/my_app pollen@reachy-mini.local:/tmp/my_app
ssh pollen@reachy-mini.local "/venvs/apps_venv/bin/pip install /tmp/my_app"
```

After updating code manually, restart the daemon or the app for changes to take effect.

---

## Debugging Apps

### Viewing logs

<hfoptions id="viewing-logs">
<hfoption id="Lite / Simulation">

If you run the daemon in a terminal, app logs (stdout/stderr) appear there directly.

```bash
reachy-mini-daemon          # Lite
reachy-mini-daemon --sim    # Simulation
# App logs will print here
```

</hfoption>
<hfoption id="Wireless">

App output is captured by the daemon and available via `journalctl`:

```bash
ssh pollen@reachy-mini.local

# Live logs
sudo journalctl -u reachy-mini-daemon -f

# Recent logs, filtered (daemon logs are noisy with HTTP access logs)
sudo journalctl -u reachy-mini-daemon --since '5 min ago' | grep -v "uvicorn\|GET \|POST "
```

</hfoption>
</hfoptions>


### Common issues

| Problem | Solution |
|---------|----------|
| "An app is already running" | Stop the current app first: `curl -X POST http://localhost:8000/api/apps/stop-current-app` |
| Daemon in a bad state | Restart it: `sudo systemctl restart reachy-mini-daemon` (wait ~30s before starting an app) |
| App not picking up code changes | Restart the app. If you deployed manually, also clear bytecode: `rm -rf __pycache__` |

### Tip: log everything at startup

A useful debugging practice is logging your configuration when the app starts:

```python
import logging
import sys

logger = logging.getLogger(__name__)

def run(self, reachy_mini, stop_event):
    logger.info("=" * 50)
    logger.info("MY APP STARTING")
    logger.info(f"  Python: {sys.version}")
    logger.info("=" * 50)
    # ...
```

---

## App Configuration

The app subprocess **inherits the daemon's environment variables**. There is no special injection mechanism — your app sees whatever the daemon process sees.

If your app needs runtime configuration (API keys, server URLs, etc.), the recommended approach is to use **the app's web UI**. Set `custom_app_url` on your class and add a settings page where users can enter values (API keys, server addresses, etc.) directly from their browser. This is the most user-friendly option and works across all platforms. See [Optional: Web UI for Your App](#optional-web-ui-for-your-app) above for how to set this up.

Other approaches:

- **Config file**: Read from a known path (e.g., `.env` see [example here](https://github.com/pollen-robotics/reachy_mini_conversation_app/blob/main/.env.example)).
- **Hardcoded defaults**: Simple and debuggable for development.

---

## Using Audio in Your App

Audio recording, playback, and direction-of-arrival detection work the same way inside an app as in a standalone script — use the SDK methods directly (`start_recording()`, `get_audio_sample()`, `push_audio_sample()`, `play_sound()`).

Refer to the official examples for working code:

- **[Sound Recording](../examples/sound_record)**: Record from the mic array and save to WAV.
- **[Sound Playback](../examples/sound_play)**: Play a WAV file or push real-time audio (e.g., from a TTS engine).
- **[Sound Direction of Arrival](../examples/sound_doa)**: Detect who is speaking and make the robot look at them.

For details on how audio streams differ between Wireless and Lite, see [Media Architecture](media-architecture.md).

---

## Further Reading

- **Blog post**: [Make and Publish Your Reachy Mini Apps](https://huggingface.co/blog/pollen-robotics/make-and-publish-your-reachy-mini-apps) — full walkthrough with screenshots
- **API Reference**: [Apps API](../API/apps) — auto-generated reference for `ReachyMiniApp`, `AppManager`, and related classes
- **REST API**: [REST API Reference](../API/rest-api) — full HTTP endpoint documentation
- **Example apps**: Browse [community apps on Hugging Face](https://hf.co/reachy-mini/#/apps) for inspiration

| App | Key Patterns | Link |
|-----|-------------|------|
| Conversation App | LLM tools, audio pipeline, control loops | [GitHub](https://github.com/pollen-robotics/reachy_mini_conversation_app) |
| Marionette | Motion recording, safe torque, HF datasets | [HF Space](https://huggingface.co/spaces/RemiFabre/marionette) |
| Radio | Antenna interaction pattern | [HF Space](https://huggingface.co/spaces/pollen-robotics/reachy_mini_radio) |
| Simon | No-GUI pattern (antenna to start) | [HF Space](https://huggingface.co/spaces/apirrone/reachy_mini_simon) |
| Hand Tracker | Camera-based real-time control loop | [HF Space](https://huggingface.co/spaces/pollen-robotics/hand_tracker_v2) |
| Spaceship Game | Head-as-joystick, antenna buttons | [HF Space](https://huggingface.co/spaces/apirrone/spaceship_game) |
