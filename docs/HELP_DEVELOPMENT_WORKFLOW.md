# Reachy Mini -- Development Workflow Guide

Turn tinkering into engineering. Build robust apps using proven patterns.

---

## Recommended Project Structure

Use the app assistant to scaffold new projects. Never create app folders manually:

```bash
# Standard app:
reachy-mini-app-assistant create my_app ~/projects/my_app --publish

# Conversation/LLM app:
reachy-mini-app-assistant create --template conversation my_ai_app ~/projects/my_ai_app --publish
```

This generates the correct metadata, entry points, and folder structure. The `--publish` flag prepares it for sharing on Hugging Face.

### Resulting structure:
```
my_app/
├── my_app/
│   └── main.py        # Entry point
├── static/            # Web UI files (if applicable)
├── pyproject.toml     # Package metadata
└── plan.md            # Your implementation plan (create this)
```

---

## Development Loop

### 1. Plan before coding

Create `plan.md` in your app directory:
- What the app should do
- Which SDK features it uses (`goto_target` vs `set_target`, media, etc.)
- Known constraints (Lite only? Wireless only? Need camera?)

### 2. Start with simulation

```bash
# Terminal 1: Start simulator
reachy-mini-daemon --sim

# Terminal 2: Run your app
python my_app/main.py
```

Simulation catches logic errors without risking hardware. Use `media_backend="no_media"` if your app does not need camera/audio.

### 3. Test on hardware

Once simulation works, connect to the real robot:
- Lite: `reachy-mini-daemon` (no `--sim` flag)
- Wireless: Daemon is already running

Timing and PID behavior will differ slightly from simulation. Tune durations and control loop frequencies on real hardware.

### 4. Test on Wireless (if applicable)

SSH into the robot and run your app there:
```bash
ssh pollen@reachy-mini.local  # password: root
cd my_app
python my_app/main.py
```

This reproduces exactly what the dashboard does when launching an installed app.

---

## Logging and Debugging

### Add logging to your app
```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug(f"Head pose: {mini.get_current_head_pose()}")
logger.info("Starting main loop")
```

### Daemon logs
```bash
# Lite: Run daemon with verbose output
reachy-mini-daemon --verbose

# Wireless: Check system logs
ssh pollen@reachy-mini.local
journalctl -u reachy-mini-daemon.service

# Restart daemon on Wireless
systemctl restart reachy-mini-daemon.service
```

### Check motor control loop health
```python
print(mini.client.get_status())
```

You should see ~50Hz (~20ms period). If the period is much higher, the control loop is too slow (CPU overloaded or high USB latency).

---

## Version Control Advice

Even for personal projects, use Git:

```bash
cd my_app
git init
git add .
git commit -m "Initial scaffold from app assistant"
```

### Commit often in small increments
Small commits make debugging easier. If something breaks, `git diff` between the working and broken state is usually enough to spot the issue.

### When something breaks after changes

Use git history to narrow down the problem:

```bash
# See recent commits
git log --oneline -10

# Go back to a known-good state
git checkout <older-commit>
python my_app/main.py  # Does it work?

# Compare the diff
git diff <good-commit> <bad-commit>
```

---

## Reference Apps

Study these production-quality apps to learn patterns:

| App | Key Patterns |
|-----|-------------|
| [reachy_mini_conversation_app](https://github.com/pollen-robotics/reachy_mini_conversation_app) | LLM integration, control loops, pose fusion, threading |
| [marionette](https://huggingface.co/spaces/RemiFabre/marionette) | Motion recording, safe torque, HuggingFace datasets |
| [fire_nation_attacked](https://huggingface.co/spaces/RemiFabre/fire_nation_attacked) | Head-as-controller, game logic, leaderboards |
| [spaceship_game](https://huggingface.co/spaces/apirrone/spaceship_game) | Head-as-joystick, antenna buttons |
| [reachy_mini_radio](https://huggingface.co/spaces/pollen-robotics/reachy_mini_radio) | Antenna interaction pattern |
| [reachy_mini_simon](https://huggingface.co/spaces/apirrone/reachy_mini_simon) | No-GUI pattern (antenna to start) |
| [hand_tracker_v2](https://huggingface.co/spaces/pollen-robotics/hand_tracker_v2) | Camera-based control loop |
| [reachy_mini_dances_library](https://github.com/pollen-robotics/reachy_mini_dances_library) | Symbolic motion definition |

Clone them for local study:
```bash
git clone https://github.com/pollen-robotics/reachy_mini_conversation_app
```

---

## Adding New Behaviors

### Using `goto_target` (choreographed)
```python
# Define a sequence of poses
poses = [
    (create_head_pose(yaw=30, degrees=True), [0.5, -0.5], 0.8),
    (create_head_pose(pitch=-20, degrees=True), [0, 0], 0.6),
    (create_head_pose(), [0, 0], 1.0),
]

for head, antennas, duration in poses:
    mini.goto_target(head=head, antennas=antennas, duration=duration)
```

### Using recorded moves
```python
from reachy_mini.motion.recorded_move import RecordedMoves

moves = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
mini.play_move(moves.get("happy"), initial_goto_duration=1.0)
```

### Using `set_target` (reactive)
See the control loop template in [HELP_CONTROL_MODES.md](HELP_CONTROL_MODES.md).

---

## Publishing Your App

Once your app works:

1. Push to a Hugging Face Space:
   ```bash
   # The app assistant handles this if you used --publish
   ```

2. Other users can install it directly from their dashboard.

3. Community apps: https://huggingface.co/spaces?q=reachy_mini

---

## Platform Compatibility Checklist

Before sharing, verify your app works on intended platforms:

| Check | How |
|-------|-----|
| Simulation | `reachy-mini-daemon --sim` + your app |
| Lite hardware | USB connection + your app |
| Wireless local | SSH + run on the robot |
| Wireless remote | Run on laptop, robot over WiFi |

Not all apps need all platforms. A camera-tracking app won't work in simulation. A simple dance works everywhere.
