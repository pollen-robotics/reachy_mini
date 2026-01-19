# Reachy Mini Development Guide for AI Agents

This guide helps AI agents (Claude, GPT, Copilot, etc.) assist users in developing Reachy Mini applications.

---

## First-Run Setup

**Before doing anything else:**

1. Check if `agents.local.md` exists (search in current directory and `~/reachy_mini_resources/`)
2. If it exists: Read it for user-specific configuration and skip completed setup steps
3. If it doesn't exist: Guide the user through setup interactively (see below)

### Initial Setup Conversation

Before downloading anything, explain to the user:

> "To help you develop Reachy Mini apps effectively, I'll download the official SDK repository and several example apps to a local folder. This gives me access to the source code, documentation, and proven patterns.
>
> The default location is `~/reachy_mini_resources/`. Would you like to use a different location?"

Once confirmed, proceed with environment setup and cloning.

---

## 1. Environment Setup

### Python Virtual Environment

Ask the user their preference, or use these defaults:

**Preferred approach (try first):**
```bash
# Using uv (faster, recommended)
uv venv .venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows
uv pip install reachy-mini
```

**Fallback (if uv not installed):**
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows
pip install reachy-mini
```

**Important:** Adapt commands to the user's shell (bash/zsh/fish/PowerShell) and OS.

Record the user's preference in `agents.local.md`.

---

## 2. Clone Local Resources

Create a single folder for all Reachy Mini resources:

```bash
mkdir -p ~/reachy_mini_resources
cd ~/reachy_mini_resources

# Main SDK (essential)
git clone https://github.com/pollen-robotics/reachy_mini

# Example apps
git clone https://github.com/pollen-robotics/reachy_mini_conversation_app
git clone https://github.com/pollen-robotics/reachy_mini_dances_library
git clone https://huggingface.co/spaces/RemiFabre/marionette
git clone https://huggingface.co/spaces/RemiFabre/fire_nation_attacked
git clone https://huggingface.co/spaces/apirrone/spaceship_game
git clone https://huggingface.co/spaces/pollen-robotics/reachy_mini_radio
git clone https://huggingface.co/spaces/apirrone/reachy_mini_simon
git clone https://huggingface.co/spaces/pollen-robotics/hand_tracker_v2
```

**After cloning, create `agents.local.md`** to record paths and avoid repeating setup.

### Key Reference Files

| Purpose | Path |
|---------|------|
| SDK API (all methods) | `reachy_mini/src/reachy_mini/reachy_mini.py` |
| App base class | `reachy_mini/src/reachy_mini/apps/app.py` |
| **Control loop reference** | `reachy_mini_conversation_app/src/reachy_mini_conversation_app/moves.py` |
| Safe torque handling | `marionette/marionette/main.py` |
| Symbolic motion | `reachy_mini_dances_library/src/reachy_mini_dances_library/rhythmic_motion.py` |
| Recorded moves example | `reachy_mini/examples/recorded_moves_example.py` |

---

## 3. Understanding the SDK

### Documentation

**Read the docs from the cloned repo** (`reachy_mini/docs/`).

Start with:
1. `docs/SDK/quickstart.md`
2. `docs/SDK/core-concept.md`
3. `docs/SDK/python-sdk.md`

### When in Doubt

1. **Check docstrings** - All functions in `reachy_mini.py` are well documented
2. **Read the source code** - It's open source, use it!
3. **Never invent functions** - Verify they exist before using them

### Providing Feedback

If you find something unclear or missing:
1. Append to `~/reachy_mini_resources/insights_for_reachy_mini_maintainers.md`
2. Encourage the user to submit a PR or issue to improve the ecosystem

---

## 4. Creating Apps

### App Creation Guide

Full tutorial: https://huggingface.co/blog/pollen-robotics/make-and-publish-your-reachy-mini-apps

### Quick Start

```bash
# Create and publish immediately (public by default)
reachy-mini-app-assistant create my_app_name /path/to/create --publish
```

### Recommended Workflow

1. **Create and publish immediately** with `--publish` to get a Git repo on Hugging Face
2. **Develop iteratively** using standard git commands (`git add`, `git commit`, `git push`)
3. **Validate** with `reachy-mini-app-assistant check /path/to/app`
4. **Request official status** only for high-quality, polished apps

### App Structure

```
my_app/
├── index.html              # HuggingFace Space landing page
├── style.css               # Landing page styles
├── pyproject.toml          # Package configuration (includes reachy_mini tag!)
├── README.md               # Must contain reachy_mini tag in YAML frontmatter
├── .gitignore
└── my_app/
    ├── __init__.py
    ├── main.py             # Your app code (run method)
    └── static/             # Optional web UI
        ├── index.html
        ├── style.css
        └── main.js
```

---

## 5. Motion Control Philosophy

### Two Methods: When to Use Each

| Method | Behavior | When to Use |
|--------|----------|-------------|
| `goto_target()` | Smooth interpolation over duration | **Default choice** - gestures, choreography, transitions |
| `set_target()` | Immediate, no interpolation | Control loops requiring real-time reactivity |

**Key insight:** `goto_target()` is simpler and ensures smooth motion. Use it when you can. But during the interpolation, you cannot react to external stimuli - the robot commits to completing the movement.

For reactive applications (face tracking, real-time control), you need `set_target()` in a control loop.

### Basic goto_target Usage

```python
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

with ReachyMini() as mini:
    pose = create_head_pose(yaw=30, pitch=10, degrees=True)
    mini.goto_target(head=pose, duration=1.0, method="minjerk")
```

---

## 6. Control Loops with set_target

When you need real-time reactivity (face tracking, game input, etc.), you must use `set_target()` in a control loop.

### The Core Rule

**A single control loop that continuously sends targets.** That's it. One place in the code calls `set_target()`, and it runs at a fixed frequency (typically 50-100 Hz).

```python
import time
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

with ReachyMini() as mini:
    while not stop_event.is_set():
        # Compute the pose you want (can change every tick!)
        pose = compute_current_target_pose()

        # Send it
        mini.set_target(head=pose, antennas=antennas)

        # Sleep to maintain frequency
        time.sleep(0.01)  # ~100Hz
```

### Why This Matters

- **Single control point** prevents race conditions and jerky motion
- **Continuous targets** maintain the "illusion of life" - even when idle, keep sending
- **Modify the pose, not where you call set_target** - if you need complex behavior, change what pose you're sending, don't add more set_target calls elsewhere

### Advanced Example: moves.py

**Reference:** `reachy_mini_conversation_app/src/reachy_mini_conversation_app/moves.py`

This file demonstrates a sophisticated implementation with:

- **Primary moves** (emotions, dances, goto, breathing) - mutually exclusive, run sequentially
- **Secondary moves** (face tracking, speech sync) - additive offsets layered on top
- **Pose fusion** - combining primary pose with secondary offsets
- **Phase-aligned timing** - uses `time.monotonic()` to measure elapsed time accurately (avoids wall-clock jumps from NTP, sleep, etc.)
- **Threading model** - dedicated worker thread owns robot output, other threads communicate via queues

The key insight from moves.py docstring:
```
- There is a single control point to the robot: `ReachyMini.set_target`.
- The control loop runs near 100 Hz and is phase-aligned via a monotonic clock.
```

---

## 7. AI Agent Integration

**Reference:** `reachy_mini_conversation_app/`

The conversation app demonstrates how to turn Reachy Mini into an intelligent, autonomous robot by integrating an LLM (OpenAI Realtime API, compatible with Grok and others) that controls the robot through function calls.

### Key Features

- **Real-time conversation** with low-latency audio streaming
- **LLM tools** that control the robot:
  - `move_head` - Queue head pose changes
  - `dance` - Play dances from the library
  - `play_emotion` - Play recorded emotions
  - `camera` - Capture and analyze images
  - `head_tracking` - Enable/disable face tracking
- **Custom profiles** - Different personalities with different instructions and enabled tools
- **Vision processing** - Cloud (GPT) or local (SmolVLM2)

### Architecture

The LLM doesn't control motors directly. Instead:
1. LLM decides to call a tool (e.g., "dance")
2. Tool implementation queues a move
3. The control loop (moves.py) executes it smoothly

This separation keeps the control loop clean and ensures smooth motion regardless of LLM latency.

---

## 8. Safe Torque Handling

**Reference:** `marionette/marionette/main.py`

When toggling motor torque (for recording, compliance, etc.), avoid jerky motion:

### Before Disabling Torque

Go to a safe position first (head will fall when torque is off):

```python
from reachy_mini.reachy_mini import SLEEP_HEAD_POSE
import numpy as np

# Move to sleep position before disabling
antenna_angle = np.deg2rad(15)
reachy_mini.goto_target(
    SLEEP_HEAD_POSE,
    antennas=[-antenna_angle, antenna_angle],
    duration=1.0,
)
reachy_mini.disable_motors()
```

### Before Enabling Torque

Set goal to current position first (prevents jump to old goal):

```python
def safe_enable_motors(reachy_mini):
    """Enable motors without jerky motion."""
    head_pose = reachy_mini.get_current_head_pose()
    _, antennas = reachy_mini.get_current_joint_positions()

    reachy_mini.goto_target(
        head=head_pose,
        antennas=list(antennas) if antennas is not None else None,
        duration=0.05,
    )
    reachy_mini.enable_motors()
```

---

## 9. Platform Compatibility

Apps should work across setups when possible. Advertise limitations clearly.

| Setup | Compute | Latency | Camera | Notes |
|-------|---------|---------|--------|-------|
| **Lite** | Full (laptop) | Low | Direct USB | Most flexible, best for development |
| **Wireless (local)** | Limited (CM4) | Low | Direct | Memory/CPU constrained |
| **Wireless (streamed)** | Full (laptop) | Higher | Via network | Some tracking quality loss |
| **Simulation** | Full | Low | N/A | Can't test camera-based tracking |

### Media/Audio

Use the official ReachyMini interfaces (`get_audio_sample`, `push_audio_sample`) instead of manual implementations - this ensures compatibility across setups.

### OS/Browser Compatibility

Aim for:
- **OS:** Linux, macOS, Windows
- **Browsers:** Chrome, Firefox, Safari (for web UIs)

---

## 10. Interaction Patterns

### Emotions & Expressiveness

Make the robot expressive! Use emotions at:
- App start (clear cue that something's happening)
- App end (graceful closure)
- Key moments during interaction

```python
from reachy_mini.motion.recorded_move import RecordedMoves

moves = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
mini.play_move(moves.get("happy"), initial_goto_duration=1.0)
```

### Antennas as Buttons

The motors use low P in PID - they're semi-passive and safe to push. Use antennas for physical interaction:

```python
_, antennas = mini.get_current_joint_positions()
if abs(antennas[0]) > threshold:  # Left antenna pulled
    trigger_action()
```

**Reference:** `reachy_mini_radio` (change station with antennas)

### Head as Controller

The head has 6 DOF - it's a powerful input device for games or recording.

**Reference:** `fire_nation_attacked`, `spaceship_game` (head as joystick), `marionette` (record head motion)

### No-GUI Pattern

For simple apps, use antenna press to start instead of a web UI:

**Reference:** `reachy_mini_simon` (antenna twitch → wait for press → start game)

---

## 11. Symbolic Motion Definition

**Reference:** `reachy_mini_dances_library/src/reachy_mini_dances_library/rhythmic_motion.py`

Alternative to recorded moves - define motion as mathematical functions:

### Advantages

- **Flexible:** Easy to tune amplitude, frequency, phase
- **Memory efficient:** Function vs thousands of data points
- **LLM-friendly:** Can generate/modify motion code dynamically

### Core Concept

```python
def simple_nod(t_beats, amplitude_rad=0.2, subcycles_per_beat=1.0):
    pitch = amplitude_rad * np.sin(2 * np.pi * subcycles_per_beat * t_beats)
    return MoveOffsets(
        position_offset=np.zeros(3),
        orientation_offset=np.array([0, pitch, 0]),
        antennas_offset=np.zeros(2),
    )

# In control loop:
t_beats = time_seconds * bpm / 60.0
offsets = simple_nod(t_beats, amplitude_rad=0.3)
```

---

## 12. Example Apps Reference

| App | Key Patterns | Source |
|-----|--------------|--------|
| **reachy_mini_conversation_app** | AI agent integration, control loop, LLM tools, face tracking, speech sync | [GitHub](https://github.com/pollen-robotics/reachy_mini_conversation_app) |
| **marionette** | Recording motion, safe torque handling, HF dataset publishing, audio sync | [HF Space](https://huggingface.co/spaces/RemiFabre/marionette) |
| **fire_nation_attacked** | Head-as-controller, leaderboards via HF dataset, game state | [HF Space](https://huggingface.co/spaces/RemiFabre/fire_nation_attacked) |
| **spaceship_game** | Head-as-joystick, antenna buttons, sensor streaming | [HF Space](https://huggingface.co/spaces/apirrone/spaceship_game) |
| **reachy_mini_radio** | Antenna interaction pattern | [HF Space](https://huggingface.co/spaces/pollen-robotics/reachy_mini_radio) |
| **reachy_mini_simon** | No-GUI pattern (antenna press to start) | [HF Space](https://huggingface.co/spaces/apirrone/reachy_mini_simon) |
| **hand_tracker_v2** | Camera-based control loop | [HF Space](https://huggingface.co/spaces/pollen-robotics/hand_tracker_v2) |
| **reachy_mini_dances_library** | Symbolic motion definition | [GitHub](https://github.com/pollen-robotics/reachy_mini_dances_library) |

---

## 13. Constants Quick Reference

**Motor Names:**
```
body_rotation, stewart_1, stewart_2, stewart_3, stewart_4, stewart_5, stewart_6, right_antenna, left_antenna
```

**Interpolation Methods:** `linear`, `minjerk` (default), `ease`, `cartoon`

**Safety Limits** (from `docs/SDK/core-concept.md`):
| Joint | Range |
|-------|-------|
| Head pitch/roll | [-40, +40] degrees |
| Head yaw | [-180, +180] degrees |
| Body yaw | [-160, +160] degrees |
| Yaw delta (head - body) | Max 65° difference |

**Note:** Gentle collisions with the body are safe - no need to enforce strict collision avoidance. The SDK clamps values automatically.

**Units:**
- Head pose: degrees (with `degrees=True`) or radians
- Head position: mm (with `mm=True`) or meters
- **Antennas: always radians** - use `np.deg2rad()`

---

## Community Resources

- **App creation guide**: https://huggingface.co/blog/pollen-robotics/make-and-publish-your-reachy-mini-apps
- **Source code**: https://github.com/pollen-robotics/reachy_mini
- **Community apps**: https://huggingface.co/spaces?q=reachy_mini
- **Discord**: https://discord.gg/Y7FgMqHsub
