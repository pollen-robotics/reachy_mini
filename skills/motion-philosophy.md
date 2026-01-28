# Skill: Motion Control Philosophy

## When to Use

- Deciding between `goto_target()` and `set_target()`
- Planning how motion will work in your app
- User asks about smooth motion or interpolation

---

## The Two Methods

| Method | Behavior | Use when |
|--------|----------|----------|
| `goto_target()` | Smooth interpolation over duration | **Default choice** - gestures, choreography, transitions |
| `set_target()` | Immediate, no interpolation | Real-time control requiring continuous reactivity |

---

## goto_target(): The Default Choice

Use this for most motion. It handles interpolation automatically.

```python
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

with ReachyMini() as mini:
    pose = create_head_pose(yaw=30, pitch=10, degrees=True)
    mini.goto_target(head=pose, duration=1.0, method="minjerk")
```

### Interpolation Methods

| Method | Character |
|--------|-----------|
| `linear` | Constant speed |
| `minjerk` | Natural, smooth (default) |
| `ease` | Slow start/end |
| `cartoon` | Exaggerated, bouncy |

### Key Insight

During `goto_target()` interpolation, **you cannot react to external stimuli**. The robot commits to completing the movement. This is fine for:
- Playing emotions
- Choreographed sequences
- Transitioning between states

---

## set_target(): For Real-Time Control

Use this when you need to react every frame (face tracking, games, joystick).

**Requirements:**
- Must run in a control loop at 50-100 Hz
- Single place in code calling `set_target()`
- Keep sending targets even when "idle"

See `control-loops.md` for implementation details.

---

## Decision Flowchart

```
Does your app need to react to input/sensors in real-time?
├── NO → Use goto_target()
│        - Simpler code
│        - Guaranteed smooth motion
│        - Good for: emotions, dances, scripted sequences
│
└── YES → Use set_target() in a control loop
          - More complex but fully reactive
          - You control smoothing
          - Good for: tracking, games, recording
```

---

## Common Mistake

Don't do this:

```python
# BAD: Multiple set_target calls scattered in code
def on_face_detected(face):
    mini.set_target(head=look_at_face(face))

def on_button_press():
    mini.set_target(head=neutral_pose)

def idle_behavior():
    mini.set_target(head=breathing_pose)
```

Do this instead:

```python
# GOOD: Single control loop, update target variables
def control_loop():
    while running:
        pose = compute_final_pose()  # Combines all inputs
        mini.set_target(head=pose)
        time.sleep(0.01)

def on_face_detected(face):
    controller.face_target = look_at_face(face)

def on_button_press():
    controller.override = neutral_pose
```
