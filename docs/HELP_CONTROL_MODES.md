# Reachy Mini -- Control Modes Guide

Clarify the mental models. Most bugs are mode misunderstandings, not code errors.

---

## The Two Motion Methods

Reachy Mini has exactly two ways to move:

| Method | What it does | When to use |
|--------|-------------|-------------|
| `goto_target()` | Smooth interpolation over a duration | **Default choice.** Gestures, emotions, choreography. |
| `set_target()` | Sets position immediately, no interpolation | Real-time control loops (tracking, games, joystick). |

### `goto_target()` -- Smooth, Blocking

```python
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

with ReachyMini() as mini:
    mini.goto_target(
        head=create_head_pose(yaw=30, pitch=10, degrees=True),
        duration=1.5,
        method="minjerk"
    )
```

**Key behaviors:**
- Blocks until the motion completes.
- You cannot react to external input during the motion.
- The robot commits to completing the movement.
- Minimum useful duration: ~0.5 seconds.

**Interpolation methods:**
| Method | Character |
|--------|-----------|
| `minjerk` | Natural, smooth (default) |
| `linear` | Constant speed |
| `ease` | Slow start and end |
| `cartoon` | Exaggerated, bouncy |

### `set_target()` -- Instant, Non-blocking

```python
import time
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

with ReachyMini() as mini:
    while True:
        pose = compute_target()  # Your logic here
        mini.set_target(head=pose)
        time.sleep(0.01)  # ~100Hz
```

**Key behaviors:**
- Returns immediately (non-blocking).
- Must be called continuously in a loop.
- You are responsible for smooth motion (send frequent updates).
- If you stop calling it, the robot holds the last sent position.

---

## Decision Flowchart

```
Does your app need to react to input in real-time?
|
+-- NO --> Use goto_target()
|          Simple code, guaranteed smooth motion.
|          Good for: emotions, dances, scripted sequences.
|
+-- YES -> Use set_target() in a control loop
           More complex, but fully reactive.
           Good for: face tracking, games, recording, teleoperation.
```

---

## Motor Modes

Separate from motion methods, motors have three torque states:

| Mode | Method | Behavior |
|------|--------|----------|
| **Stiff** | `mini.enable_motors()` | Motors hold position. Normal operating mode. |
| **Limp** | `mini.disable_motors()` | No power. Head drops under gravity. |
| **Compliant** | `mini.enable_gravity_compensation()` | Soft. You can move the head by hand and it stays where you leave it. |

### When to switch modes:

| Use case | Sequence |
|----------|----------|
| Normal operation | `enable_motors()` then `goto_target()` / `set_target()` |
| Teaching by demonstration | `enable_gravity_compensation()` then `start_recording()` |
| Shutting down | `goto_target(SLEEP_HEAD_POSE)` then `disable_motors()` |
| Idle (save power) | `disable_motors()` |

---

## Simulation vs Real Hardware

| Behavior | Simulation (MuJoCo) | Real Hardware |
|----------|---------------------|---------------|
| Motor response | Instant, perfect | Subject to PID tuning, inertia |
| Gravity compensation | Works | Works (requires Placo kinematics) |
| Camera | Not available | Available |
| Audio | May not work | Available |
| Safety limits | Applied identically | Applied identically |
| Timing | May differ slightly | Real-world timing |

**Important:** Code that works in simulation will work on hardware. But **timing-sensitive** code (control loops, audio sync) may need tuning on real hardware.

To run without needing media (cameras/audio), initialize with:
```python
with ReachyMini(media_backend="no_media") as mini:
    # Works in sim or on hardware without camera/mic access
```

---

## Common Mistakes

### Mistake 1: Mixing `goto_target` and `set_target`
```python
# BAD: These fight each other
mini.goto_target(head=pose_a, duration=1.0)  # Starts interpolation
mini.set_target(head=pose_b)  # Interrupts mid-motion
```

Pick one method per behavior phase. If you need to transition between modes, complete the `goto_target` first.

### Mistake 2: Calling `set_target` from multiple places
```python
# BAD: Race condition, jerky motion
def on_face_detected(face):
    mini.set_target(head=look_at_face(face))

def idle_behavior():
    mini.set_target(head=breathing_pose)
```

**Fix:** One control loop, one `set_target()` call. Update target variables from callbacks.

```python
# GOOD: Single control loop
def control_loop():
    while running:
        pose = compute_final_pose()  # Combines all inputs
        mini.set_target(head=pose)
        time.sleep(0.01)
```

### Mistake 3: `set_target` too slowly
If you call `set_target()` below 30Hz, motion will look jerky. Aim for 50--100Hz.

### Mistake 4: Enabling motors without setting goal to current position
The robot will jump to the last goal position. See HELP_FIRST_MOTION.md for the safe enable pattern.

---

## Control Loop Template

For apps that need real-time reactivity:

```python
import time
import threading
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

class MyApp:
    def __init__(self):
        self.stop = threading.Event()
        self.target_yaw = 0.0
        self.target_pitch = 0.0

    def control_loop(self, mini):
        """The ONLY place that calls set_target."""
        while not self.stop.is_set():
            pose = create_head_pose(
                yaw=self.target_yaw,
                pitch=self.target_pitch,
                degrees=True
            )
            mini.set_target(head=pose)
            time.sleep(0.01)  # ~100Hz

    def update_from_sensor(self, yaw, pitch):
        """Called from other threads."""
        self.target_yaw = yaw
        self.target_pitch = pitch

app = MyApp()
with ReachyMini() as mini:
    thread = threading.Thread(target=app.control_loop, args=(mini,))
    thread.start()
    # ... update targets from sensor callbacks ...
    app.stop.set()
    thread.join()
```

---

## Frequency Guidelines

| Loop frequency | Quality |
|----------------|---------|
| 100 Hz | Excellent. Real-time tracking, games. |
| 50 Hz | Good. Most interactive apps. |
| 30 Hz | Minimum for acceptable smoothness. |
| < 30 Hz | Visibly jerky. Not recommended. |
