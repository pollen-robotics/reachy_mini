# Skill: Control Loops

## When to Use

- Building an app that needs real-time reactivity (face tracking, games, joystick control)
- User asks about `set_target()` or continuous motion control
- App needs to respond to sensor input at > 10Hz (ideally 50Hz+)

## Quick Check

If the app only needs choreographed, predetermined motions, use `goto_target()` instead (see `motion-philosophy.md`).

---

## The Core Rule

**One control loop, one place calling `set_target()`, running at fixed frequency (~100Hz).**

```python
import time
from reachy_mini import ReachyMini

with ReachyMini() as mini:
    while not stop_event.is_set():
        # Compute target pose (can change every tick!)
        pose = compute_current_target_pose()

        # Send it
        mini.set_target(head=pose, antennas=antennas)

        # Maintain frequency
        time.sleep(0.01)  # ~100Hz
```

## Why This Matters

- **Single control point** prevents race conditions and jerky motion
- **Continuous targets** maintain "illusion of life" - even when idle, keep sending
- **Modify the pose, not where you call set_target** - complex behavior = change what pose you compute, don't add more set_target calls

---

## Reference Implementation: moves.py

**Best example:** `~/reachy_mini_resources/reachy_mini_conversation_app/src/reachy_mini_conversation_app/moves.py`

> If this folder doesn't exist, run `skills/setup-environment.md` to clone reference apps.

This file demonstrates control with:

### Primary vs Secondary Moves

- **Primary moves** (emotions, dances, goto, breathing) - mutually exclusive, run sequentially
- **Secondary moves** (face tracking, speech sync) - additive offsets layered on top
- **Pose fusion** - combining primary pose with secondary offsets

### Phase-Aligned Timing

Uses `time.monotonic()` to measure elapsed time accurately:
- Avoids wall-clock jumps from NTP sync, sleep/wake, etc.
- Ensures consistent timing regardless of system load

### Threading Model

- Dedicated worker thread owns robot output
- Other threads communicate via queues
- Never call `set_target()` from multiple threads

---

## Basic Control Loop Template

```python
import time
import threading
import numpy as np
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

class MyController:
    def __init__(self):
        self.stop_event = threading.Event()
        self.target_yaw = 0.0
        self.target_pitch = 0.0

    def control_loop(self, mini: ReachyMini):
        """Main control loop - only place that calls set_target."""
        while not self.stop_event.is_set():
            # Build pose from current targets
            pose = create_head_pose(
                yaw=self.target_yaw,
                pitch=self.target_pitch,
                degrees=True
            )

            # Send to robot
            mini.set_target(head=pose)

            # ~100Hz
            time.sleep(0.01)

    def update_target(self, yaw: float, pitch: float):
        """Called from other threads/callbacks to update targets."""
        self.target_yaw = yaw
        self.target_pitch = pitch

# Usage
controller = MyController()
with ReachyMini() as mini:
    loop_thread = threading.Thread(target=controller.control_loop, args=(mini,))
    loop_thread.start()

    # Update targets from elsewhere (e.g., tracking callback)
    controller.update_target(yaw=30, pitch=10)

    # ... eventually
    controller.stop_event.set()
    loop_thread.join()
```

---

## Common Patterns

### Adding Idle Motion (Breathing)

```python
def compute_breathing_offset(t: float) -> float:
    """Subtle pitch oscillation for 'alive' feeling."""
    return 2.0 * np.sin(2 * np.pi * 0.2 * t)  # 0.2 Hz, 2 degree amplitude

# In control loop:
t = time.monotonic()
breathing = compute_breathing_offset(t)
pose = create_head_pose(yaw=target_yaw, pitch=target_pitch + breathing, degrees=True)
```

---

## Frequency Considerations

| Frequency | Use case |
|-----------|----------|
| 100 Hz | Real-time tracking, games |
| 50 Hz | Most interactive apps |
| 30 Hz | Minimum for smooth motion |
| < 30 Hz | Might look jerky |
