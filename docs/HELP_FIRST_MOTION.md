# Reachy Mini -- First Motion Guide

Safely move something as fast as possible. Understand what happens when you do.

---

## Prerequisites

Before attempting motion:
1. Power supply is connected (USB alone will NOT power motors)
2. Daemon is running (`reachy-mini-daemon` for Lite, automatic for Wireless)
3. Dashboard shows robot is connected: http://localhost:8000

---

## Your First Movement: Antenna Wiggle

This is the safest possible first motion. Antennas are lightweight and cannot collide with anything.

```python
from reachy_mini import ReachyMini

with ReachyMini() as mini:
    # Wiggle right antenna up, left antenna down
    mini.goto_target(antennas=[0.5, -0.5], duration=0.5)
    # Reverse
    mini.goto_target(antennas=[-0.5, 0.5], duration=0.5)
    # Return to neutral
    mini.goto_target(antennas=[0, 0], duration=0.5)
```

Antenna values are in **radians**. Typical range: roughly -3.0 to +3.0.

---

## Your Second Movement: Head Nod

```python
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

with ReachyMini() as mini:
    # Nod down (positive pitch)
    mini.goto_target(
        head=create_head_pose(pitch=15, degrees=True),
        duration=1.0
    )
    # Return to neutral
    mini.goto_target(
        head=create_head_pose(),
        duration=1.0
    )
```

`create_head_pose()` with no arguments creates the neutral (home) position.

---

## Your Third Movement: Full Combo

```python
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
import numpy as np

with ReachyMini() as mini:
    # Look right + turn body right + spread antennas
    mini.goto_target(
        head=create_head_pose(yaw=-30, degrees=True),
        body_yaw=np.deg2rad(-20),
        antennas=[0.5, -0.5],
        duration=1.5,
        method="minjerk"
    )
    # Return to neutral
    mini.goto_target(
        head=create_head_pose(),
        body_yaw=0.0,
        antennas=[0, 0],
        duration=1.5
    )
```

---

## How to Stop Motion Immediately

### From code:
```python
# Disable all motors -- head will go limp (gravity will pull it down)
mini.disable_motors()
```

### From the dashboard:
Open http://localhost:8000 and use the motor controls to disable torque.

### Physical:
Unplug the power adapter. Motors will immediately lose torque.

**Important:** Disabling motors causes the head to drop under gravity. This is normal and not harmful, but be prepared for it.

---

## Safety Limits (Automatic)

The SDK automatically clamps all values to safe ranges. You cannot command the robot into self-collision through software.

| Joint | Safe Range |
|-------|-----------|
| Head pitch | -40 to +40 degrees |
| Head roll | -40 to +40 degrees |
| Head yaw | -180 to +180 degrees |
| Body yaw | -160 to +160 degrees |
| Head-body yaw difference | Max 65 degrees |

If you send a value outside these limits, the SDK silently clamps it to the nearest safe value. Your code will not crash.

---

## Motor Enable / Disable

```python
# Motors ON -- robot holds position (stiff)
mini.enable_motors()

# Motors OFF -- robot goes limp
mini.disable_motors()

# Gravity compensation -- robot is "soft", you can move it by hand
# and it will hold where you leave it
mini.enable_gravity_compensation()
```

### Safe Enable Pattern

When enabling motors, set the goal to the current position first. This prevents the head from jumping to a stale goal position:

```python
# Read where the head actually is
head_pose = mini.get_current_head_pose()
_, antennas = mini.get_current_joint_positions()

# Set the goal to the current position (very short duration)
mini.goto_target(
    head=head_pose,
    antennas=list(antennas) if antennas is not None else None,
    duration=0.05
)

# Now safely enable
mini.disable_motors()   # Full reset (handles mixed motor state edge case)
mini.enable_motors()
```

### Safe Disable Pattern

Go to a safe resting position before disabling torque:

```python
import numpy as np
from reachy_mini.reachy_mini import SLEEP_HEAD_POSE

antenna_angle = np.deg2rad(15)
mini.goto_target(
    SLEEP_HEAD_POSE,
    antennas=[-antenna_angle, antenna_angle],
    duration=1.0
)
mini.disable_motors()
```

---

## Interpolation Methods

`goto_target` supports four interpolation styles:

| Method | Character | Best for |
|--------|-----------|----------|
| `minjerk` | Smooth, natural acceleration/deceleration | Default. Most gestures. |
| `linear` | Constant speed | Mechanical, predictable motion |
| `ease` | Slow start and end | Gentle transitions |
| `cartoon` | Exaggerated, bouncy overshoot | Playful, expressive motion |

```python
mini.goto_target(
    head=create_head_pose(yaw=30, degrees=True),
    duration=1.0,
    method="cartoon"  # Try each one to see the difference
)
```

---

## What to Read Next

- [HELP_CONTROL_MODES.md](HELP_CONTROL_MODES.md) -- Understanding `goto_target` vs `set_target`
- [HELP_SAFETY.md](HELP_SAFETY.md) -- Physical safety around the robot
- [HELP_COMMON_ERRORS.md](HELP_COMMON_ERRORS.md) -- If something goes wrong during your first motion
