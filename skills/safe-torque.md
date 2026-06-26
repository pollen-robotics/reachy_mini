# Skill: Safe Torque Handling

## Motor Control Modes

Three modes selected via `setMotorMode(...)` (JS) or `enable_motors()` / `disable_motors()` / `enable_gravity_compensation()` (Python):

- **`enabled`** — torque on, position control. Motors hold the commanded pose. Default working state.
- **`disabled`** — torque off. The head is fully compliant; it falls under gravity. Used during motion recording so the user can move the head by hand.
- **`gravity_compensation`** — torque on, current control. Motors actively cancel gravity, so the head feels weightless and can be pushed around by hand without falling. Requires the Placo kinematics engine.

`enable_motors()` pins all targets to the present pose before flipping torque on, so the head never snaps to an old goal. Set targets *after* enabling, not before — targets set before are overwritten.

## When to Use

- Enabling or disabling motor torque
- Recording motion (compliance mode)
- Any app that toggles motors on/off

---

## The Problem

When you toggle motor torque carelessly:
- **Disabling torque**: Head falls (gravity)
- **Enabling torque**: Head jumps to old goal position

Both cause jerky, unpleasant motion.

---

## Before Disabling Torque

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

---

## Before Enabling Torque

Set goal to current position first (prevents jump to old goal):

```python
def _goto_current_pose(reachy_mini, duration=0.05):
    """Set goal to current position to prevent jumps."""
    head_pose = reachy_mini.get_current_head_pose()
    _, antennas = reachy_mini.get_current_joint_positions()

    reachy_mini.goto_target(
        head=head_pose,
        antennas=list(antennas) if antennas is not None else None,
        duration=duration,
    )
```

---

## Known Bug: Partial Motor Enable/Disable

There is a known edge case when enabling/disabling only a **subset** of motors. The workaround is to call a full disable before enabling:

```python
def _safe_enable_motors(self, reachy_mini: ReachyMini) -> None:
    """Enable motors safely, handling edge cases."""
    self._goto_current_pose(reachy_mini, duration=0.05)
    reachy_mini.disable_motors()  # Needed to handle edge case with mixed motor states
    reachy_mini.enable_motors()
```

**Reference:** See `~/reachy_mini_resources/fire_nation_attacked/fire_nation_attacked/main.py` for a working implementation of this pattern.

> If `~/reachy_mini_resources/` doesn't exist, run `skills/setup-environment.md` to clone reference apps.

---

## Complete Safe Enable Pattern

```python
def safe_enable_motors(reachy_mini: ReachyMini) -> None:
    """Enable motors without jerky motion."""
    # 1. Read current position
    head_pose = reachy_mini.get_current_head_pose()
    _, antennas = reachy_mini.get_current_joint_positions()

    # 2. Set goal to current (very short duration)
    reachy_mini.goto_target(
        head=head_pose,
        antennas=list(antennas) if antennas is not None else None,
        duration=0.05,
    )

    # 3. Full disable (handles mixed motor state edge case)
    reachy_mini.disable_motors()

    # 4. Enable all motors
    reachy_mini.enable_motors()
```

---

## Reference Implementation

**Best example:** `~/reachy_mini_resources/marionette/marionette/main.py`

This app toggles torque for motion recording and demonstrates the full safe pattern.
