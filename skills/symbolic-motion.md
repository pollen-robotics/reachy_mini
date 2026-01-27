# Skill: Symbolic Motion Definition

## When to Use

- Creating rhythmic or repetitive motion (nodding, swaying, dancing)
- Want to define motion mathematically rather than recording it
- LLM needs to generate or modify motion dynamically
- Memory-efficient motion (function vs thousands of data points)

---

## Reference Implementation

**Primary reference:** `~/reachy_mini_resources/reachy_mini_dances_library/src/reachy_mini_dances_library/rhythmic_motion.py`

> If this folder doesn't exist, run `skills/setup-environment.md` to clone reference apps.

---

## Core Concept

Define motion as mathematical functions of time:

```python
import numpy as np
from dataclasses import dataclass

@dataclass
class MoveOffsets:
    position_offset: np.ndarray   # [x, y, z] in meters
    orientation_offset: np.ndarray  # [roll, pitch, yaw] in radians
    antennas_offset: np.ndarray   # [left, right] in radians

def simple_nod(t_beats: float, amplitude_rad: float = 0.2) -> MoveOffsets:
    """Nodding motion synchronized to beat."""
    pitch = amplitude_rad * np.sin(2 * np.pi * t_beats)
    return MoveOffsets(
        position_offset=np.zeros(3),
        orientation_offset=np.array([0, pitch, 0]),
        antennas_offset=np.zeros(2),
    )
```

---

## Using in Control Loop

```python
import time

bpm = 120  # beats per minute
start_time = time.monotonic()

while running:
    elapsed = time.monotonic() - start_time
    t_beats = elapsed * bpm / 60.0  # Convert to beat units

    offsets = simple_nod(t_beats, amplitude_rad=0.3)

    # Apply offsets to base pose
    final_pitch = base_pitch + np.rad2deg(offsets.orientation_offset[1])
    pose = create_head_pose(pitch=final_pitch, degrees=True)

    mini.set_target(head=pose)
    time.sleep(0.01)
```

---

## Advantages Over Recorded Motion

| Aspect | Recorded | Symbolic |
|--------|----------|----------|
| Storage | Thousands of frames | Single function |
| Tunability | Re-record to change | Adjust parameters |
| BPM sync | Fixed or complex resampling | Natural (t_beats) |
| LLM generation | Can't easily modify | Can generate/tweak code |
| Variations | Need multiple recordings | Parameterize |

---

## Common Patterns

### Head Bob (Vertical)

```python
def head_bob(t_beats, amplitude_mm=10):
    z = amplitude_mm * 0.001 * np.sin(2 * np.pi * t_beats)
    return MoveOffsets(
        position_offset=np.array([0, 0, z]),
        orientation_offset=np.zeros(3),
        antennas_offset=np.zeros(2),
    )
```

### Side-to-Side Sway

```python
def sway(t_beats, amplitude_rad=0.15):
    roll = amplitude_rad * np.sin(2 * np.pi * 0.5 * t_beats)  # Half tempo
    return MoveOffsets(
        position_offset=np.zeros(3),
        orientation_offset=np.array([roll, 0, 0]),
        antennas_offset=np.zeros(2),
    )
```

### Antenna Wave

```python
def antenna_wave(t_beats, amplitude_rad=0.3, phase_offset=np.pi/4):
    left = amplitude_rad * np.sin(2 * np.pi * t_beats)
    right = amplitude_rad * np.sin(2 * np.pi * t_beats + phase_offset)
    return MoveOffsets(
        position_offset=np.zeros(3),
        orientation_offset=np.zeros(3),
        antennas_offset=np.array([left, right]),
    )
```

### Combining Moves

```python
def dance_move(t_beats):
    nod = simple_nod(t_beats, amplitude_rad=0.15)
    bob = head_bob(t_beats, amplitude_mm=8)
    wave = antenna_wave(t_beats, amplitude_rad=0.2)

    return MoveOffsets(
        position_offset=bob.position_offset,
        orientation_offset=nod.orientation_offset,
        antennas_offset=wave.antennas_offset,
    )
```

---

## Tips

- Use `t_beats` (not raw seconds) for music-synced motion
- Keep amplitudes reasonable (check safety limits in agents.md)
- Test with different BPMs to ensure motion scales well
- Combine simple functions for complex motion
