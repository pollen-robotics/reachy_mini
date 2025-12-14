# Python SDK Reference

## Movement

### Basic Control (`goto_target`)
Smooth interpolation between points. You can control `head`, `antennas`, and `body_yaw`.

```python
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
import numpy as np

mini = ReachyMini()

# Move everything at once
mini.goto_target(
    head=create_head_pose(z=10, mm=True),    # Up 10mm
    antennas=np.deg2rad([45, 45]),           # Antennas out
    body_yaw=np.deg2rad(30),                 # Turn body
    duration=2.0,                            # Take 2 seconds
    method="minjerk"                         # Smooth acceleration
)
```

**Interpolation methods:** `linear`, `minjerk` (default), `ease`, `cartoon`.

### Instant Control (`set_target`)
Bypasses interpolation. Use this for high-frequency control (e.g., following a joystick or generated trajectory).

## Sensors & Media

### Camera 📷
Get raw frames (OpenCV format) or use the GStreamer backend for low latency.

```python
import cv2
frame = mini.media.get_frame()
cv2.imshow("Reachy", frame)
```

### Audio 🎙️ 🔊
Reachy uses `sounddevice` for audio I/O.

```python
# Record
sample = mini.media.get_audio_sample()

# Play
mini.media.play_audio("hello.wav")
# Or push raw chunks:
# mini.media.push_audio_sample(chunk)
```

## Recording Moves
You can record a motion by moving the robot (compliant mode) or sending commands, and save it for later replay.

```python
mini.start_recording()
# ... robot moves ...
recorded_data = mini.stop_recording()
```
