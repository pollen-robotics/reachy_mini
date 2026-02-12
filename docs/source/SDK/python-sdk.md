# Python SDK Reference

> **💡 Reminder:** The SDK now auto-detects whether it should connect over USB/localhost or over the network, so `ReachyMini()` works out of the box. You can still force a mode with `ReachyMini(connection_mode="localhost_only" | "network")` if needed.

## Movement

### Basic Control (`goto_target`)
Smooth interpolation between points. You can control `head`, `antennas`, and `body_yaw`.

```python
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
import numpy as np

with ReachyMini() as mini:
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

The media architecture is described in detail in the [Media Architecture](media-architecture.md) section. Although accesssing audio and video from the SDK is similar across Reachy Mini versions, the underlying implementation differs.

### Camera 📷

The frames of the camera can be accessed as follows :

```python
from reachy_mini import ReachyMini

with ReachyMini(media_backend="default") as mini:
    frame = mini.media.get_frame()
```
The returned frame is a numpy array with shape `(height, width, 3)` and data type `uint8`.

### IMU 🧭

> ⚠️ The IMU is only available with the wireless version of Reachy Mini

Take a look at [this example](https://github.com/pollen-robotics/reachy_mini/tree/main/examples/imu_example.py)
```python
with ReachyMini() as mini:
    imu_data = mini.imu
    accel_x, accel_y, accel_z = imu_data["accelerometer"] # (m/s^2)
    gyro_x, gyro_y, gyro_z = imu_data["gyroscope"] # (rad/s)
    quat_w, quat_x, quat_y, quat_z = imu_data["quaternion"] # (w, x, y, z)
    temperature = imu_data["temperature"] # (°C)

```


### Audio 🎙️ 🔊

Audio inputs (microphones) and outputs (speaker) is handled as follows:

```python
from reachy_mini import ReachyMini
from scipy.signal import resample
import time

with ReachyMini(media_backend="default") as mini:
    # Initialization - After this point, both audio devices (input/output) will be seen as busy by other applications!
    mini.media.start_recording()
    mini.media.start_playing()

    # Record
    samples = mini.media.get_audio_sample()

    # Resample (if needed)
    samples = resample(samples, mini.media.get_output_audio_samplerate()*len(samples)/mini.media.get_input_audio_samplerate())

    # Play
    mini.media.push_audio_sample(samples)
    time.sleep(len(samples) / mini.media.get_output_audio_samplerate())

    # Get Direction of Arrival
    # 0 radians is left, π/2 radians is front/back, π radians is right.
    doa, is_speech_detected = mini.media.get_DoA()
    print(doa, is_speech_detected)

    # Release audio devices (input/output)
    mini.media.stop_recording()
    mini.media.stop_playing()
```

**Audio data format:**
- `get_audio_sample()` returns a numpy array with shape `(samples, 2)` and data type `float32`, sampled at 16kHz.
- `push_audio_sample()` expects a numpy array with shape `(samples, 1 or 2)` and data type `float32`, sampled at 16kHz.

In both cases, the channels and samplerate information can be reliably retrieved with `get_input/output_audio_samplerate()` and `get_input/output_channels()`.

> **⚠️ Note:** `push_audio_sample()` is non-blocking, meaning it returns immediately while audio plays in the background. If you need to wait for playback completion, calculate the duration based on sample length and sample rate.

## Media Backend Options

Choose the appropriate media backend based on your Reachy Mini version and requirements:

**Reachy Mini Lite:**
- `media_backend="default"` - Uses OpenCV for camera and Sounddevice for audio (recommended for most users)
- `media_backend="gstreamer"` - Uses GStreamer for both camera and audio ([installation required](gstreamer-installation.md))

**Reachy Mini Wireless:**
- **Local execution** (running on the robot with SSH): Automatically uses `"gstreamer"`
- **Remote execution** (controlling from your computer): Automatically uses `"webrtc"`. With this backend, GStreamer runs locally on the Raspberry Pi, and streams both audio and video on the remote computer using WebRTC.

> **💡 Tip:** For wireless setups, the backend is automatically selected based on whether you're running locally or remotely. No need to specify the `media_backend` value !

> **💡 Tip:** For wireless setups, the WebRTC backend is requires a specific installation see [gstreamer-installation.md](gstreamer-installation.md). For now only the Linux platform is supported as a client. Other platforms (Windows, macOS) will be supported in [future releases](https://github.com/pollen-robotics/reachy_mini/issues/572).

## Recording Moves
You can record a motion by moving the robot (compliant mode) or sending commands, and save it for later replay.

```python
from reachy_mini import ReachyMini
with ReachyMini() as mini:
    mini.start_recording()
    # ... robot moves ...
    recorded_data = mini.stop_recording()
```

## Next Steps
* **[Browse the Examples Folder](https://github.com/pollen-robotics/reachy_mini/tree/main/examples)**
* **[AI Integrations](integration.md)**: Connect LLMs, build Apps, and publish to Hugging Face.
* **[Core Concepts](core-concept.md)**: Architecture, coordinate systems, and safety limits.

## ❓ Troubleshooting

Encountering an issue? 👉 **[Check the Troubleshooting & FAQ Guide](../troubleshooting.md)**
