# Python SDK Reference

> **Heads up:** The Python and **[JavaScript](javascript-sdk.md)** SDKs both give you full control of the robot — they simply target different audiences. If you want to build apps that are easy to share (zero-install, open a link in a browser), the JavaScript/Web path is usually the better fit. The Python SDK shines for scripting, control loops, and code running directly on the robot.

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

**Interpolation methods:** `linear`, `minjerk` (default), `ease_in_out`, `cartoon`.

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

### Head Tracking 👀

The daemon can track the closest face and turn the head to follow it (aiming at the nose). Detection runs inside the daemon.

```python
from reachy_mini import ReachyMini

with ReachyMini() as mini:
    mini.start_head_tracking()
    face = mini.get_tracked_face()  # detected, x, y in [-1, 1], roll
    mini.stop_head_tracking()
```

`start_head_tracking(weight=...)` blends tracking with application motion: `1.0` lets tracking own the head, `0.0` pauses detection (freeing the head and CPU) without stopping the tracker, so applications can toggle it cheaply per turn. See the [Head Tracking example](../examples/head_tracking.md).

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

### Head Wobbling

With wobbling enabled, the head moves with the speech that Reachy Mini plays (`play_sound()`, `push_audio_sample()`, incoming WebRTC audio and the daemon's own sounds):

```python
from reachy_mini import ReachyMini

with ReachyMini() as mini:
    mini.enable_wobbling()
    mini.media.play_sound("speech.wav")
    ...
    mini.disable_wobbling()
```

See the [Sound TTS example](../examples/sound_tts.md).

The audio analysis is done by a speech tapper. The official wobbler (v0) is the default and needs no configuration. Other tappers are opt-in, selected with the `WOBBLER_VERSION` environment variable in the process that runs the wobbler (the daemon, or your script with the local media backend):

| `WOBBLER_VERSION` | Tapper |
|---|---|
| unset or `v0` | Official wobbler. |
| `v4` | Strict silence gate (the head is exactly still when nobody speaks), speaker-relative loudness, a gesture on each syllable onset. |
| `v5` | v4 plus a head pitch that follows the intonation. |
| `v6` | Prosody wobbler: v5 plus phrase-level gestures and an optional emotional colouring. |

An unknown value logs a warning and falls back to the official wobbler.

The Prosody wobbler (v6) takes two optional inputs. Without them it is neutral, that is v5 plus a small yaw drift at each phrase start and a short tilt at each phrase end (up on a question, down on a statement).

- `WOBBLER_EMOTION`: `neutral` (default), `angry`, `sassy`, `sad` or `pleading`. An unknown name logs a warning and falls back to `neutral`.
- `WOBBLER_ENERGY`: amplitude scale, `1.0` by default, clamped to `[0, 1.5]`.

```bash
WOBBLER_VERSION=v6 WOBBLER_EMOTION=sassy WOBBLER_ENERGY=1.2 reachy-mini-daemon
```

The colouring can also change while the robot speaks, with `HeadWobbler.set_emotion(emotion, energy=None)` (`reachy_mini.motion.head_wobbler`). It cross-fades over about 300 ms, keeps the energy when omitted, survives a wobbler reset, and does nothing with the tappers that have no emotion input. It is not exposed through `ReachyMini` or the REST API yet.

All tappers only use numpy, there is no extra dependency. How v4 to v6 work and how they were tuned is written up in [examples/wobbler_lab](https://github.com/pollen-robotics/reachy_mini/tree/main/examples/wobbler_lab).

## Media Backend Options

Choose the appropriate media backend based on your Reachy Mini version and requirements:

- `media_backend="default"` - Auto-detects the best backend: LOCAL when running on the same machine as the daemon, WEBRTC when remote (recommended for most users).
- `media_backend="local"` - Forces the LOCAL backend (GStreamer IPC camera + GStreamer audio). Use when running on the same machine as the daemon.
- `media_backend="webrtc"` - Forces the WEBRTC backend. The daemon streams H.264 video and Opus audio over WebRTC to the client.
- `media_backend="no_media"` - Deactivates the media manager and tells the daemon to release camera and audio hardware. Use this when you need direct access via OpenCV, sounddevice, or any other external library. The hardware is automatically re-acquired when the context manager exits. See [Media Architecture - Disabling Media](media-architecture.md#disabling-media--direct-hardware-access) and the [Custom Media Manager](../examples/custom_media_manager.md) example.

> **💡 Tip:** For most setups, the backend is automatically selected based on whether you're running locally or remotely. No need to specify the `media_backend` value!

> **💡 Tip:** The WebRTC backend requires GStreamer to be installed on the client machine. See [GStreamer Installation](gstreamer-installation.md). For now only Linux is fully supported as a remote client. Other platforms (Windows, macOS) will be supported in [future releases](https://github.com/pollen-robotics/reachy_mini/issues/572).

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
