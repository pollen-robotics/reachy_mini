# Media Architecture

Understanding the media architecture of Reachy Mini is essential for effectively utilizing its audio and visual capabilities.

## Unified Architecture

The daemon always manages the camera and audio hardware via `GstMediaServer` (`media_server.py`), regardless of whether you are using a Reachy Mini (Wireless) or Reachy Mini Lite. This unification means both models work the same way:

- The **daemon** owns the physical camera and audio devices.
- **Local clients** (same machine) read camera frames from a local IPC endpoint and open the audio device directly via GStreamer — the `LOCAL` backend.
- **Remote clients** stream camera + audio over WebRTC from the daemon — the `WEBRTC` backend.
- The SDK auto-detects which backend to use based on whether the daemon's IPC endpoint is reachable.

### Daemon Side

The daemon starts its media pipeline automatically unless the `--no-media` flag is passed. It:
1. Opens the camera (platform-aware: v4l2, libcamera, DirectShow, AVFoundation, or UDP for simulation).
2. Opens the audio device (platform-aware: PulseAudio, ALSA, WASAPI, CoreAudio).
3. Feeds both into a WebRTC server (`webrtcsink`) for remote streaming.
4. Exposes raw camera frames via a local IPC endpoint (`unixfdsink` on Linux/macOS, `win32ipcvideosink` on Windows).

[![Reachy Mini Media Daemon](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/reachymini_media_daemon.png)]()

### Client Side

The SDK `MediaManager` selects the backend automatically:

- **LOCAL**: Used when the client runs on the same machine as the daemon. Reads camera frames from IPC and opens the audio device directly via GStreamer. No encode/decode overhead.
- **WEBRTC**: Used when the client is remote. Streams camera + audio over WebRTC.
- **NO_MEDIA**: Skips all media initialisation (headless operation).

[![Reachy Mini Media Client](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/reachymini_media_client.png)]()

### Web Access

Thanks to WebRTC, the audio and video streams can be consumed directly from a web browser — no native client required. Two channels are involved:

- **Signaling** runs through a central signaling server hosted on a Hugging Face Space: HF OAuth sign-in, listing the robots you can reach, and the SDP/ICE handshake that sets up the connection.
- **Media** is a direct **WebRTC peer-to-peer** connection from the robot. The browser decodes **h264** video and **opus** audio locally.

Browser apps build on the [`@pollen-robotics/reachy-mini-sdk`](https://www.npmjs.com/package/@pollen-robotics/reachy-mini-sdk) package, which ships the `ReachyMini` browser SDK plus an optional **host shell** (`./host`). The host shell provides the HF OAuth sign-in, a robot picker, and a top bar, and runs the user app in an iframe that loads the SDK — so app code only owns its own UI. Both the [Reachy Mini Control desktop app](../platforms/reachy_mini_lite/get_started.md#3--download-reachy-mini-control) and Hugging Face Spaces apps use this path.

See the [JavaScript SDK runtime reference](./javascript-sdk.md) for the API, and the [App Creation Guide](https://github.com/pollen-robotics/reachy_mini/blob/main/ts/APP_CREATION_GUIDE.md) for building and deploying a JS app.

[![Reachy Mini Media Web](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/reachymini_media_web.png)]()

## Disabling Media / Direct Hardware Access

By default the daemon owns the camera and audio devices. If you need direct access to the hardware — for example to use OpenCV, sounddevice, or a custom vision pipeline — you can deactivate the built-in media manager:

```python
from reachy_mini import ReachyMini

with ReachyMini(media_backend="no_media") as mini:
    # The daemon has released camera and audio hardware.
    # Use OpenCV, sounddevice, or any other library directly.
    import cv2
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    # Robot control still works normally.
    mini.goto_target(antennas=[0.3, -0.3], duration=0.5)

# On exit, the daemon automatically re-acquires the hardware.
```

When `media_backend="no_media"` is passed, the SDK:
1. Asks the daemon to **release** the camera and audio devices (stops the GStreamer pipeline).
2. Sets the local `MediaManager` to `NO_MEDIA` (no camera/audio through the SDK).
3. On context manager exit (`__exit__`), tells the daemon to **re-acquire** the hardware automatically.

You can also call `release_media()` and `acquire_media()` manually at any point:

```python
mini = ReachyMini()

# ... use the built-in media manager ...
frame = mini.media.get_frame()

# Switch to direct access
mini.release_media()
# ... use OpenCV, sounddevice, etc. ...

# Switch back to the SDK media manager
mini.acquire_media()
frame = mini.media.get_frame()
```

> **⚠️ Note:** Both methods are idempotent — calling `release_media()` twice is safe.

For a complete example with both OpenCV and sounddevice, see [Custom Media Manager](../examples/custom_media_manager.md).

## Cloud Backends

For Python services running off-robot (e.g. an HF Space doing inference
on the camera stream and steering the head), there's a hardware-free
aiortc client built on the same central relay protocol — see
[Cloud-Backend Consumer](./cloud-backend-consumer.md).

## Advanced Controls

Please refer to the dedicated pages to fine-tune camera and microphone parameters for [Reachy Mini](../platforms/reachy_mini/media_advanced_controls.md) and [Reachy Mini Lite](../platforms/reachy_mini_lite/media_advanced_controls.md).
