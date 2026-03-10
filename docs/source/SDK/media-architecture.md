# Media Architecture

Understanding the media architecture of Reachy Mini is essential for effectively utilizing its audio and visual capabilities.

## Unified Architecture

The daemon always manages the camera and audio hardware via its WebRTC media backend (`webrtc_daemon.py`), regardless of whether you are using a Reachy Mini (Wireless) or Reachy Mini Lite. This unification means both models work the same way:

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

[![Reachy Mini Media Daemon](https://github.com/pollen-robotics/reachy_mini/raw/develop/docs/assets/reachymini_media_daemon.png)]()

### Client Side

The SDK `MediaManager` selects the backend automatically:

- **LOCAL**: Used when the client runs on the same machine as the daemon. Reads camera frames from IPC and opens the audio device directly via GStreamer. No encode/decode overhead.
- **WEBRTC**: Used when the client is remote. Streams camera + audio over WebRTC.
- **NO_MEDIA**: Skips all media initialisation (headless operation).

[![Reachy Mini Media Client](https://github.com/pollen-robotics/reachy_mini/raw/develop/docs/assets/reachymini_media_client.png)]()

### Web Access

Thanks to WebRTC, the audio and video streams can also be accessed directly from a web browser. For instance the [desktop application](../platforms/reachy_mini_lite/get_started.md#3--download-reachy-mini-control) uses this feature.

[![Reachy Mini Media Web](https://github.com/pollen-robotics/reachy_mini/raw/develop/docs/assets/reachymini_media_web.png)]()

## Advanced Controls

Please refer to the dedicated pages to fine-tune camera and microphone parameters for [Reachy Mini](../platforms/reachy_mini/media_advanced_controls.md) and [Reachy Mini Lite](../platforms/reachy_mini_lite/media_advanced_controls.md).
