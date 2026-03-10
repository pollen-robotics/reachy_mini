# Media Architecture Unification — Context

## Goal
Unify Lite and Wireless media architecture. The daemon always starts the media backend
(webrtc_daemon.py) and the client uses webrtc_client_gstreamer.py (remote) or
camera_gstreamer.py + audio_gstreamer.py (local IPC). Sounddevice and OpenCV backends
are removed.

## Key Decisions
- **RPi encoding**: Keep explicit `v4l2h264enc` for RPi. Other platforms: raw video to `webrtcsink`.
- **Sim mode**: Daemon receives UDP from MuJoCo. TODO: optimize later with direct `appsrc`.
- **`--wireless-version`**: Stays for CM4-specific tasks, no longer gates WebRTC.
- **`--no-media`**: Replaces `--deactivate-audio`. Disables all media on daemon side.
- **Client `no_media`**: Independent — daemon still streams, this client just doesn't connect.
- **Daemon `--no-media` + client auto**: Client detects via `DaemonStatus.no_media`, gets `NO_MEDIA`.
- **Daemon sounds**: `play_sound()` in `webrtc_daemon.py` via `playbin` (not through MediaManager).
- **IPC**: `unixfdsink`/`unixfdsrc` (Linux/macOS), `win32ipcvideosink`/`win32ipcvideosrc` (Windows).
- **Audio sharing**: ALSA dmix/dsnoop (Linux+asoundrc), PipeWire/Pulse native, WASAPI shared, CoreAudio.
- **`opencv` extra**: Kept for camera calibration, not as media backend.

## MediaBackend Enum (new)
```python
class MediaBackend(Enum):
    NO_MEDIA = "no_media"
    LOCAL = "local"        # IPC camera + direct audio
    WEBRTC = "webrtc"      # remote client
    DEFAULT = LOCAL        # alias
```
Deprecated aliases: `"gstreamer"`, `"sounddevice_opencv"`, `"gstreamer_no_video"`,
`"sounddevice_no_video"` → map to LOCAL with FutureWarning.

## Execution Status

### Source Code
- [x] `media/webrtc_daemon.py` — platform-aware camera/audio, play_sound, sim, IPC
- [x] `daemon/utils.py` — Windows pipe constant, is_local_camera_available
- [x] `daemon/daemon.py` — always start WebRTC (gated by no_media not wireless)
- [x] `daemon/app/main.py` — --no-media replaces --deactivate-audio
- [x] `io/protocol.py` — add no_media to DaemonStatus
- [x] `daemon/backend/abstract.py` — delegate play_sound to webrtc daemon
- [x] `media/camera_gstreamer.py` — gut to IPC reader only
- [x] `media/media_manager.py` — new enum, simplified routing
- [x] `reachy_mini.py` — simplify _configure_mediamanager
- [x] Delete `camera_opencv.py`, `audio_sounddevice.py`
- [x] `media/__init__.py` — update docstring
- [x] `pyproject.toml` — remove sounddevice extra, audio_sounddevice marker
- [x] `daemon/app/routers/daemon.py` — fix args.use_audio → args.no_media

### Tests
- [x] `test_audio.py` — remove SOUNDDEVICE refs, update MediaBackend values
- [x] `test_video.py` — remove SOUNDDEVICE_OPENCV refs, update MediaBackend values
- [x] `test_daemon.py` — use_audio → no_media
- [x] `test_undistort.py` — no change needed
- [x] `test_volume_control.py` — no change needed
- [x] `test_wireless.py` — no change needed

### Examples
- [x] `take_picture.py` — update --backend choices
- [x] `look_at_image.py` — same
- [x] `sound_play.py` — update --backend choices
- [x] `sound_record.py` — same

### Documentation
- [x] `examples/sound_play.md` — backend names updated
- [x] `examples/sound_record.md` — backend names updated
- [x] `platforms/reachy_mini_lite/media_advanced_controls.md` — remove opencv refs
- [ ] `SDK/python-sdk.md` — update backend options (if references exist)
- [ ] `troubleshooting.md` — update media_backend strings (if references exist)

## Resolved: Audio source breaks IPC video

### Root cause
`pulsesrc` (and other PulseAudio/PipeWire audio sources) provides its own clock.
When added to the pipeline, GStreamer selects the PulseAudio clock as the pipeline
clock instead of the video source's clock. `unixfdsink` cannot synchronise video
buffers against the audio clock, causing the IPC branch to stall.

This was reproducible in both Python and `gst-launch-1.0` — not Python-specific.

### Fix
Set `provide-clock=false` on the audio source element in `_configure_audio()`.
This keeps the video source (v4l2src / libcamerasrc) as the pipeline clock provider.

```python
audiosrc.set_property("provide-clock", False)
```

### Debugging journey
1. Initially thought it was Python-specific (CLI appeared to work) — turned out
   the CLI test was missing audio, so it wasn't a valid comparison.
2. Tried `sync=false` on `unixfdsink` — no effect (buffers never reached the sink).
3. Tried reordering `_configure_audio` before `_configure_video` — no effect.
4. Tried `leaky=downstream` on queue, removing `identity`, replacing `webrtcsink`
   with `fakesink` — none worked.
5. Confirmed `pulsesrc provide-clock=false` fixes it in CLI, then applied to Python.
