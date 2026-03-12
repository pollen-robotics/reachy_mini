# Media Architecture Unification — Context

## Goal
Unify Lite and Wireless media architecture. The daemon always starts the media backend
(`media_server.py` / `GstMediaServer`) and the client uses `webrtc_client_gstreamer.py`
(remote) or `camera_gstreamer.py` + `audio_gstreamer.py` (local IPC). Sounddevice and
OpenCV backends are removed.

## Key Decisions
- **RPi encoding**: Keep explicit `v4l2h264enc` for RPi. Other platforms: raw video to `webrtcsink`.
- **Sim mode**: Daemon receives UDP from MuJoCo. TODO: optimize later with direct `appsrc`.
- **`--wireless-version`**: Stays for CM4-specific tasks, no longer gates WebRTC.
- **`--no-media`**: Replaces `--deactivate-audio`. Disables all media on daemon side.
- **Client `no_media`**: Independent — daemon still streams, this client just doesn't connect.
- **Daemon `--no-media` + client auto**: Client detects via `DaemonStatus.no_media`, gets `NO_MEDIA`.
- **Daemon sounds**: `play_sound()` in `GstMediaServer` via `playbin` (not through MediaManager).
- **IPC**: `unixfdsink`/`unixfdsrc` (Linux/macOS), `win32ipcvideosink`/`win32ipcvideosrc` (Windows).
- **Audio sharing**: PipeWire/PulseAudio handles concurrent access — no special ALSA config needed.
- **`opencv` extra**: Kept for camera calibration, not as media backend.
- **No ABCs**: `CameraBase` and `AudioBase` removed. Concrete classes are standalone.
- **DoA extracted**: Direction of Arrival logic in `audio_doa.py`, not coupled to audio playback.
- **Camera specs propagation**: Daemon detects camera type, broadcasts `camera_specs_name` in
  `DaemonStatus`. Clients resolve specs via `get_camera_specs_by_name()`. REST: `GET /api/camera/specs`.
- **Fallback specs**: Unknown/empty camera name → `ReachyMiniLiteCamSpecs` with `logger.warning()`.
- **Keep ArducamSpecs**: Still valid for beta robots.

## MediaBackend Enum
```python
class MediaBackend(Enum):
    NO_MEDIA = "no_media"
    LOCAL = "local"        # IPC camera + direct audio
    WEBRTC = "webrtc"      # remote client
    DEFAULT = LOCAL        # alias
```
Deprecated aliases: `"gstreamer"`, `"sounddevice_opencv"`, `"gstreamer_no_video"`,
`"sounddevice_no_video"` → map to LOCAL with FutureWarning.

## Commits on Branch (947-unify-gstreamer-wireless-and-lite-architecture)

1. **`73e5b304`** — Unify media architecture: daemon always owns camera/audio, clients use LOCAL or WEBRTC
2. **`584c3140`** — Fix IPC video freeze when audio source is in pipeline (`provide-clock=false`)
3. **`bda75bbb`** — Remove CameraBase/AudioBase ABCs, extract DoA into `audio_doa.py`, fix daemon DoA route
4. **`578d621f`** — Fix `backend.audio` AttributeError in volume routes
5. **`0896588b`** — Propagate camera specs from daemon to client, add `GET /api/camera/specs`
6. **`b7726207`** — Add IPC video source fixture (`conftest.py`), make video tests self-contained
7. **`e81b0ec6`** — Rename `GstWebRTC` → `GstMediaServer`, `webrtc_daemon.py` → `media_server.py`
8. **`a6eee23e`** — Fix wake-up sound not playing on daemon startup (media server setup before wake_up)

## Execution Status

### Source Code
- [x] `media/media_server.py` — platform-aware camera/audio, play_sound, sim, IPC (renamed from webrtc_daemon.py)
- [x] `daemon/utils.py` — Windows pipe constant, is_local_camera_available
- [x] `daemon/daemon.py` — always start media server (gated by no_media not wireless), media setup before wake_up
- [x] `daemon/app/main.py` — --no-media replaces --deactivate-audio
- [x] `io/protocol.py` — add no_media + camera_specs_name to DaemonStatus
- [x] `daemon/backend/abstract.py` — delegate play_sound to media server, setup_media_server(), DoA init
- [x] `media/camera_gstreamer.py` — IPC reader, accepts camera_specs param, standalone (no ABC)
- [x] `media/audio_gstreamer.py` — standalone (no ABC)
- [x] `media/audio_doa.py` — DoA extraction from AudioBase
- [x] `media/webrtc_client_gstreamer.py` — standalone, accepts camera_specs param
- [x] `media/media_manager.py` — new enum, simplified routing, Union types, forwards camera_specs
- [x] `media/camera_constants.py` — MujocoCameraSpecs.name, get_camera_specs_by_name(), _SPECS_BY_NAME
- [x] `reachy_mini.py` — resolve camera specs from daemon_status, pass to MediaManager
- [x] Delete `camera_base.py`, `audio_base.py`, `camera_opencv.py`, `audio_sounddevice.py`
- [x] `media/__init__.py` — update docstring
- [x] `pyproject.toml` — remove sounddevice extra, audio_sounddevice marker, add ipc_resolution marker
- [x] `daemon/app/routers/daemon.py` — fix args.use_audio → args.no_media
- [x] `daemon/app/routers/state.py` — backend.doa instead of backend.audio
- [x] `daemon/app/routers/volume.py` — backend.play_sound() instead of backend.audio.play_sound()
- [x] `daemon/app/routers/camera.py` — new GET /api/camera/specs endpoint

### Tests
- [x] `test_audio.py` — remove SOUNDDEVICE refs, update MediaBackend values
- [x] `test_video.py` — uses ipc_video_source fixture, split WebRTC test, self-contained
- [x] `conftest.py` — IPC video source pytest fixture (videotestsrc → unixfdsink)
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

## Pre-existing Bugs Found and Fixed
- **Daemon `/api/state/doa` route crash**: `backend.audio` never existed on `Backend`. Fixed → `backend.doa`.
- **Daemon `/api/volume/set` and `/api/volume/test-sound` crash**: Same `backend.audio` pattern. Fixed → `backend.play_sound()`.
- **Wake-up sound not playing**: `setup_media_server()` was called after `wake_up()`, so `play_sound()` was a no-op. Fixed by reordering startup sequence.

## Pre-existing Test Failures (not caused by our changes)
- `test_play_sound[MediaBackend.WEBRTC]` — requires network to `reachy-mini.local`
- `test_set_output_volume_clamps`, `test_set_and_restore_*_volume` — hardware-dependent
- `test_app_manager` — app install metadata/discovery issue
- All GStreamer files show `"GLib" is unknown import symbol` — PyGObject type stub issue

## Resolved: Audio source breaks IPC video

### Root cause
`pulsesrc` (and other PulseAudio/PipeWire audio sources) provides its own clock.
When added to the pipeline, GStreamer selects the PulseAudio clock as the pipeline
clock instead of the video source's clock. `unixfdsink` cannot synchronise video
buffers against the audio clock, causing the IPC branch to stall.

### Fix
Set `provide-clock=false` on the audio source element in `_configure_audio()`.

### IPC pipeline findings
- `unixfdsink` behind a `tee` with `webrtcsink` needs `identity drop-allocation=true` + `videoconvert` to produce memfd-backed buffers. Standalone `videotestsrc → videoconvert → unixfdsink` works without the workaround.
- `unixfdsrc → queue → appsink` reads frames correctly when socket exists.
- IPC works end-to-end on both Lite and Wireless (confirmed).
