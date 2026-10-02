# 03 - Vision

Status: **already implemented in the daemon** (this doc describes the existing
surface; the conversation service reuses it, it does not rebuild it).

Vision is a daemon capability, not part of the conversation service. Two things
matter to us: **camera frames** (a read-only fan-out any consumer can read) and
**face/head tracking** (a daemon-side worker that feeds the gaze layer). Scene
understanding ("what does the robot see?") is **cloud**, via the realtime model,
not a local VLM.

## Public interface (what already exists)

### Camera frames

One `GstMediaServer` owns the camera and fans it out over two channels from a
single pipeline: a **WebRTC video track** (remote clients) and a **local IPC
branch** (Unix socket `/tmp/reachymini_camera_socket`, Windows named pipe). BGR,
paced at ~10 fps on the local feed.

| Entry point | Where | Notes |
|---|---|---|
| `reachy.media.get_frame()` / `get_frame_jpeg()` | `media/media_manager.py` | The public "give me a frame" API (LOCAL IPC or WEBRTC, auto-selected) |
| Unix socket / Win pipe | `daemon/utils.py` (`CAMERA_SOCKET_PATH` / `CAMERA_PIPE_NAME`) | Raw frame fan-out; multiple readers OK |
| `GET /camera/specs` | `daemon/app/routers/camera.py` | Name, resolutions, intrinsics `K`, distortion `D`. No raw-frame HTTP endpoint. |

The camera is a **read-only fan-out**: face tracking, the `camera` tool and remote
WebRTC video all consume it concurrently. No exclusive lock (unlike audio).

### Face / head tracking

A `FaceTracker` daemon thread reads the same IPC socket, detects the largest/nearest
face, smooths it, and exposes an aim the motion layer eases toward.

| Surface | Method / route / command |
|---|---|
| SDK | `start_head_tracking(weight=1.0)`, `stop_head_tracking()`, `get_tracked_face()`, `look_at_image(...)`, `look_at_world(...)` |
| HTTP | `POST /media/tracking/enable` (`{weight}`), `POST /media/tracking/disable`, `GET /media/tracking/face` |
| Data-channel RPC | `set_head_tracking` (`{enabled, weight}`), `get_tracked_face` |
| Status | `DaemonStatus.face_target` = `{ detected, x, y, roll, ts }` |

`weight` in `[0, 1]` blends the gaze aim; `weight=0` pauses the worker without
tearing it down. Tracking output (normalized center + roll) is what the motion
service consumes as its **gaze layer**.

### Detector

- **YuNet on ONNX Runtime**, CPU, single-thread. Model pulled from HF Hub
  (`pollen-robotics/face_detection_yunet_2026may`) on first tracking start.
- Not YOLO, not MediaPipe (those only ever existed in an app-side experiment).

## How the conversation service uses it

- **Gaze**: the motion service ([`02-motion-service.md`](./02-motion-service.md))
  consumes the tracker's aim as its additive gaze layer. It does not run detection
  itself.
- **`look` / `camera` tool** ([`04-tools.md`](./04-tools.md)): grabs one frame via
  `media.get_frame()`, JPEG-encodes it, and hands it to the **cloud realtime model**
  with the user's question. The picture is understood in the cloud, not on-device.

## Key facts / decisions

- **Reuse, don't rebuild.** Camera fan-out, YuNet tracking and look-at geometry are
  already daemon-side and battle-tested. The conversation service wires into them.
- **No local VLM in the daemon.** Scene understanding is cloud. A local SmolVLM2 was
  prototyped in an app branch but is not part of the daemon and is out of scope
  here.
- **Read-only fan-out.** The camera has no single-owner constraint, so vision never
  contends with the conversation the way audio does
  ([`01-conversation-engine.md`](./01-conversation-engine.md)).
- **Tracking is opt-in.** The worker is lazily created on the first
  `enable`/`start_head_tracking` call and runs at lowest CPU priority.

## Not in scope for this design

Vision is pre-existing daemon infrastructure. There is no PR1 work item here beyond
**wiring** the gaze layer and the `camera`/`look` tool to the surfaces above. Any
on-device VLM, alternative detectors, or richer perception is a separate daemon
effort, not part of the conversation service.
