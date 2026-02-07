# Reachy Mini -- Data Flow

Make debugging logical instead of magical. Know where data comes from, where it goes, and where to probe.

---

## Primary Data Pipeline

```
┌─────────┐     ┌──────────┐     ┌─────────┐     ┌──────────┐     ┌─────────┐
│ HARDWARE │────►│ BACKEND  │────►│ ZENOH   │────►│ SDK      │────►│ USER    │
│          │     │          │     │ SERVER  │     │ CLIENT   │     │ CODE    │
│ Motors   │ 50Hz│ FK comp  │ 50Hz│ Publish │ sub │ Cache    │ get │         │
│ IMU      │────►│ State    │────►│ Topics  │────►│ Pose     │────►│         │
│ Camera   │     │ update   │     │         │     │ Joints   │     │         │
└─────────┘     └──────────┘     └─────────┘     └──────────┘     └─────────┘
                     ▲                                  │
                     │              ┌───────────────────┘
                     │              │ send_command()
                     │              ▼
                ┌────┴─────┐  ┌──────────┐
                │ IK comp  │◄─│ ZENOH    │
                │ Target   │  │ CLIENT   │
                │ joints   │  │ command  │
                └──────────┘  └──────────┘
```

---

## Zenoh Topics (Pub/Sub)

All topics are prefixed with the robot name (default: `reachy_mini`).

### Published by Daemon (Server → Client)

| Topic | Frequency | Format | Contents |
|-------|-----------|--------|----------|
| `{prefix}/joint_positions` | 50Hz | JSON | `{"head_joint_positions": [7], "antennas_joint_positions": [2]}` |
| `{prefix}/head_pose` | 50Hz | JSON | `{"head_pose": [16]}` (flattened 4x4 matrix) |
| `{prefix}/imu_data` | 50Hz | JSON | `{"accelerometer": [3], "gyroscope": [3], "quaternion": [4], "temperature": float}` |
| `{prefix}/daemon_status` | 1Hz | JSON | DaemonStatus with state, backend_status, errors |
| `{prefix}/task_progress` | On event | JSON | `{"uuid": str, "finished": bool, "error": str\|null}` |
| `{prefix}/recorded_data` | On stop | JSON | Array of timestamped joint position frames |

### Published by Client (Client → Server)

| Topic | Frequency | Format | Contents |
|-------|-----------|--------|----------|
| `{prefix}/command` | On demand | JSON | Command dict (see below) |
| `{prefix}/task` | On demand | JSON | TaskRequest (goto_target, play_move) |

### Command Message Structure

```json
{
  "torque": true,                         // Enable/disable motors
  "ids": ["stewart_1", "right_antenna"],  // Specific motors (null = all)
  "head_pose": [16 floats],              // Target 4x4 pose (flattened)
  "head_joint_positions": [7 floats],    // Target joint positions
  "body_yaw": 0.5,                       // Target body yaw (rad)
  "antennas_joint_positions": [2 floats],// Target antenna angles (rad)
  "gravity_compensation": true,           // Toggle mode
  "automatic_body_yaw": true,            // Auto body tracking
  "start_recording": true,               // Begin recording
  "stop_recording": true,                // Stop and publish
  "set_target_record": {...}             // Recording frame data
}
```

All fields are optional. Only present fields are applied.

---

## REST/WebSocket Data Flow

### State Reading

```
GET /api/state/full
  → Daemon reads backend.current_head_pose, current_joint_positions, etc.
  → Serializes to FullState Pydantic model
  → Returns JSON

WebSocket /api/state/ws/full?frequency=10
  → Every 100ms: read backend state → serialize → send
```

### Motion Commands

```
POST /api/move/goto
  Body: {head_pose, antennas, body_yaw, duration, interpolation}
  → Router creates GotoMove
  → Launches asyncio.Task with backend.goto_target()
  → Returns MoveUUID immediately
  → Task runs play_move at 100Hz, setting targets each tick
  → Control loop picks up targets at 50Hz
  → On completion: publishes to WebSocket /api/move/ws/updates

POST /api/move/set_target
  Body: {target_head_pose, target_antennas, target_body_yaw}
  → If move is running: returns {"status": "ignored"}
  → Otherwise: sets backend.target_head_pose, ik_required=True
  → Returns {"status": "ok"}
  → Next control loop tick: IK computed, joints written to hardware
```

---

## Camera Data Flow

Camera data does NOT flow through the daemon (except Wireless WebRTC):

### Lite (Direct USB)
```
Camera (USB) → OpenCV VideoCapture → mini.media.get_frame() → User code
```

### Wireless Local (on CM4)
```
Camera (CSI) → GStreamer pipeline → Unix socket → mini.media.get_frame() → User code
```

### Wireless Remote
```
Camera (CSI) → GStreamer → WebRTC encode → Network → WebRTC decode → mini.media.get_frame() → User code
```

---

## Audio Data Flow

### Recording (Microphone → User)
```
Mic array → SoundDevice InputStream callback → Ring buffer (deque)
  → mini.media.get_audio_sample() → User code
```

Buffer: deque with 60-second max capacity. Overflow drops oldest chunks.

### Playback (User → Speaker)
```
User code → mini.media.push_audio_sample(data) → Output buffer (list)
  → SoundDevice OutputStream callback → Speaker
```

Push is non-blocking. Callback drains buffer in real-time. Silence padding on underrun.

---

## Recording Data Flow

```
1. mini.start_recording()
   → ZenohClient sends {"start_recording": true}
   → ZenohServer._handle_command() sets backend.is_recording = True

2. Control loop (50Hz):
   → backend.append_record({timestamp, head_joints, antennas, head_pose, body_yaw})
   → Guarded by _rec_lock

3. mini.stop_recording()
   → ZenohClient sends {"stop_recording": true}
   → Backend: acquire _rec_lock, swap buffer, publish to Zenoh
   → ZenohClient: wait for recorded_data message (timeout 5s)
   → Returns recorded data to user
```

---

## Where to Probe When Things Go Wrong

### "Robot doesn't respond to commands"

```
Probe points (in order):
1. Is daemon running?
   → GET /api/daemon/status
   → Check state == "running"

2. Is Zenoh connected?
   → SDK: mini.client._is_alive should be True
   → Check heartbeat: client receives joint_positions within 1s

3. Is control loop running?
   → GET /api/daemon/status → backend_status.control_loop_stats
   → mean_control_loop_frequency should be ~50Hz

4. Are motors enabled?
   → GET /api/motors/status → mode should be "enabled"

5. Is a move blocking commands?
   → GET /api/move/running → should be empty list
```

### "Motion is jerky"

```
Probe points:
1. Control loop overruns?
   → backend_status.control_loop_stats.max_control_loop_interval
   → Should be <25ms. If >50ms, something is blocking the loop.

2. How often is set_target being called?
   → Add logging: timestamp each set_target call
   → Should be 50-100Hz for smooth motion

3. Multiple set_target sources?
   → Search code for all set_target/goto_target calls
   → Must be exactly ONE place calling set_target in your control loop

4. IK failures?
   → Check daemon logs for "IK error" warnings
   → Frequent IK failures = jerky motion (old target retained)
```

### "State looks stale"

```
Probe points:
1. Zenoh subscriber receiving?
   → SDK: check mini.client._last_head_pose changes on each read

2. Backend publishing?
   → Check daemon logs for publish errors

3. Network latency (Wireless)?
   → Compare timestamps between backend and client
   → >100ms latency = noticeable lag
```

### "Audio/video not working"

```
Probe points:
1. Media backend selection
   → Log mini.media.backend value
   → Ensure correct backend for your platform

2. Camera device detected?
   → OpenCV: cv2.VideoCapture(device_id).isOpened()
   → GStreamer: check pipeline element creation

3. Audio device detected?
   → SoundDevice: sd.query_devices() → look for "Reachy Mini Audio"
```

---

## Message Lifecycle: goto_target (Complete Path)

```
1. User calls mini.goto_target(head=pose, duration=1.0, method="minjerk")

2. SDK creates GotoTaskRequest, assigns UUID
   → ZenohClient publishes to {prefix}/task

3. ZenohServer receives task
   → Deserializes GotoTaskRequest
   → Creates asyncio.Task running backend.goto_target()

4. Backend.goto_target():
   a. Creates GotoMove(start=current_pose, target=pose, duration=1.0, method=minjerk)
   b. Acquires _play_move_lock
   c. Enters play_move loop at 100Hz:
      - t = time.time() - t0
      - head, antennas, body_yaw = move.evaluate(t)
      - backend.target_head_pose = head  (sets ik_required=True)
      - await asyncio.sleep(0.01)

5. Backend control loop (50Hz, separate thread):
   a. Reads current joint positions from hardware
   b. Computes FK → updates current_head_pose
   c. Sees ik_required=True → computes IK from target_head_pose
   d. Writes target_head_joint_positions to motors
   e. Publishes updated joint_positions and head_pose via Zenoh

6. Move completes (t >= duration):
   a. Releases _play_move_lock
   b. Publishes TaskProgress(uuid, finished=True) via Zenoh

7. SDK receives TaskProgress
   → Sets event on task UUID
   → goto_target() returns to user
```
