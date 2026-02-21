# Reachy Mini -- System Architecture

How the robot thinks. A mental map for contributors.

---

## High-Level Block Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                     USER CODE / APP                                  │
│  (Python script, ReachyMiniApp subclass, or HTTP/WebSocket client)  │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │     SDK  (ReachyMini)        │
              │  reachy_mini/reachy_mini.py  │
              │                              │
              │  - goto_target / set_target  │
              │  - look_at_image / world     │
              │  - enable/disable motors     │
              │  - media (camera, audio)     │
              │  - recording / playback      │
              └──────────────┬──────────────┘
                             │
              ┌──────────────┴──────────────┐
              │    ZENOH TRANSPORT LAYER     │
              │   io/zenoh_client.py         │
              │   io/zenoh_server.py         │
              │                              │
              │  Pub/Sub over localhost:7447  │
              │  or network (peer scouting)  │
              └──────────────┬──────────────┘
                             │
┌────────────────────────────┴─────────────────────────────────────────┐
│                         DAEMON                                       │
│  daemon/app/main.py  (FastAPI + Uvicorn, port 8000)                 │
│                                                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────────┐│
│  │  REST API        │  │  WebSocket API  │  │  Dashboard (HTML)    ││
│  │  /api/move/      │  │  /api/state/ws  │  │  /                   ││
│  │  /api/state/     │  │  /api/move/ws   │  │  /settings           ││
│  │  /api/motors/    │  │                 │  │                      ││
│  │  /api/apps/      │  │                 │  │                      ││
│  │  /api/daemon/    │  │                 │  │                      ││
│  └────────┬────────┘  └────────┬────────┘  └──────────────────────┘│
│           │                    │                                     │
│  ┌────────┴────────────────────┴─────────┐                          │
│  │         DAEMON ORCHESTRATOR           │                          │
│  │         daemon/daemon.py              │                          │
│  │                                       │                          │
│  │  State: NOT_INITIALIZED → STARTING    │                          │
│  │         → RUNNING ↔ ERROR → STOPPING  │                          │
│  │         → STOPPED                     │                          │
│  └────────────────────┬──────────────────┘                          │
│                       │                                              │
│  ┌────────────────────┴──────────────────┐                          │
│  │     BACKEND  (runs in own thread)     │                          │
│  │                                       │                          │
│  │  ┌─────────────┐ ┌─────────────────┐ │                          │
│  │  │RobotBackend │ │ MujocoBackend   │ │                          │
│  │  │(real HW)    │ │ (physics sim)   │ │                          │
│  │  └──────┬──────┘ └────────┬────────┘ │                          │
│  │         │        ┌────────┴────────┐ │                          │
│  │         │        │MockupSimBackend │ │                          │
│  │         │        │(no physics)     │ │                          │
│  │         │        └─────────────────┘ │                          │
│  └─────────┼────────────────────────────┘                          │
│            │                                                         │
│  ┌─────────┴──────────┐  ┌────────────────────┐                    │
│  │  KINEMATICS ENGINE │  │  MEDIA MANAGER     │                    │
│  │  (pluggable)       │  │  (pluggable)       │                    │
│  │                    │  │                    │                    │
│  │  Analytical (Rust) │  │  OpenCV / SoundDev │                    │
│  │  Placo (Python)    │  │  GStreamer         │                    │
│  │  NN (ONNX)         │  │  WebRTC            │                    │
│  └────────────────────┘  └────────────────────┘                    │
└──────────────────────────────────────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │      HARDWARE / SIM         │
              │                              │
              │  9 Dynamixel motors          │
              │  Camera (IMX708)             │
              │  4-mic array (XMOS XVF3800) │
              │  Speaker (5W)                │
              │  IMU (BMI088, Wireless only) │
              └─────────────────────────────┘
```

---

## Core Subsystems and Responsibilities

### 1. SDK Client (`reachy_mini.py`)

**Responsibility:** User-facing Python API. Translates high-level intent into commands.

- Owns the `ReachyMini` context manager
- Sends commands via `ZenohClient` (pub/sub) or REST (HTTP)
- Receives state updates via Zenoh subscriptions
- Manages media backend selection and lifecycle
- Performs `look_at_image/world` coordinate transforms using camera calibration matrix
- Does NOT perform IK/FK -- that is delegated to the daemon

### 2. Zenoh Transport (`io/`)

**Responsibility:** Reliable pub/sub messaging between SDK and daemon.

- `ZenohClient`: Publishes commands to `{prefix}/command`, receives joint positions and pose
- `ZenohServer`: Subscribes to commands, publishes state at control loop frequency
- Supports two network modes:
  - **Localhost** (`tcp/localhost:7447`): Single machine, no discovery
  - **Network** (`tcp/0.0.0.0:7447`): LAN with multicast/gossip scouting
- Heartbeat monitoring: client checks liveness every 1 second

### 3. Daemon (`daemon/`)

**Responsibility:** Hardware abstraction, safety enforcement, app lifecycle.

- FastAPI server (port 8000) for REST/WebSocket API
- `Daemon` class manages backend lifecycle (start/stop/restart)
- `AppManager` runs user apps as subprocesses with process isolation
- Publishes daemon status every 1 second
- Mounts 10+ API routers for different concerns

### 4. Backend (`daemon/backend/`)

**Responsibility:** Motor control loop. The only code that talks to hardware.

- Abstract `Backend` base class defines the control interface
- Three implementations: `RobotBackend`, `MujocoBackend`, `MockupSimBackend`
- Runs the 50Hz control loop in a dedicated thread
- Owns all state: current positions, target positions, motor mode
- Manages move execution (goto_target, play_move) with RLock serialization
- Performs IK/FK computation using pluggable kinematics engine

### 5. Kinematics (`kinematics/`)

**Responsibility:** Forward and inverse kinematics for the 6-DOF Stewart platform head.

- Three pluggable engines with the same interface (IK + FK methods)
- `AnalyticalKinematics` (Rust FFI): Default, fastest (~1ms)
- `PlacoKinematics` (constraint solver): Most accurate, supports gravity compensation and collision detection (~50ms)
- `NNKinematics` (ONNX): Fast inference (~5ms)
- Safety limits enforced at the IK level (head pitch/roll, yaw delta)

### 6. Motion (`motion/`)

**Responsibility:** Time-parameterized trajectory generation.

- `Move` abstract base: `evaluate(t) -> (head_pose, antennas, body_yaw)`
- `GotoMove`: Interpolated motion with 4 methods (minjerk, linear, ease, cartoon)
- `RecordedMove`: Playback from HuggingFace datasets with binary search interpolation
- Moves are deterministic functions of time -- no state, no side effects

### 7. Media (`media/`)

**Responsibility:** Camera frames and audio I/O.

- `MediaManager` selects backend based on hardware variant and connection mode
- Camera: `get_frame()` returns BGR numpy array
- Audio: Ring-buffer recording and callback-based playback
- Direction of Arrival from XMOS mic array
- Four backends: OpenCV+SoundDevice (default), GStreamer, WebRTC, no_media

### 8. Apps (`apps/`)

**Responsibility:** Discoverable app ecosystem.

- `ReachyMiniApp` base class with lifecycle (run, stop_event)
- `AppManager` runs apps as isolated subprocesses
- Sources: HuggingFace Spaces, local installs, curated store
- App assistant CLI for scaffolding and publishing

---

## Data Ownership

| Data | Owner | Published via |
|------|-------|---------------|
| Current joint positions | Backend (read from hardware) | Zenoh `{prefix}/joint_positions` @ 50Hz |
| Current head pose | Backend (computed via FK) | Zenoh `{prefix}/head_pose` @ 50Hz |
| Target joint positions | Backend (computed via IK from target pose) | Internal only |
| Target head pose | SDK client or move executor | Zenoh `{prefix}/command` |
| Motor control mode | Backend | REST `/api/motors/status` |
| Daemon state | Daemon orchestrator | Zenoh `{prefix}/daemon_status` @ 1Hz |
| IMU data | Backend (Wireless only) | Zenoh `{prefix}/imu_data` @ 50Hz |
| Camera frames | MediaManager (client-side) | Not published (local to client) |
| Audio samples | MediaManager (client-side) | Not published (local to client) |
| Recorded motion data | Backend | Zenoh `{prefix}/recorded_data` (on stop) |

---

## Real-Time vs Non-Real-Time Paths

### Real-Time (50Hz control loop, deterministic timing)
- Backend `_update()` method: read joints → FK → IK → write targets → publish
- Must complete within 20ms per iteration
- Runs in dedicated daemon thread, no GIL contention from asyncio
- Uses `multiprocessing.Event.wait()` for cross-platform precise timing

### Soft Real-Time (100Hz motion playback)
- `play_move()` loop: evaluate trajectory → set targets → sleep
- Runs as asyncio task, subject to event loop scheduling
- Acceptable jitter: up to 5ms

### Non-Real-Time (human-scale latency acceptable)
- REST API requests (state queries, motor mode changes)
- WebSocket state streaming (configurable, default 10Hz)
- App lifecycle management (start/stop/install)
- Dashboard rendering
- Dataset downloading and caching

---

## Key Architectural Decisions

1. **Client-server split via Zenoh.** The SDK (client) never touches hardware directly. This allows running AI workloads on a powerful laptop while the daemon runs on a Raspberry Pi CM4.

2. **Backend runs in its own thread.** The control loop must not be blocked by FastAPI request handling, move execution, or app management.

3. **IK lives in the daemon, not the client.** The client sends desired poses; the daemon computes joint angles. This centralizes safety enforcement and allows the daemon to reject infeasible poses.

4. **Moves are pure functions of time.** `Move.evaluate(t)` takes a timestamp and returns a target. No internal state mutation. This makes moves composable and testable.

5. **Single move at a time.** An RLock prevents concurrent goto_target and play_move calls. `set_target` is blocked while a move is running. This prevents race conditions on target joint positions.

6. **Media is client-side.** Camera frames and audio are not routed through the daemon (except on Wireless via GStreamer/WebRTC). This avoids adding latency to the perception pipeline.
