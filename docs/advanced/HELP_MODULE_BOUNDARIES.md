# Reachy Mini -- Module Boundaries

What each module is allowed to do. What it must never do. Where the lines are.

---

## Module Ownership Map

### `reachy_mini.py` (SDK Client)

**Owns:**
- User-facing Python API surface
- Connection mode selection (auto, localhost, network)
- Camera calibration transforms (`look_at_image`, `look_at_world`)
- Media backend lifecycle
- `create_head_pose()` convenience function

**Must never:**
- Compute IK/FK directly
- Write to motor hardware
- Enforce safety limits (that is the daemon's job)
- Manage daemon state transitions
- Run the control loop

**Public interface:** `ReachyMini` class with documented methods. Everything else is internal.

---

### `daemon/daemon.py` (Daemon Orchestrator)

**Owns:**
- Daemon lifecycle state machine (NOT_INITIALIZED → RUNNING → STOPPED)
- Backend creation and teardown
- ZenohServer initialization
- WebRTC setup (Wireless)
- Status publishing (1Hz)

**Must never:**
- Read/write motor hardware directly (that is Backend's job)
- Compute kinematics
- Process motion commands
- Manage app subprocesses (that is AppManager's job)

**Public interface:** `start()`, `stop()`, `restart()`, `get_status()`

---

### `daemon/backend/abstract.py` (Backend Base)

**Owns:**
- Current and target joint positions/poses (all state variables)
- Control loop execution (`run()`)
- IK/FK computation dispatch
- Move execution lock (`_play_move_lock`)
- Recording buffer management
- Motor control mode transitions
- Wake-up and sleep animations

**Must never:**
- Decide which backend to instantiate (Daemon does that)
- Manage network connections (ZenohServer does that)
- Handle HTTP/WebSocket requests (routers do that)
- Know about the app system

**Public interface:** Abstract methods (`run`, `get_status`, `get_present_*`, `set_motor_control_mode`, `set_motor_torque_ids`), plus shared concrete methods (`goto_target`, `play_move`, `set_target`, `start_recording`, `stop_recording`).

---

### `daemon/backend/robot/backend.py` (Robot Backend)

**Owns:**
- Serial communication with motor controller (via `ReachyMiniPyControlLoop`)
- Hardware error detection and reporting
- PID gain application
- Gravity compensation torque computation
- IMU reading (Wireless)
- Motor operation mode switching (position vs torque vs current-limiting)

**Must never:**
- Define the control interface (Backend base does that)
- Perform kinematics (pluggable engine does that)
- Handle network or API concerns

---

### `daemon/backend/mujoco/backend.py` (MuJoCo Backend)

**Owns:**
- MuJoCo physics simulation step
- Rendering and UDP frame streaming
- Scene management (empty, minimal)
- Joint-to-MuJoCo actuator mapping (including antenna sign flip)

**Must never:**
- Touch real hardware
- Manage network connections

---

### `kinematics/` (Kinematics Engines)

**Owns:**
- Forward kinematics: joints → 4x4 pose
- Inverse kinematics: 4x4 pose → joints
- Safety limit enforcement (max yaw delta, pitch/roll bounds)
- Collision detection (Placo only)
- Gravity torque computation (Placo only)

**Must never:**
- Read/write motor positions directly
- Know about Zenoh, REST, or any transport layer
- Manage timing or control loop frequency

**Public interface per engine:** `ik(pose, body_yaw) -> joints`, `fk(joints) -> pose`

---

### `motion/` (Motion Primitives)

**Owns:**
- Time-parameterized trajectory evaluation
- Interpolation methods (minjerk, linear, ease, cartoon)
- Recorded move parsing and binary-search playback
- HuggingFace dataset loading and caching

**Must never:**
- Call `set_target` or `goto_target` on the backend
- Know about hardware, network, or daemon state
- Maintain mutable state between evaluations

**Public interface:** `Move.evaluate(t) -> (head_pose, antennas, body_yaw)`, `Move.duration`

---

### `media/` (Media Manager)

**Owns:**
- Camera frame acquisition
- Audio recording and playback buffers
- Direction of arrival from mic array
- Backend selection logic (OpenCV, GStreamer, WebRTC)
- Device detection (auto-find "Reachy Mini Audio" or "reSpeaker")

**Must never:**
- Send motor commands
- Know about robot state or kinematics
- Participate in the control loop timing

**Public interface:** `get_frame()`, `start_recording()`, `get_audio_sample()`, `push_audio_sample()`, `get_DoA()`

---

### `io/` (Zenoh Transport)

**Owns:**
- Zenoh session and publisher/subscriber lifecycle
- Topic key expressions (`{prefix}/command`, `{prefix}/joint_positions`, etc.)
- Message serialization/deserialization (JSON)
- Connection liveness monitoring (heartbeat)

**Must never:**
- Interpret command semantics (ZenohServer just dispatches)
- Perform kinematics or motion computation
- Know about specific backend implementations

**Public interface:** `ZenohClient.send_command()`, `ZenohClient.send_task_request()`, `ZenohServer._handle_command()`, `ZenohServer._handle_task_request()`

---

### `apps/` (App System)

**Owns:**
- App discovery (HuggingFace, local, installed)
- App installation/removal
- Subprocess execution and monitoring
- App scaffolding (app assistant CLI)

**Must never:**
- Import or call backend code directly
- Access motor hardware
- Modify daemon state (except through the REST API)

**Public interface:** `AppManager.start_app()`, `stop_current_app()`, `install_new_app()`, `list_available_apps()`

---

### `daemon/routers/` (API Routers)

**Owns:**
- HTTP request validation (Pydantic models)
- Response serialization
- WebSocket connection lifecycle
- Background job tracking (UUIDs)

**Must never:**
- Access hardware directly
- Compute kinematics
- Own state -- all state lives in the backend or daemon

**Each router is a thin translation layer** between HTTP semantics and backend method calls.

---

## Boundary Violations to Watch For

| Violation | Why it's dangerous | What to do instead |
|-----------|-------------------|-------------------|
| SDK computing IK locally | Safety limits bypassed, duplicate logic | Send target pose to daemon, let daemon compute IK |
| Router modifying backend state directly | Bypasses thread synchronization | Call async backend methods through daemon |
| Motion calling set_target | Creates coupling between trajectory and transport | Motion returns targets; backend applies them |
| Backend knowing about FastAPI | Prevents backend reuse in other contexts | Backend exposes methods; routers call them |
| Kinematics engine reading motor positions | Breaks pluggability | Backend passes positions to kinematics |
| App importing backend modules | Breaks process isolation | App uses ReachyMini SDK or REST API |

---

## Stable vs Experimental APIs

### Stable (safe to depend on)
- `ReachyMini` class public methods
- `create_head_pose()` utility
- `Move` abstract interface
- `RecordedMoves` dataset loading
- REST API endpoints (`/api/move/`, `/api/state/`, `/api/motors/`)
- `ReachyMiniApp` base class

### Experimental (may change)
- Placo kinematics gravity compensation
- WebRTC media backend
- GStreamer pipeline configurations
- App assistant templates
- Dashboard WebSocket streaming format
- Raw packet interface (`/api/move/ws/raw/write`)
