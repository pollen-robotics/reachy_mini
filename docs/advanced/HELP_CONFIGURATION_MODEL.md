# Reachy Mini -- Configuration Model

Separate code from tuning. Know what is hardcoded, what is configurable, and when configs take effect.

---

## Configuration Layers

```
┌─────────────────────────────────────────────────┐
│  HARDCODED CONSTANTS  (code changes required)   │
│  Safety limits, coordinate frames, poses        │
├─────────────────────────────────────────────────┤
│  BOOT-TIME CONFIG  (daemon restart required)    │
│  Hardware YAML, kinematics engine, serial port  │
├─────────────────────────────────────────────────┤
│  RUNTIME CONFIG  (takes effect immediately)     │
│  Motor mode, targets, recording, app start/stop │
└─────────────────────────────────────────────────┘
```

---

## Hardcoded Constants (Code Changes Required)

### Safety Limits

**Location:** `kinematics/analytical_kinematics.py`, enforced in IK

| Parameter | Value | Source |
|-----------|-------|--------|
| Head pitch range | [-40, +40] degrees | Mechanical constraint |
| Head roll range | [-40, +40] degrees | Mechanical constraint |
| Head yaw range | [-180, +180] degrees | Full rotation |
| Body yaw range | [-160, +160] degrees | Mechanical constraint |
| Max yaw delta (head - body) | 65 degrees | Collision avoidance |

These are not configurable. Changing them risks physical damage to the robot.

### Predefined Poses

**Location:** `daemon/backend/abstract.py`

```python
INIT_HEAD_POSE = np.eye(4)                    # Neutral position (identity matrix)
SLEEP_HEAD_POSE = [specific 4x4 matrix]       # Head down resting position
SLEEP_HEAD_JOINT_POSITIONS = [0, -0.98, ...]  # Joint-space sleep pose
SLEEP_ANTENNAS_JOINT_POSITIONS = [-3.05, 3.05]# Antennas folded
```

### Camera Calibration

**Location:** `reachy_mini.py`

```python
T_head_cam = [
    Position: [0.0437, 0, 0.0512] meters
    Rotation: 90 deg roll, then 90 deg pitch
]
```

### Kinematics Tolerances

**Location:** `daemon/backend/abstract.py`

```python
_fk_kin_tolerance = 1e-3    # ~0.25 degrees for FK convergence
_ik_kin_tolerance = {
    "rad": 2e-3,             # ~0.1 degrees for IK
    "m": 0.5e-3,             # 0.5mm position tolerance
}
```

---

## Boot-Time Configuration (Daemon Restart Required)

### Hardware Config YAML

**Location:** `src/reachy_mini/assets/config/hardware_config.yaml`

**Format:**
```yaml
version: beta
serial:
  baudrate: 1000000

motors:
  body_rotation:
    id: 10
    offset: 0
    angle_limit:
      lower: 0
      upper: 4095
    return_delay_time: 0
    shutdown_error: 52
    operating_mode: 3
    pid:
      p: 200
      i: 0
      d: 0

  stewart_1:
    id: 11
    offset: 1024
    angle_limit:
      lower: 1502
      upper: 2958
    pid:
      p: 300
      i: 0
      d: 0
  # ... stewart_2 through stewart_6, right_antenna, left_antenna
```

**What you can tune:**
- **PID gains** (p, i, d): Reduce P to ~180 and increase D to ~10 to reduce motor jitter
- **Angle limits**: Hardware position bounds in servo ticks (0-4095)
- **Offset**: Encoder zero-point offset per motor

**What you should not change:**
- Motor IDs (10-18) -- must match physical wiring
- Baudrate -- must match firmware
- Operating mode -- position control (3) is the standard mode

**Custom config path:**
```bash
reachy-mini-daemon --hardware-config-filepath /path/to/my_config.yaml
```

### Daemon CLI Arguments

**Location:** `daemon/app/main.py` `Args` dataclass

| Argument | Default | Type | Notes |
|----------|---------|------|-------|
| `--sim` | False | flag | MuJoCo simulation mode |
| `--mockup-sim` | False | flag | Lightweight mock mode |
| `--scene` | "empty" | str | MuJoCo scene: "empty", "minimal" |
| `--headless` | False | flag | No GUI (sim only) |
| `--serialport` | "auto" | str | Serial port path or "auto" |
| `--kinematics-engine` | "AnalyticalKinematics" | str | "AnalyticalKinematics", "Placo", "NN" |
| `--check-collision` | False | flag | Enable collision detection (Placo only) |
| `--use-audio` | True | flag | Enable audio hardware |
| `--log-level` | "INFO" | str | Python logging level |
| `--fastapi-host` | "0.0.0.0" | str | Bind address |
| `--fastapi-port` | 8000 | int | HTTP port |
| `--robot-name` | "reachy_mini" | str | Zenoh topic prefix |
| `--wake-up-on-start` | True | flag | Play wake-up animation |
| `--goto-sleep-on-stop` | True | flag | Play sleep animation on shutdown |
| `--preload-datasets` | False | flag | Pre-download motion datasets |
| `--dataset-update-interval-hours` | 24.0 | float | Auto-update check interval |
| `--wireless-version` | False | flag | Enable wireless-specific features |
| `--localhost-only` | None | bool | Force Zenoh to localhost mode |

These are set once at daemon startup. Changing them requires a daemon restart.

### Zenoh Network Configuration

**Determined at startup based on `localhost_only` flag:**

| Mode | Client Config | Server Config |
|------|--------------|---------------|
| Localhost | `tcp/localhost:7447`, no scouting | `tcp/localhost:7447`, no multicast |
| Network | Peer mode, multicast + gossip | `tcp/0.0.0.0:7447`, multicast + gossip |

---

## Runtime Configuration (Immediate Effect)

### Motor Control Mode

Changed via REST API or SDK, takes effect on next control loop tick (20ms):

```python
# SDK
mini.enable_motors()
mini.disable_motors()
mini.enable_gravity_compensation()

# REST
POST /api/motors/set_mode/enabled
POST /api/motors/set_mode/disabled
POST /api/motors/set_mode/gravity_compensation
```

### Motion Targets

Changed on every call, applied at control loop frequency:

```python
mini.set_target(head=pose, antennas=[0.5, -0.5], body_yaw=0.3)
mini.goto_target(head=pose, duration=1.0, method="minjerk")
```

### Media Backend

Selected at `ReachyMini()` construction time:

```python
with ReachyMini(media_backend="default") as mini:     # OpenCV + SoundDevice
with ReachyMini(media_backend="gstreamer") as mini:    # GStreamer
with ReachyMini(media_backend="webrtc") as mini:       # WebRTC (remote Wireless)
with ReachyMini(media_backend="no_media") as mini:     # No camera/audio
```

Auto-detection happens if not specified: checks daemon for wireless flag, then checks local camera availability.

### Recording

Start/stop at any time:

```python
mini.start_recording()
# ... robot moves ...
data = mini.stop_recording()
```

### App Lifecycle

Start/stop apps without restarting daemon:

```
POST /api/apps/start-app/{app_name}
POST /api/apps/stop-current-app
```

---

## Safe Defaults

Every configuration has a safe default that works out of the box:

| Parameter | Default | Why safe |
|-----------|---------|----------|
| Kinematics | AnalyticalKinematics | Fastest, fits in timing budget |
| Collision check | False | Avoid false positives during development |
| Media backend | "default" (auto) | Works on all platforms |
| Wake up on start | True | Robot visually confirms it's ready |
| Sleep on stop | True | Robot goes to safe rest position |
| Serial port | "auto" | Discovers correct port automatically |
| Control loop | 50Hz | Proven frequency for Dynamixel motors |
| Motion playback | 100Hz | 2x oversampling for smooth motion |
| Audio sample rate | 16000 Hz | ReSpeaker native rate |

---

## Configuration File Locations

| File | Location | Purpose |
|------|----------|---------|
| Hardware config | `src/reachy_mini/assets/config/hardware_config.yaml` | Motor PID, limits, IDs |
| Kinematics data | `src/reachy_mini/assets/config/kinematics_data.json` | Stewart platform geometry |
| URDF models | `src/reachy_mini/descriptions/reachy_mini/urdf/` | Robot model for Placo |
| MuJoCo scenes | `src/reachy_mini/descriptions/reachy_mini/mjcf/scenes/` | Simulation environments |
| ONNX models | `src/reachy_mini/assets/models/` | Neural network kinematics |
| Audio assets | `src/reachy_mini/assets/sounds/` | Wake-up, sleep, dance sounds |
| Firmware | `src/reachy_mini/assets/firmware/` | Motor firmware binaries |
