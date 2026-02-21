# Reachy Mini -- Extension Points

Enable new features without refactoring core systems. Turn users into contributors.

---

## Where New Behaviors Plug In

### 1. New App (Most Common Extension)

The app system is the primary extension point. Apps are isolated subprocesses that use the SDK.

**How to add:**
```bash
reachy-mini-app-assistant create my_app ~/projects/my_app --publish
```

**What you get:**
- Scaffolded `main.py` with `ReachyMiniApp` subclass
- `pyproject.toml` with `reachy_mini_apps` entry point
- Optional `static/` directory for web UI
- Publishable to HuggingFace Spaces

**What not to modify:** The app system itself (`apps/manager.py`, `apps/sources/`). Apps interact exclusively through the SDK or REST API.

**Lifecycle your app must respect:**
```python
class MyApp(ReachyMiniApp):
    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            # Your logic here
            time.sleep(0.01)
```

The `stop_event` is set when the user stops your app from the dashboard. You must check it regularly and exit cleanly.

---

### 2. New Kinematics Engine

The kinematics system is pluggable via a common interface.

**Interface to implement:**
```python
class MyKinematics:
    def __init__(self, automatic_body_yaw: bool = True):
        ...

    def ik(
        self,
        pose: np.ndarray,       # 4x4 SE(3) matrix
        body_yaw: float = 0.0,
        check_collision: bool = False,
        no_iterations: int = 0,
    ) -> np.ndarray:            # [7]: [body_yaw, stewart_1-6]
        ...

    def fk(
        self,
        joint_angles: np.ndarray,  # [7]: [body_yaw, stewart_1-6]
        check_collision: bool = False,
        no_iterations: int = 3,
    ) -> np.ndarray:              # 4x4 SE(3) matrix
        ...
```

**Where to add it:**
1. Create `src/reachy_mini/kinematics/my_kinematics.py`
2. Add import in `src/reachy_mini/kinematics/__init__.py` with try/except fallback
3. Register the name in `daemon/app/main.py` `Args.kinematics_engine` choices
4. Add selection logic in `daemon/daemon.py` `_setup_backend()`

**What not to modify:** The backend control loop structure or the `ik()`/`fk()` interface signatures.

**Constraints your engine must satisfy:**
- IK must return 7-element array `[body_yaw, stewart_1, stewart_2, ..., stewart_6]`
- FK must return 4x4 homogeneous transformation matrix
- IK should return `None` or raise `ValueError` for unreachable poses
- Must enforce safety limits (head pitch/roll: [-40,+40]deg, yaw delta: max 65deg)
- IK latency must fit within 20ms control loop budget (or only recompute when target changes)

---

### 3. New Motion Primitive

Extend the motion system by subclassing `Move`.

**Interface to implement:**
```python
from reachy_mini.motion.move import Move

class MyMove(Move):
    @property
    def duration(self) -> float:
        return self._duration

    @property
    def sound_path(self) -> Path | None:
        return self._sound_path  # Optional synchronized audio

    def evaluate(self, t: float) -> tuple[
        np.ndarray | None,    # 4x4 head pose (None = don't change)
        np.ndarray | None,    # [2] antenna angles in radians
        float | None,         # body yaw in radians
    ]:
        # Your trajectory computation here
        # t is in [0, self.duration]
        ...
```

**How to use it:**
```python
move = MyMove(duration=2.0)
await backend.play_move(move, play_frequency=100.0)
```

**What not to modify:** The `Move` base class, the `play_move()` loop, or the interpolation utilities.

**Design constraints:**
- `evaluate()` must be a pure function of `t` -- no side effects, no state mutation
- Must complete in <1ms per call (called 100 times per second)
- Return `None` for any component you don't want to control

---

### 4. New Backend

For alternative simulation engines or hardware platforms.

**Interface to implement:** Subclass `Backend` (from `daemon/backend/abstract.py`) and implement:

```python
class MyBackend(Backend):
    def run(self) -> None:
        """Control loop. Must call self.ready.set() when initialized.
        Must check self.should_stop.is_set() to exit."""
        ...

    def get_status(self) -> MyBackendStatus:
        ...

    def get_present_head_joint_positions(self) -> np.ndarray:  # [7]
        ...

    def get_present_antenna_joint_positions(self) -> np.ndarray:  # [2]
        ...

    def get_motor_control_mode(self) -> MotorControlMode:
        ...

    def set_motor_control_mode(self, mode: MotorControlMode) -> None:
        ...

    def set_motor_torque_ids(self, ids: list[str], on: bool) -> None:
        ...
```

**Where to add it:**
1. Create `src/reachy_mini/daemon/backend/my_backend/backend.py`
2. Add selection logic in `daemon/daemon.py` `_setup_backend()`
3. Add CLI flag in `daemon/app/main.py` `Args` dataclass

**Critical contract:** Your `run()` method must:
- Run in a loop until `self.should_stop.is_set()`
- Call `self.ready.set()` after initialization (daemon waits 2s for this)
- Read targets set by the parent class (`self.target_head_pose`, etc.)
- Update current state (`self.current_head_pose`, etc.)
- Call `self.update_head_kinematics_model()` each iteration
- Publish state via the Zenoh publishers provided by `ZenohServer`

---

### 5. New Media Backend

For alternative camera/audio hardware.

**Camera interface:**
```python
from reachy_mini.media.camera_base import CameraBase

class MyCamera(CameraBase):
    def open(self) -> None: ...
    def read(self) -> np.ndarray | None: ...  # BGR uint8 frame
    def close(self) -> None: ...
```

**Audio interface:**
```python
from reachy_mini.media.audio_base import AudioBase

class MyAudio(AudioBase):
    def start_recording(self) -> None: ...
    def get_audio_sample(self) -> np.ndarray | None: ...  # float32
    def stop_recording(self) -> None: ...
    def start_playing(self) -> None: ...
    def push_audio_sample(self, data: np.ndarray) -> None: ...
    def stop_playing(self) -> None: ...
    def play_sound(self, sound_file: str) -> None: ...
```

**Where to add it:**
1. Create implementation files in `src/reachy_mini/media/`
2. Add a new `MediaBackend` enum value in `media/media_manager.py`
3. Add the selection case in `MediaManager.__init__()`

---

### 6. New REST API Endpoint

For exposing new daemon capabilities.

**How to add:**
1. Create a router file in `src/reachy_mini/daemon/app/routers/my_router.py`
2. Use FastAPI patterns:
   ```python
   from fastapi import APIRouter, Request
   router = APIRouter(prefix="/my_feature", tags=["my_feature"])

   @router.get("/status")
   async def get_status(request: Request):
       backend = request.app.state.daemon.backend
       return {"status": "ok"}
   ```
3. Mount in `daemon/app/main.py` `create_app()`:
   ```python
   app.include_router(my_router.router, prefix="/api")
   ```

**What not to modify:** Existing router files (add new ones instead).

---

## What is Stable vs Experimental

### Stable (safe to build extensions on)

| Component | Why stable |
|-----------|-----------|
| `ReachyMini` class public API | Versioned, documented, widely used |
| `Move` abstract class | Simple contract, unlikely to change |
| `Backend` abstract methods | Core architecture, three implementations depend on it |
| `CameraBase` / `AudioBase` interfaces | Clean abstractions with multiple implementations |
| REST API endpoints | Used by dashboard, web clients, external tools |
| Zenoh topic structure | SDK client depends on exact topics |
| Hardware config YAML format | Motor configuration must be backward compatible |

### Experimental (may change between versions)

| Component | Why experimental |
|-----------|----------------|
| Placo gravity compensation | Empirical torque constants, may need recalibration |
| WebRTC media backend | Platform support limited (Linux client only) |
| GStreamer pipeline details | Hardware-dependent configuration |
| App assistant templates | Evolving based on user feedback |
| Dashboard WebSocket message format | Not formally versioned |
| Raw packet interface | Direct hardware access, use at own risk |
| MuJoCo scene files | Scene format may change with MuJoCo updates |

---

## Extension Checklist

Before submitting an extension:

- [ ] Implements the documented interface (no monkey-patching)
- [ ] Does not modify existing module files (add new files instead)
- [ ] Respects timing constraints (see HELP_TIMING_AND_REALTIME.md)
- [ ] Has unit tests (see HELP_TESTING_STRATEGY.md)
- [ ] Does not break existing tests: `pytest -vv -m 'not audio and not video...'`
- [ ] Passes linting: `ruff check src/` and `ruff format src/`
- [ ] Passes type checking: `mypy --install-types --non-interactive`
- [ ] Documented in this file (update extension point section)
