# Reachy Mini -- Testing Strategy

Increase capability without fear. Know what to test and how.

---

## Test Pyramid

```
                    ┌─────────┐
                    │ Hardware │  Requires physical robot
                    │ In Loop  │  (marked tests, manual)
                   ─┤         ├─
                  / └─────────┘ \
                 /                \
                ┌──────────────────┐
                │   Integration    │  Daemon + backend + SDK
                │   (test_daemon)  │  MuJoCo sim, no hardware
               ─┤                 ├─
              / └──────────────────┘ \
             /                        \
            ┌──────────────────────────┐
            │         Unit Tests       │  Kinematics, collision,
            │  (test_analytical_kin,   │  imports, app lifecycle
            │   test_collision, etc.)  │
            └──────────────────────────┘
```

---

## Running Tests

### Standard CI command (no hardware required)

```bash
MUJOCO_GL=disable pytest -vv \
  -m 'not audio and not video and not audio_gstreamer and not video_gstreamer and not wireless and not wireless_gstreamer' \
  --tb=short
```

This runs all tests except those requiring physical hardware. Set `MUJOCO_GL=disable` when running without a display (CI, SSH, headless).

### Run a specific test file

```bash
pytest tests/test_analytical_kinematics.py -vv
```

### Run tests matching a pattern

```bash
pytest -k "daemon" -vv
```

---

## Test Markers

Tests that require specific hardware or system capabilities are marked. CI excludes all of these.

| Marker | What it requires | Example test |
|--------|-----------------|-------------|
| `audio` | Audio hardware (speaker + mic) | `test_audio.py` |
| `audio_gstreamer` | GStreamer audio pipeline | `test_audio.py` |
| `video` | Camera hardware | `test_video.py` |
| `video_gstreamer` | GStreamer video pipeline | `test_video.py` |
| `wireless` | Wireless Reachy Mini connected | `test_wireless.py` |
| `wireless_gstreamer` | GStreamer on Wireless version | `test_wireless.py` |

To run hardware tests (when connected to a robot):

```bash
pytest -m "audio" -vv          # Audio tests only
pytest -m "wireless" -vv       # Wireless tests only
```

---

## Test Categories

### Import Tests (`test_import.py`)

**What they verify:** Package structure, all public modules importable.

```python
def test_import():
    from reachy_mini import ReachyMini
    from reachy_mini.utils import create_head_pose
    # ... verifies core imports work
```

**When they fail:** Broken imports, missing dependencies, circular imports.

### Kinematics Tests (`test_analytical_kinematics.py`)

**What they verify:** IK-FK round-trip accuracy.

```python
def test_analytical_kinematics():
    # IK: identity pose → joint angles
    joints = kin.ik(np.eye(4))
    # FK: joint angles → recovered pose
    recovered = kin.fk(joints, no_iterations=10)
    # Must match within tolerance
    np.testing.assert_allclose(recovered, np.eye(4), atol=1e-2)

def test_analytical_kinematics_with_yaw():
    # Same but with body_yaw = π/4
    # Verifies head-body coupling
```

**Tolerance:** `atol=1e-2` (~0.57 degrees). This is generous to account for numerical precision.

**When they fail:** Kinematics engine bug, geometry data file corruption, tolerance regression.

### Collision Tests (`test_collision.py`)

**What they verify:** Placo kinematics rejects unreachable poses.

```python
def test_collision():
    # Reachable pose → IK returns solution
    assert kin.ik(create_head_pose()) is not None

    # Unreachable pose → IK returns None
    assert kin.ik(create_head_pose(x=20, y=20, mm=True)) is None
```

**Requires:** `placo_kinematics` extra installed.

### Daemon Tests (`test_daemon.py`)

**What they verify:** Full daemon lifecycle with MuJoCo simulation.

```python
async def test_daemon_start_stop():
    daemon = Daemon()
    await daemon.start(sim=True, headless=True, wake_up_on_start=False)
    # Daemon is RUNNING, backend is ready
    await daemon.stop(goto_sleep_on_stop=False)
    # Daemon is STOPPED

async def test_daemon_client_disconnection():
    # Start daemon, connect client, stop daemon
    # Verify client detects disconnection
```

**Pattern:** Uses `asyncio` with `pytest-asyncio`. Runs MuJoCo in headless mode.

### App Tests (`test_app.py`)

**What they verify:** App lifecycle (install, start, stop, remove).

```python
def test_app_manager():
    # Install ok_app → verify listed
    # Start ok_app → verify RUNNING
    # Wait for completion → verify DONE
    # Remove ok_app → verify gone

def test_faulty_app():
    # Install faulty_app → start → verify ERROR state
```

**Test fixtures:** `tests/ok_app/` (working app) and `tests/faulty_app/` (crashes on purpose).

---

## What to Test for New Features

### New kinematics engine
1. **IK-FK round-trip** for neutral pose, tilted pose, and extreme yaw
2. **Safety limits** enforced (pitch/roll clamped to [-40, +40])
3. **Yaw delta constraint** (max 65 degrees head-body difference)
4. **Unreachable pose** returns None or raises ValueError
5. **Timing** -- IK completes within 20ms budget

### New motion primitive
1. **Boundary values** -- `evaluate(0)` matches start, `evaluate(duration)` matches end
2. **Continuity** -- no jumps between consecutive evaluations
3. **Duration property** is finite and positive
4. **None handling** -- returns None for uncontrolled components

### New backend
1. **Start/stop lifecycle** -- clean startup, graceful shutdown
2. **State publishing** -- joint_positions and head_pose published at expected frequency
3. **Motor mode transitions** -- enable → disable → gravity compensation
4. **Move execution** -- goto_target and play_move work through the backend

### New API endpoint
1. **Happy path** -- returns expected response
2. **Invalid input** -- returns 400 with descriptive error
3. **Backend not running** -- returns 503 or appropriate error
4. **Concurrent access** -- no race conditions

---

## CI Pipeline

| Workflow | Trigger | What runs |
|----------|---------|-----------|
| `pytest.yml` | PR touching `src/`, `tests/`, `pyproject.toml` | Full test suite (Ubuntu + macOS, Python 3.10), excludes hardware markers |
| `lint.yml` | Push/PR touching `src/`, `tests/`, `pyproject.toml` | Ruff v0.12.0 + MyPy strict mode |

### Local pre-commit checks

```bash
# Install hooks (one-time)
pre-commit install

# Run manually
pre-commit run --all-files
```

This runs `ruff-check` and `ruff-format`. Code that fails pre-commit will fail CI.

---

## Testing Discipline

1. **Every IK/FK change needs a round-trip test.** If you modify kinematics math, prove it with a test.
2. **Every new motion type needs boundary tests.** Verify start, end, and mid-point behavior.
3. **New REST endpoints need at least a happy-path test.** Use FastAPI's `TestClient`.
4. **Run the full test suite before pushing.** Don't rely solely on CI.
5. **Mark hardware-dependent tests.** Use `@pytest.mark.audio`, `@pytest.mark.video`, etc.
6. **Tests must be deterministic.** No random seeds, no timing-dependent assertions (use generous timeouts).
