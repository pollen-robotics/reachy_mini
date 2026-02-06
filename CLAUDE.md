# Claude Code Instructions

Read `agents.md` in this directory for full instructions on developing Reachy Mini applications.

---

## Project Overview

**Reachy Mini** is a Python SDK and daemon for controlling the Reachy Mini robot, a small expressive robot with a 6-DOF head (Stewart platform), body rotation, and two antenna motors. The project is developed by [Pollen Robotics](https://github.com/pollen-robotics).

- **Package name:** `reachy_mini`
- **Version:** 1.3.0
- **License:** Apache 2.0
- **Python:** >= 3.10
- **Build system:** setuptools

---

## Repository Structure

```
reachy_mini/
├── src/reachy_mini/           # Main package source code
│   ├── reachy_mini.py         # Core SDK class (ReachyMini)
│   ├── apps/                  # App system (assistant, manager, templates)
│   ├── daemon/                # FastAPI daemon server
│   │   ├── app/               # Uvicorn entry point, API routers, dashboard
│   │   ├── backend/           # Robot, MuJoCo, Mockup backends
│   │   └── routers/           # REST API route handlers
│   ├── io/                    # Zenoh communication (client/server)
│   ├── kinematics/            # IK/FK: analytical, Placo, neural network
│   ├── motion/                # Motion primitives (Move, Goto, RecordedMove)
│   ├── media/                 # Audio/video: OpenCV, GStreamer, WebRTC
│   ├── tools/                 # Camera calibration, motor reflash utilities
│   ├── utils/                 # Hardware config, interpolation, wireless utils
│   ├── descriptions/          # URDF & MuJoCo robot models
│   └── assets/                # Firmware binaries, audio files, NN models
├── tests/                     # Pytest test suite
├── examples/                  # Runnable example scripts (23 files)
├── docs/                      # Full Markdown documentation
│   └── source/
│       ├── SDK/               # SDK guides (quickstart, python-sdk, core-concept)
│       ├── platforms/         # Platform-specific docs (Lite, Wireless, Simulation)
│       ├── examples/          # Example tutorials
│       └── API/               # Auto-generated API reference
├── skills/                    # AI agent skill guides (12 files)
├── agents.md                  # Main AI development guide
├── agents.local.md.template   # User config template
├── pyproject.toml             # Build config, dependencies, tool settings
├── conftest.py                # Pytest marker registration
├── MANIFEST.in                # Package data inclusion
├── uv.lock                    # Dependency lock file (uv)
└── .pre-commit-config.yaml    # Pre-commit hooks
```

---

## Development Setup

### Install dependencies

```bash
# Using uv (preferred):
uv pip install -e ".[dev]"

# Using pip:
pip install -e ".[dev]"
```

### Optional extras

| Extra | Purpose |
|-------|---------|
| `dev` | pytest, ruff, mypy, pre-commit |
| `mujoco` | MuJoCo physics simulation |
| `nn_kinematics` | ONNX neural network kinematics |
| `placo_kinematics` | Placo IK solver (Linux/macOS only) |
| `gstreamer` | GStreamer media (PyGObject) |
| `rerun` | Rerun 3D visualization |
| `wireless-version` | Wireless hardware support (Linux only) |
| `all` | Everything above |

### Entry points (console scripts)

| Command | Module |
|---------|--------|
| `reachy-mini-daemon` | `reachy_mini.daemon.app.main:main` |
| `reachy-mini-app-assistant` | `reachy_mini.apps.app:main` |
| `reachy-mini-reflash-motors` | `reachy_mini.tools.reflash_motors:main` |

---

## Code Quality

### Linting and formatting: Ruff (v0.12.0)

```bash
# Check linting
ruff check src/

# Auto-fix
ruff check --fix src/

# Format
ruff format src/
```

**Configuration (pyproject.toml):**
- Enabled rules: `I` (import sorting), `D` (docstrings)
- Ignored: `D203`, `D213` (conflicting docstring style rules)
- Excluded: `src/reachy_mini/__init__.py`, `build/`, `conftest.py`, `tests/`, `src/reachy_mini_dashboard/`

### Type checking: MyPy (v1.18.2)

```bash
mypy --install-types --non-interactive
```

- Target: Python 3.10
- **Strict mode enabled**
- Scans `src/` directory
- `ignore_missing_imports = true`

### Pre-commit hooks

```bash
pre-commit install
pre-commit run --all-files
```

Runs `ruff-check` and `ruff-format` via the `astral-sh/ruff-pre-commit` hooks.

---

## Testing

### Running tests

```bash
# Run all non-hardware tests (standard CI command):
pytest -vv -m 'not audio and not video and not audio_gstreamer and not video_gstreamer and not wireless and not wireless_gstreamer' --tb=short

# Set this env var when running without a display:
MUJOCO_GL=disable pytest ...
```

### Test markers

Tests that require physical hardware or specific system capabilities are marked:

| Marker | Requires |
|--------|----------|
| `audio` | Audio hardware |
| `audio_gstreamer` | GStreamer audio |
| `video` | Video/camera hardware |
| `video_gstreamer` | GStreamer video |
| `wireless` | Wireless Reachy Mini connected |
| `wireless_gstreamer` | GStreamer on Wireless |

### Test files

| File | Tests |
|------|-------|
| `test_import.py` | Basic package import checks |
| `test_app.py` | App system functionality |
| `test_daemon.py` | Daemon server behavior |
| `test_analytical_kinematics.py` | Analytical IK/FK |
| `test_placo.py` | Placo kinematics solver |
| `test_collision.py` | Collision detection |
| `test_video.py` | Video/camera (marked: video) |
| `test_audio.py` | Audio playback/recording (marked: audio) |
| `test_wireless.py` | Wireless-specific features |

Test fixtures include `ok_app/` and `faulty_app/` directories for app system testing.

---

## CI/CD (GitHub Actions)

| Workflow | Trigger | What it does |
|----------|---------|--------------|
| `pytest.yml` | PR on `src/`, `tests/`, `pyproject.toml` | Matrix tests: Ubuntu + macOS, Python 3.10, 10min timeout |
| `lint.yml` | Push/PR on `src/`, `tests/`, `pyproject.toml` | Ruff linting + MyPy type checking |
| `wheels.yml` | Release created | Build & publish to PyPI |
| `build_documentation.yml` | Documentation changes | Build docs |
| `build_pr_documentation.yml` | PR documentation changes | Build PR docs preview |
| `upload_pr_documentation.yml` | After PR docs build | Upload PR docs artifact |
| `uv-lock-check.yml` | Dependency changes | Validate uv.lock is up to date |

---

## Architecture

### Core SDK (`src/reachy_mini/reachy_mini.py`)

The `ReachyMini` class is the primary interface. Key methods:

- **Connection:** `ReachyMini()` with context manager support
- **Motion:** `goto_target()` (interpolated), `set_target()` (real-time)
- **State:** `wake_up()`, `goto_sleep()`, `get_current_joint_positions()`, `get_current_head_pose()`
- **Vision:** `look_at_image()`, `look_at_world()`
- **Motors:** `enable_motors()`, `disable_motors()`, `enable_gravity_compensation()`
- **Recording:** `start_recording()`, `stop_recording()`, `play_move()`, `async_play_move()`
- **Media:** `mini.media` property (audio/video access)

### Daemon (`src/reachy_mini/daemon/`)

FastAPI server exposing REST/WebSocket API at port 8000.

**Backends:**
- `RobotBackend` - Real hardware via serial
- `MujocoBackend` - Full physics simulation
- `MockupSimBackend` - Lightweight mock

**API routes:** `/api/apps/`, `/api/daemon/`, `/api/move/`, `/api/state/`, `/api/motors/`, `/api/kinematics/`, `/api/logs/`, `/api/cache/`, `/api/wifi_config/`, `/api/hf_auth/`, `/api/volume/`, `/api/update/`

### Communication

Uses **Zenoh** middleware for pub/sub messaging between SDK clients and the daemon, over localhost or network.

### Kinematics

Three solver implementations:
- `analytical_kinematics.py` - Fast analytical solution
- `placo_kinematics.py` - Constraint-based (Linux/macOS)
- `nn_kinematics.py` - Neural network via ONNX runtime

### Media

Pluggable backends for audio/video:
- **OpenCV** (Lite/default), **GStreamer** (Wireless), **WebRTC** (remote Wireless)
- Audio via `sounddevice` (Lite) or GStreamer (Wireless)

---

## Key Conventions

### Code style
- All source under `src/reachy_mini/` (src layout)
- Strict MyPy type checking - maintain full type annotations
- Ruff-formatted with import sorting
- Docstrings follow D211/D212 conventions

### Robot safety limits
- Head pitch/roll: [-40, +40] degrees
- Head yaw: [-180, +180] degrees
- Body yaw: [-160, +160] degrees
- Yaw delta (head - body): max 65 degrees
- SDK clamps values automatically

### Motor names
`body_rotation`, `stewart_1` through `stewart_6`, `right_antenna`, `left_antenna`

### Interpolation methods
`linear`, `minjerk` (default), `ease`, `cartoon`

### App development
- Always use Python for discoverable/shareable apps
- Use `reachy-mini-app-assistant create <name> <path>` to scaffold
- Web UIs go in a `static/` subdirectory
- Create `plan.md` before implementing any app

---

## Documentation

Full SDK documentation lives in `docs/source/`:

| Topic | Path |
|-------|------|
| Quickstart | `docs/source/SDK/quickstart.md` |
| Python SDK reference | `docs/source/SDK/python-sdk.md` |
| Core concepts | `docs/source/SDK/core-concept.md` |
| AI/LLM integration | `docs/source/SDK/integration.md` |
| Media architecture | `docs/source/SDK/media-architecture.md` |
| Troubleshooting | `docs/source/troubleshooting.md` |
| Platform guides | `docs/source/platforms/` |

---

## Skills Reference (for AI agents)

Detailed guides in `skills/` for specific topics:

| Skill | When to use |
|-------|-------------|
| `setup-environment.md` | First session setup |
| `create-app.md` | Creating new apps |
| `control-loops.md` | Real-time reactive apps |
| `motion-philosophy.md` | Choosing goto_target vs set_target |
| `safe-torque.md` | Motor enable/disable patterns |
| `ai-integration.md` | LLM-powered apps |
| `symbolic-motion.md` | Choreography / mathematical motion |
| `interaction-patterns.md` | Antennas as buttons, head as controller |
| `debugging.md` | Troubleshooting crashes and connectivity |
| `testing-apps.md` | Testing before delivery |
| `rest-api.md` | HTTP/WebSocket API usage |
| `deep-dive-docs.md` | When to read full documentation |
