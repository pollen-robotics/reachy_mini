# Reachy Mini SDK — Architecture Overview

> Mental model for contributors (human or AI) before editing this repo. For app-builder context, see [`building_apps.md`](building_apps.md).

Reachy Mini is a small expressive desktop robot — a 6-DoF Stewart-platform head, a body that rotates around the vertical axis, and two antennas. This repository ships three things together:

1. A **Python SDK** (`src/reachy_mini/`) — the user-facing `ReachyMini` class plus all motion, media, kinematics, and IO primitives.
2. A **FastAPI daemon** (`src/reachy_mini/daemon/`) — a background service that mediates hardware (or simulator) and exposes REST + WebSocket at `:8000/api`.
3. A **browser JS SDK** (`js/reachy-mini.js`) — a single ES module that lets a Hugging Face Space drive the robot remotely over WebRTC.

The daemon is the only process that talks to the hardware. Both the local Python SDK (over WebSocket) and the browser JS SDK (over WebRTC, signaled through a central HF relay) connect to it.

---

## Top-level directory map

| Path | What it is |
|---|---|
| `src/reachy_mini/` | The Python package (everything published to PyPI). |
| `js/` | Browser SDK (`reachy-mini.js`) — single-file ES module. |
| `tests/` | `unit_tests/` (no hardware) + `integration_tests/` (markers gate hardware paths). |
| `examples/` | Runnable demo scripts (`minimal_demo.py`, `look_at.py`, `joy_controller.py`, `rerun_viewer.py`, …). |
| `docs/` | Documentation. `docs/source/` is the doc-builder source; `docs/contributing/` is this folder. |
| `docs/source/API/` | Generated API reference (per-module `*.mdx` + `openapi.json`). |
| `scripts/` | Maintenance scripts (`generate_openapi.py` and others). |
| `pyproject.toml` | Package metadata, console scripts, optional dependency groups. |
| `.pre-commit-config.yaml` | Ruff hooks (import order + docstrings + format). |

---

## Inside `src/reachy_mini/`

| Subpackage | Purpose | Central file(s) |
|---|---|---|
| (root) | `ReachyMini` — the user-facing class (context-manager, `goto_target`, `set_target`, sensor accessors). | `reachy_mini.py` |
| `daemon/` | FastAPI service + backends. Three backends select target: real hardware, MuJoCo, mockup. | `daemon/app/main.py`, `daemon/app/routers/`, `daemon/backend/{robot,mujoco,mockup}_backend.py` |
| `media/` | Audio/video. `AudioBase` and `CameraBase` abstractions, GStreamer pipelines, DoA helper. Three backends: `LOCAL` (IPC), `WEBRTC` (remote), `NO_MEDIA` (headless). | `media/audio_base.py`, `media/audio_control_utils.py`, `media/media_manager.py` |
| `motion/` | Trajectory generation + recorded-move playback. Interpolations: `linear`, `minjerk` (default), `ease_in_out`, `cartoon`. | `motion/recorded_move.py`, `motion/interpolation.py` |
| `io/` | WebSocket protocol between local Python SDK and daemon. | `io/protocol.py` |
| `apps/` | App scaffolding metadata (`ReachyMiniApp`, `AppInfo`, `SourceKind`); powers `reachy-mini-app-assistant`. | `apps/app.py` |
| `kinematics/` | Frames (head, body, world) and forward/inverse kinematics. | `kinematics/` |
| `descriptions/` | URDFs and hardware metadata. | `descriptions/` |
| `tools/` | Utilities (motor firmware reflashing). | `tools/reflash_motors.py` |
| `utils/` | Helpers (interpolation math, mDNS discovery, rerun visualization, constants). | `utils/` |

---

## Entry points

Defined in `pyproject.toml` under `[project.scripts]`:

| Console script | Maps to |
|---|---|
| `reachy-mini-daemon` | `src/reachy_mini/daemon/app/main.py` (starts the FastAPI service). |
| `reachy-mini-app-assistant` | `src/reachy_mini/apps/app.py` (CLI for scaffolding apps). |
| `reachy-mini-reflash-motors` | `src/reachy_mini/tools/reflash_motors.py` (firmware utility). |

Library entry: `from reachy_mini import ReachyMini`.

---

## Connection model — daemon ↔ SDK ↔ JS

The daemon is a single process, but it exposes **two parallel transports** that meet at the `Backend`. FastAPI is *not* on the WebRTC signaling path — that runs through GStreamer's `webrtcsink` and a separate central relay. Anything you want to expose to both transports has to be wired into both (a REST router and a WebRTC command type), each delegating to the same backend method.

```
                                  ┌───────────────── Daemon process ─────────────────┐
                                  │                                                  │
   Local Python SDK               │   FastAPI app          GstMediaServer            │
   ┌────────────────┐  REST/WS    │  ┌──────────────┐    ┌──────────────────────┐   │
   │ ReachyMini     │  :8000/api  │  │ routers/*    │    │ webrtcsink           │   │
   │ (io/protocol)  │ ◄─────────► │  │ /sdk_ws      │    │ (signaling :8443     │   │
   └────────────────┘             │  └──────┬───────┘    │  inside the daemon)  │   │
                                  │         │            └──────────┬───────────┘   │
                                  │         │                       │ ▲             │
                                  │         │                       │ │             │
                                  │         ▼                       ▼ │             │
                                  │  ┌──────────────────────────────┴─┴──────────┐  │
                                  │  │              Backend                      │  │
                                  │  │  RobotBackend / MujocoBackend / Mockup    │  │
                                  │  │  process_command(...)  + REST handlers    │  │
                                  │  └───────────────────────────────────────────┘  │
                                  │                                  ▲              │
                                  │                                  │ data channel │
                                  │   ┌──────────────────────────────┴───────────┐  │
                                  │   │  central_signaling_relay (in-process)    │  │
                                  │   │  bridges HF SSE  ◄──►  ws://127.0.0.1:8443│  │
                                  │   └──────────────────────────────┬───────────┘  │
                                  └──────────────────────────────────┼──────────────┘
                                                                     │
                                                                     │ SSE (signaling)
                                                                     ▼
                                                         ┌───────────────────────┐
                                                         │  Hugging Face         │
                                                         │  central signaling    │
                                                         │  Space (relay)        │
                                                         └───────────┬───────────┘
                                                                     │ SSE
                                                                     ▼
                                                         ┌───────────────────────┐
                                                         │  Browser (JS SDK,     │
                                                         │  HF Space)            │
                                                         │  WebRTC peer          │
                                                         └───────────────────────┘
```

**Two transports, one backend:**

- **REST / WebSocket on `:8000`** — handled by FastAPI in `src/reachy_mini/daemon/app/`. The local Python SDK uses this (`io/protocol.py`).
- **WebRTC on `:8443`** — handled by GStreamer's `webrtcsink` inside `GstMediaServer` (`src/reachy_mini/media/media_server.py`). The browser JS SDK uses this. Media tracks (camera, audio) and a data channel are negotiated automatically when a peer connects.

**Signaling is out-of-band.** The browser doesn't talk to FastAPI to negotiate SDP. Instead:

1. Browser → SSE to the central signaling Space on Hugging Face (`js/reachy-mini.js` — `connect()`).
2. The daemon runs a **central signaling relay** (`src/reachy_mini/media/central_signaling_relay.py`, started by `Daemon.start()`) that subscribes to that SSE stream and forwards messages to the local `webrtcsink`'s WebSocket on `127.0.0.1:8443`.
3. `webrtcsink` produces the SDP offer/answer and ICE candidates, which the relay sends back through HF to the browser.

Once WebRTC is up, the data channel becomes the command pipe. `media_server.py` handles the raw messages and forwards them to a callback registered by the backend via `setup_media_server()` (`src/reachy_mini/daemon/backend/abstract.py`). That callback parses each message with `command_adapter` and dispatches to `Backend.process_command()` — the same method the REST routers ultimately rely on.

**REST routers** live under `src/reachy_mini/daemon/app/routers/` (one file per concern: `media.py`, `volume.py`, `motion.py`, …). The OpenAPI spec at `docs/source/API/openapi.json` is generated from those routers via `scripts/generate_openapi.py` — keep it in sync (CI fails if it drifts).

**To expose a new feature on both transports:** add a REST endpoint *and* a command type in `command_adapter` + a branch in `process_command`. Both call the same backend method. Don't try to forward REST through WebRTC or vice versa.

### Discovering existing API surface

`docs/source/API/` is the canonical source of truth for what is already exported:

| File | Contents |
|---|---|
| `openapi.json` | Every daemon REST route + schema (also at `:8000/docs` when running). |
| `rest-api.mdx`, `daemon.mdx` | Curated REST/daemon overview. |
| `reachymini.mdx` | Public surface of the `ReachyMini` class. |
| `motion.mdx`, `media.mdx`, `apps.mdx`, `tools.mdx`, `utils.mdx` | Per-subpackage public APIs. |

Grep here before introducing a new helper or endpoint.

---

## Optional dependency groups

Defined in `pyproject.toml`. Install with `uv sync --group <name>` or `--all-extras`:

| Group | When you need it |
|---|---|
| `examples` | Running scripts under `examples/` (pynput, soundfile, opencv-python). |
| `mujoco` | Simulator backend. |
| `nn_kinematics`, `placo_kinematics` | Alternative IK solvers. |
| `rerun` | 3D visualization (`utils/rerun.py`). |
| `wireless-version` | Onboard CM4-only deps (IMU, GPIO, mDNS). |
| `opencv` | Camera utilities. |
| `dev` (under `[dependency-groups]`) | ruff, mypy, pytest, pre-commit. |

To match CI exactly: `uv sync --all-extras --group dev`.

---

## Test layout

- `tests/unit_tests/` — pure-Python tests, run on every PR.
- `tests/integration_tests/` — hardware/daemon paths, gated by pytest markers.

Markers (declared in `pyproject.toml`):

| Marker | Requires |
|---|---|
| `audio` | Reachy Mini Audio board over USB. |
| `video` | A camera. |
| `wireless` | A Wireless robot on the network. |
| `ipc_resolution` | Local IPC machinery. |

The CI no-hardware run is:

```bash
pytest -m "not audio and not video and not wireless and not ipc_resolution"
```

---

## Where to look for X

| You want to… | Look here |
|---|---|
| Add a REST endpoint | `src/reachy_mini/daemon/app/routers/` (then regenerate `openapi.json`). |
| Add a daemon command exposed over WebRTC | `src/reachy_mini/daemon/backend/abstract.py` — `command_adapter` + `process_command`. |
| Add an audio control parameter | `src/reachy_mini/media/audio_control_utils.py` — extend `PARAMETERS`. |
| Add a motion interpolation curve | `src/reachy_mini/motion/` (and the `interpolation` enum used by `goto_target`). |
| Change the JS API | `js/reachy-mini.js` (the file header, ~90 lines, doubles as reference docs). |
| Add a backend (sim, mock, alt-hardware) | `src/reachy_mini/daemon/backend/` — implement the `Backend` abstract class. |
| Wire a new media backend | `src/reachy_mini/media/` — implement `AudioBase` / `CameraBase`. |
| Generate or refresh API docs | `scripts/generate_openapi.py` and `doc-builder` (see `generate_docs.md`). |
| Find an existing helper before writing one | `docs/source/API/*.mdx`. |
