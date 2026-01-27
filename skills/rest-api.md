# Skill: REST API

## When to Use

- Building web UIs or dashboards (JavaScript/TypeScript)
- Creating clients in non-Python languages
- Controlling robot from a different machine
- Building AI applications that connect via HTTP
- Creating MCP (Model Context Protocol) servers

---

## Overview

The Reachy Mini daemon exposes a **FastAPI-based REST API** with HTTP and WebSocket endpoints. This allows control from any language or platform.

**Base URL:** `http://localhost:8000/api` (when daemon is running)

**Interactive docs:** `http://localhost:8000/docs` (Swagger UI)

---

## Key Endpoints

### Movement Control

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/move/goto` | POST | Move to target (head pose, antennas, body yaw) |
| `/api/move/set_target` | POST | Set instant target (high-frequency control) |
| `/api/move/play/wake_up` | POST | Wake up the robot |
| `/api/move/play/goto_sleep` | POST | Put robot to sleep |
| `/api/move/play/recorded-move-dataset/{dataset}/{move}` | POST | Play recorded moves |
| `/api/move/running` | GET | List running move tasks |
| `/api/move/stop` | POST | Stop a running move |

### State Queries

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/state/full` | GET | Complete robot state |
| `/api/state/present_head_pose` | GET | Current head pose |
| `/api/state/present_body_yaw` | GET | Current body rotation |
| `/api/state/present_antenna_joint_positions` | GET | Antenna positions |
| `/api/state/doa` | GET | Direction of arrival (microphones) |

### Motor Control

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/motors/status` | GET | Motor status/control mode |
| `/api/motors/set_mode/{mode}` | POST | Change mode (enabled, disabled, gravity_compensation) |

### WebSocket Endpoints

| Endpoint | Description |
|----------|-------------|
| `ws://localhost:8000/api/state/ws/full` | Real-time state streaming |
| `ws://localhost:8000/api/move/ws/updates` | Movement event streaming |
| `ws://localhost:8000/api/move/ws/set_target` | Stream target commands |

---

## When to Use REST API vs Python SDK

**Use REST API when:**
- Building web frontends
- Using non-Python languages (JavaScript, Go, Rust, etc.)
- Running on a different machine from the robot
- Need language-agnostic access

**Use Python SDK when:**
- Building Python apps on the same machine
- Need direct kinematic calculations
- Working with media streams (camera, audio)
- Building high-frequency control loops (better latency)
- Working with recorded moves

---

## Example: JavaScript Fetch

```javascript
// Move head to position
const response = await fetch('http://localhost:8000/api/move/goto', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    head_pose: { yaw: 30, pitch: 10 },
    duration: 1.0
  })
});

// Get current state
const state = await fetch('http://localhost:8000/api/state/full')
  .then(r => r.json());
console.log(state);
```

---

## Example: WebSocket State Streaming

```javascript
const ws = new WebSocket('ws://localhost:8000/api/state/ws/full');

ws.onmessage = (event) => {
  const state = JSON.parse(event.data);
  console.log('Head pose:', state.head_pose);
};
```

---

## Source Code Reference

For implementation details (in this repository):
- **Main FastAPI app:** `src/reachy_mini/daemon/app/main.py`
- **Router modules:** `src/reachy_mini/daemon/app/routers/`
- **API docs:** `docs/source/troubleshooting.md` (REST API FAQ section)
