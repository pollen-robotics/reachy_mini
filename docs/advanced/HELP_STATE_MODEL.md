# Reachy Mini -- State Model

Make behavior predictable. Most robotics bugs are illegal state transitions, not math errors.

---

## State Machines Overview

Reachy Mini has three independent state machines that operate concurrently:

1. **Daemon State** -- Is the daemon running?
2. **Motor Control Mode** -- Are motors stiff, limp, or compliant?
3. **App State** -- Is a user app running?

---

## 1. Daemon State

```
                    start()
  NOT_INITIALIZED ──────────► STARTING
                                 │
                         ┌───────┴────────┐
                         │ backend.ready   │ backend fails
                         │ within 2s       │ or timeout
                         ▼                 ▼
                      RUNNING           ERROR
                         │                 │
                  stop() │                 │ restart()
                         ▼                 │
                      STOPPING ◄───────────┘
                         │
                 ┌───────┴────────┐
                 │ thread joins   │ timeout or
                 │ within 5s      │ exception
                 ▼                ▼
              STOPPED           ERROR
```

**Enum: `DaemonState`**
```
NOT_INITIALIZED  -- Created but not started
STARTING         -- Backend initializing, waiting for ready event
RUNNING          -- Control loop active, accepting commands
STOPPING         -- Shutdown in progress (optional goto_sleep)
STOPPED          -- Clean shutdown complete
ERROR            -- Unrecoverable failure
```

**Transition rules:**
- Only `start()` can move from NOT_INITIALIZED or STOPPED to STARTING
- Only the backend ready event (within 2s timeout) moves STARTING to RUNNING
- `stop()` moves RUNNING to STOPPING, then STOPPED (or ERROR on timeout)
- `restart()` calls `stop()` then `start()` sequentially
- Any unhandled exception in the backend thread transitions to ERROR

**Who can trigger transitions:**
- REST API: `/api/daemon/start`, `/api/daemon/stop`, `/api/daemon/restart`
- Daemon lifespan (FastAPI startup/shutdown)
- Backend thread crash (automatic ERROR transition)

---

## 2. Motor Control Mode

```
                 enable_motors()
  DISABLED ◄──────────────────────► ENABLED
     │                                  │
     │      disable_motors()            │
     │◄─────────────────────────────────┤
     │                                  │
     │      enable_gravity_             │
     │      compensation()              │
     └──────────────────► GRAVITY_COMPENSATION
                                  │
                   disable_       │
                   motors()       │
                   ◄──────────────┘
```

**Enum: `MotorControlMode`**
```
Enabled                -- Torque ON, position control (operating mode 3)
Disabled               -- Torque OFF, motors limp
GravityCompensation    -- Torque ON, current control (operating mode 0)
                          Requires Placo kinematics
```

**Transition details:**

| From | To | What happens internally |
|------|----|------------------------|
| Disabled → Enabled | `enable_motors()` | Set goal to current position, then enable torque |
| Enabled → Disabled | `disable_motors()` | Remove torque from all motors |
| Disabled → GravityComp | `enable_gravity_compensation()` | Switch to torque mode (op mode 0), enable torque, start gravity torque loop |
| GravityComp → Enabled | `enable_motors()` | Disable torque, switch to position mode (op mode 3), set goal to current, re-enable |
| GravityComp → Disabled | `disable_motors()` | Remove torque |
| Enabled → GravityComp | `enable_gravity_compensation()` | Disable torque, switch to torque mode, re-enable |

**Constraints:**
- GravityCompensation requires PlacoKinematics. Raises `RuntimeError` with AnalyticalKinematics or NNKinematics.
- Motor mode transitions are serialized (one at a time via backend method calls).

**Who can trigger transitions:**
- SDK: `mini.enable_motors()`, `mini.disable_motors()`, `mini.enable_gravity_compensation()`
- REST: `POST /api/motors/set_mode/{mode}`
- Zenoh: Command with `"torque": true/false` or `"gravity_compensation": true`
- WebRTC: Data channel command `set_motor_mode`

---

## 3. App State

```
                start_app()
  (no app) ──────────────────► STARTING
                                   │
                          ┌────────┴────────┐
                          │ app runs        │ app crashes
                          │ successfully     │ on startup
                          ▼                 ▼
                       RUNNING           ERROR
                          │
                 ┌────────┴────────┐
                 │ app finishes    │ stop_current_app()
                 │ normally        │ (SIGINT, 20s timeout)
                 ▼                 ▼
               DONE            STOPPING
                                   │
                          ┌────────┴────────┐
                          │ process exits   │ force kill
                          ▼                 ▼
                        DONE             DONE
```

**Enum: `AppState`**
```
STARTING   -- Subprocess launched, waiting for first output
RUNNING    -- App producing output, no errors detected
ERROR      -- App crashed or reported error
STOPPING   -- SIGINT sent, waiting for graceful exit
DONE       -- Process exited (success or after forced kill)
```

**Constraints:**
- Only one app can run at a time. `start_app()` fails if `is_app_running()` returns true.
- `set_target` commands from the SDK are blocked while a move is running (not while an app is running -- apps use the SDK internally).

**Post-app recovery:**
When an app stops, the daemon returns the robot to `INIT_HEAD_POSE` with antennas at [0, 0] over 1 second.

---

## 4. Move Execution State

This is not a formal state machine but a critical guard:

```
  IDLE (no move running)
     │
     │ goto_target() or play_move()
     │ acquires _play_move_lock
     ▼
  MOVE_RUNNING
     │
     │ - set_target commands are BLOCKED
     │ - new goto_target/play_move calls FAIL silently
     │ - control loop continues at 50Hz
     │
     │ move.evaluate(t) reaches duration
     │ releases _play_move_lock
     ▼
  IDLE
```

**Guard mechanism:** `_play_move_lock` (RLock) with `_active_move_depth` counter.

- `_try_start_move()`: Non-blocking acquire. Returns False if locked by another thread.
- `is_move_running`: Property checking `_active_move_depth > 0`.
- ZenohServer `_handle_command()`: Logs warning and drops command if `is_move_running`.

---

## What Happens on Failure

### Backend thread crashes
1. Exception stored in `backend.error`
2. `backend.close()` called for cleanup
3. Daemon state transitions to ERROR
4. Status published with error message
5. SDK `send_command()` raises `ConnectionError` (heartbeat fails within 1s)

### Motor communication lost
1. `RobotBackend._update()` detects no response for 1+ second
2. Sets error: `"No response from the robot's motor for the last second."`
3. Raises `RuntimeError`, causing backend thread to exit
4. Daemon transitions to ERROR

### IK computation fails
1. `update_target_head_joints_from_ik()` catches `ValueError`
2. Warning logged (throttled to 0.5s interval)
3. Previous valid target positions retained
4. Control loop continues -- does NOT crash

### App subprocess crashes
1. Monitor task detects non-zero exit code
2. Last 10 stderr lines captured
3. App state transitions to ERROR
4. Robot returned to init pose via recovery procedure
5. Daemon continues running normally

---

## State Queries

| What you want to know | How to check |
|------------------------|-------------|
| Is daemon running? | `GET /api/daemon/status` → `state == "running"` |
| Are motors enabled? | `GET /api/motors/status` → `mode` |
| Is a move executing? | `GET /api/move/running` → list of UUIDs |
| Is an app running? | `GET /api/apps/current-app-status` → `state` |
| Is the backend ready? | `GET /api/daemon/status` → `backend_status.ready` |
| Last control loop frequency? | `GET /api/daemon/status` → `backend_status.control_loop_stats` |
