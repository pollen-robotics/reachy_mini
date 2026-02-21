# Reachy Mini -- Error Handling

Make failures graceful and recoverable. A system that fails well feels robust.

---

## Error Categories

### 1. Hardware Errors (Motor Controller)

**Source:** `RobotBackend.read_hardware_errors()`, checked every 1 second.

| Error Flag | Meaning | Severity |
|-----------|---------|----------|
| Input Voltage Error | Voltage outside motor spec range | **Ignore** -- intentional (robot runs at upper range) |
| Overheating Error | Motor thermal shutdown | **Recover** -- power off, wait 5min, power on |
| Electrical Shock Error | Short circuit or power supply issue | **Investigate** -- check cables |
| Overload Error | Motor physically stuck | **Investigate** -- check orientation marks, cables |

**Handling pattern:**
```python
# In RobotBackend._update()
hardware_errors = self.read_hardware_errors()
for motor_name, errors in hardware_errors.items():
    # Voltage errors are filtered out (expected)
    logger.error(f"Motor '{motor_name}' hardware errors: {errors}")
```

Hardware errors are logged but do not stop the control loop. The robot continues operating with degraded motors.

### 2. Communication Errors (Serial / Zenoh)

**Motor communication lost:**
```python
# In RobotBackend._update()
if last_alive + 1.0 < time.time():
    self.error = "No response from the robot's motor for the last second."
    raise RuntimeError(self.error)
    # → Backend thread exits → Daemon state → ERROR
```

This is the most critical error. If the motor controller does not respond for 1 second, the backend assumes the serial connection is broken and shuts down.

**Zenoh connection lost (client-side):**
```python
# In ZenohClient.check_alive() (background thread, 1Hz)
_is_alive = is_connected()  # Waits 1s for joint_positions message
# If False, next send_command() raises ConnectionError
```

### 3. Kinematics Errors (IK Failures)

**Unreachable pose:**
```python
# In Backend.update_target_head_joints_from_ik()
try:
    joints = self.head_kinematics.ik(pose, body_yaw=body_yaw)
    if joints is None or np.any(np.isnan(joints)):
        raise ValueError("Collision detected or head pose not achievable!")
except ValueError as e:
    log_throttling.by_time(logger, interval=0.5).warning(f"IK error: {e}")
    # Previous valid target retained -- robot holds last good position
```

IK failures are **not fatal**. The control loop continues with the last valid target. This happens when:
- User requests a pose outside the workspace
- Collision detection rejects the pose (Placo only)
- Numerical issues in the solver

### 4. Application Errors (App Crashes)

**Detection:** `AppManager.monitor_process()` watches stdout/stderr.

```python
# App exits with non-zero code
if process.returncode != 0:
    last_stderr = captured_stderr[-10:]  # Last 10 lines
    status.state = AppState.ERROR
    status.error = "\n".join(last_stderr)
```

**Recovery:** Post-app cleanup returns robot to init pose:
```python
await backend.goto_target(INIT_HEAD_POSE, antennas=[0.0, 0.0], duration=1.0)
```

### 5. Task Errors (Async Motion)

**Detection:** asyncio task wrapper catches exceptions:
```python
async def wrapped_task():
    error = None
    try:
        await task_coroutine()
    except Exception as e:
        error = str(e)
    progress = TaskProgress(uuid=task_uuid, finished=True, error=error)
    task_progress_pub.put(progress.model_dump_json())
```

**Client-side propagation:**
```python
def wait_for_task_completion(uid, timeout=5.0):
    if not tasks[uid].event.wait(timeout):
        raise TimeoutError("Task did not complete in time.")
    if tasks[uid].error is not None:
        raise Exception(f"Task failed: {tasks[uid].error}")
```

---

## Retry vs Fail-Fast Rules

| Error Type | Strategy | Rationale |
|-----------|----------|-----------|
| IK failure | **Retain previous target** | Transient -- next target may be reachable |
| Motor communication timeout (< 1s) | **Retry next loop** | Brief glitches happen on serial bus |
| Motor communication timeout (> 1s) | **Fail-fast: shutdown** | Sustained loss means hardware disconnected |
| Zenoh connection lost | **Fail-fast: raise ConnectionError** | Client should reconnect or alert user |
| App crash | **Fail-fast: mark ERROR** | Do not auto-restart (may crash again) |
| goto_target move lock busy | **Fail silently** | Another move is running; caller can retry |
| set_target while move running | **Drop command** | Move has priority; command would conflict |
| Dataset download failure | **Return empty, log warning** | Non-critical; app can work without new data |

---

## Escalation Paths

```
Level 1: Log warning (no user action needed)
├── IK failure (throttled to every 0.5s)
├── Control loop overrun (single occurrence)
├── Hardware error: Input Voltage Error
└── Circular buffer overrun (no video consumer)

Level 2: Log error (user should investigate)
├── Hardware error: Overload, Overheating, Electrical Shock
├── Motor jitter (PID tuning needed)
├── App process crashed
└── Control loop sustained overruns

Level 3: Daemon stops (user must restart)
├── Motor communication lost > 1 second
├── Backend initialization failure (wrong serial port, etc.)
├── Backend thread unhandled exception
└── Daemon stop timeout (5 seconds)

Level 4: User must intervene physically
├── Motor thermal shutdown (wait 5 minutes)
├── Cable damage (inspect and replace)
├── Power supply failure (check connections)
└── Board switch in wrong position (Wireless)
```

---

## User-Visible vs Internal Errors

### User-Visible (Surfaced to SDK / REST API)

| Error | How surfaced | Recovery |
|-------|-------------|----------|
| Connection lost | `ConnectionError` from SDK | Recreate `ReachyMini()` context |
| Motor timeout | `GET /api/daemon/status` → `error` field | Restart daemon |
| Task failure | `wait_for_task_completion()` raises | Check error message, fix command |
| App crash | `GET /api/apps/current-app-status` → `error` | Fix app code, restart |
| Invalid input | `ValueError` from `goto_target` / `set_target` | Fix input format |

### Internal (Logged but not surfaced)

| Error | Where logged | Why not surfaced |
|-------|-------------|-----------------|
| IK failure | Daemon log (warning) | Transient, self-healing |
| Control loop overrun | Daemon log (debug) | Brief jitters are normal |
| Hardware voltage error | Daemon log (filtered out) | Expected by design |
| Recording buffer overflow | Daemon log (warning) | Self-correcting (drops old data) |

---

## Error Handling Patterns in Code

### Pattern 1: Graceful degradation (IK)
```python
try:
    result = compute()
except ValueError:
    log_throttling.warning("...")  # Don't spam logs
    # Keep previous state -- system continues
```

### Pattern 2: Fatal shutdown (motor loss)
```python
if not_responding_for_too_long:
    self.error = "descriptive message"
    raise RuntimeError(self.error)  # Backend thread exits
    # Daemon catches and transitions to ERROR state
```

### Pattern 3: Subprocess isolation (apps)
```python
process = await asyncio.create_subprocess_exec(...)
returncode = await process.wait()
if returncode != 0:
    status.state = AppState.ERROR
    status.error = last_stderr_lines
# Daemon is NOT affected -- continues running
```

### Pattern 4: Lock-based conflict avoidance (moves)
```python
if not self._try_start_move():
    return  # Another move running -- silently skip
try:
    await execute_move()
finally:
    self._end_move()  # Always release lock
```

### Pattern 5: Timeout with escalation (daemon lifecycle)
```python
await daemon.stop()
backend_thread.join(timeout=5.0)
if backend_thread.is_alive():
    state = DaemonState.ERROR  # Could not stop cleanly
else:
    state = DaemonState.STOPPED
```

---

## Writing Error-Resilient Extensions

When adding new code to Reachy Mini:

1. **Never swallow exceptions silently.** At minimum, log them.
2. **Never let exceptions escape the control loop.** Catch at the boundary and log.
3. **Use `log_throttling`** for high-frequency error paths (avoids log flood).
4. **Always release locks in `finally` blocks.**
5. **Set `self.error` before raising** in backend code, so the daemon can report it.
6. **Prefer returning `None` over raising** for query methods that may fail.
7. **Use timeouts on all blocking operations** (joins, event waits, network calls).
