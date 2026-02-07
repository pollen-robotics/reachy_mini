# Reachy Mini -- Timing and Real-Time Constraints

Protect determinism. Know what must be fast and what can be slow.

---

## Timing Budget Overview

```
                     HARD REAL-TIME
                     ┌──────────────────────────────┐
                     │  Backend Control Loop         │
                     │  Period: 20ms (50Hz)          │
                     │  Budget: <20ms per iteration  │
                     │                               │
                     │  Read joints:  ~2ms           │
                     │  FK compute:   ~1ms           │
                     │  IK compute:   ~1ms (analyt.) │
                     │  Write targets: ~0.2ms        │
                     │  Publish state: ~0.5ms        │
                     │  Margin:       ~15ms          │
                     └──────────────────────────────┘

                     SOFT REAL-TIME
                     ┌──────────────────────────────┐
                     │  Motion Playback              │
                     │  Period: 10ms (100Hz)         │
                     │  Runs as asyncio task         │
                     │  Jitter: up to ~5ms OK        │
                     │                               │
                     │  Evaluate trajectory: <1ms    │
                     │  Set targets: <0.1ms          │
                     │  asyncio.sleep: remainder     │
                     └──────────────────────────────┘

                     BEST EFFORT
                     ┌──────────────────────────────┐
                     │  REST API, Dashboard, Apps    │
                     │  WebSocket state streaming    │
                     │  App lifecycle management     │
                     │  Dataset downloading          │
                     │  Audio playback callbacks     │
                     └──────────────────────────────┘
```

---

## Control Loop (50Hz -- Hard Real-Time)

**Location:** `Backend.run()` in each backend subclass

**What happens every 20ms:**

| Step | Duration | Notes |
|------|----------|-------|
| Read joint positions from hardware | ~2ms | Serial communication |
| Update FK (kinematics model) | ~1ms | Analytical default |
| Compute IK if target changed | ~1ms | Only when `ik_required=True` |
| Write target positions to motors | ~0.2ms | Serial write |
| Publish joint_positions via Zenoh | ~0.5ms | JSON serialization + pub |
| Publish head_pose via Zenoh | ~0.5ms | Same |
| Hardware error check | ~0.1ms | Only every 1 second |
| **Total** | **~5ms** | **15ms margin** |

**Timing mechanism (RobotBackend):**
```python
period = 1.0 / 50.0  # 20ms

while not should_stop.is_set():
    start = time.time()
    _update()
    took = time.time() - start
    sleep_time = max(period - took, 0.001)
    next_call_event.clear()
    next_call_event.wait(sleep_time)  # Interruptible, cross-platform
```

If `_update()` takes longer than 20ms, a warning is logged and the loop runs as fast as possible (minimum 1ms sleep).

**What makes it deterministic:**
- Runs in a dedicated `threading.Thread` (daemon thread)
- No asyncio involvement -- pure threading + blocking I/O
- Motor controller uses `multiprocessing.Event.wait()` for precise timing
- No GC pressure: minimal allocations per iteration

---

## Motion Playback (100Hz -- Soft Real-Time)

**Location:** `Backend.play_move()`

```python
async def play_move(move, play_frequency=100.0):
    t0 = time.time()
    sleep_period = 1.0 / play_frequency  # 10ms

    while time.time() - t0 < move.duration:
        t = time.time() - t0
        head, antennas, body_yaw = move.evaluate(t)
        # Set targets (picked up by control loop at next 50Hz tick)
        elapsed = time.time() - t0 - t
        await asyncio.sleep(max(sleep_period - elapsed, 0.001))
```

**Why 100Hz, not 50Hz?**
The motion playback runs at 2x the control loop frequency to ensure no target update is missed. The control loop reads the latest target on each tick.

**What can cause jitter:**
- asyncio event loop congestion (other coroutines running)
- GIL contention from backend thread
- Heavy REST API requests during playback
- Acceptable: up to ~5ms jitter (smoothed by motor PID)

---

## IK Engine Latency Comparison

| Engine | IK Latency | FK Latency | Notes |
|--------|-----------|-----------|-------|
| AnalyticalKinematics (Rust) | ~1ms | ~1ms | Default. Fits easily in 20ms budget. |
| NNKinematics (ONNX) | ~5ms | ~5ms | Fits in budget with less margin. |
| PlacoKinematics (Python) | ~50ms | ~50ms | **Exceeds budget.** Used only with special care. |

When PlacoKinematics is selected, the control loop will overrun its 20ms budget on every IK computation. This is mitigated by only recomputing IK when `ik_required=True` (target changed).

---

## Publishing Frequencies

| Data | Frequency | Publisher | Consumer |
|------|-----------|-----------|----------|
| Joint positions | 50Hz | Backend → ZenohServer | SDK client, dashboard |
| Head pose | 50Hz | Backend → ZenohServer | SDK client, dashboard |
| IMU data | 50Hz | Backend → ZenohServer | SDK client (Wireless) |
| Daemon status | 1Hz | Daemon orchestrator | SDK client, dashboard |
| Task progress | On completion | ZenohServer | SDK client |
| Recorded data | On stop_recording | Backend | SDK client |
| WebSocket state | Configurable (default 10Hz) | State router | Dashboard, web clients |

---

## Threading Model

```
┌─────────────────────────────────────────────────────────┐
│  PROCESS: reachy-mini-daemon                            │
│                                                         │
│  Main Thread (asyncio event loop)                       │
│  ├── FastAPI request handlers                           │
│  ├── WebSocket connections                              │
│  ├── App subprocess monitoring (asyncio.Task)           │
│  ├── Motion playback (asyncio.Task via play_move)       │
│  ├── Dataset updater (asyncio.Task)                     │
│  └── Health check timeout (asyncio.Task)                │
│                                                         │
│  Backend Thread (daemon thread)                         │
│  └── Control loop @ 50Hz (_update method)               │
│      ├── Serial I/O to motor controller                 │
│      ├── FK/IK computation                              │
│      └── Zenoh publishing                               │
│                                                         │
│  Status Publisher Thread (daemon thread)                 │
│  └── Publishes daemon_status every 1s                   │
│                                                         │
│  MuJoCo Rendering Thread (daemon thread, sim only)      │
│  └── UDP frame streaming @ 25Hz                         │
│                                                         │
│  Zenoh Background Threads (managed by zenoh library)    │
│  └── Subscriber callbacks                               │
│                                                         │
│  APP SUBPROCESS (separate process, isolated)            │
│  └── User's ReachyMiniApp.run()                         │
│      └── User's control loop (set_target @ 50-100Hz)    │
└─────────────────────────────────────────────────────────┘
```

**Critical synchronization points:**

| Lock | Type | Protects | Contention |
|------|------|----------|------------|
| `_play_move_lock` | RLock | Move execution serialization | asyncio tasks vs Zenoh commands |
| `_rec_lock` | Lock | Recording buffer swap | Backend thread vs API calls |
| `ZenohServer._lock` | Lock | Command processing | Zenoh subscriber callback vs API |
| `busy_lock` (daemon router) | Lock | Daemon start/stop operations | Concurrent REST requests |

---

## User-Side Control Loop Guidelines

When building apps with `set_target()`:

| Frequency | Quality | CPU Impact |
|-----------|---------|------------|
| 100Hz (10ms) | Excellent. Matches motion playback. | Low |
| 50Hz (20ms) | Good. Matches control loop. | Very low |
| 30Hz (33ms) | Minimum acceptable. May appear slightly jerky. | Negligible |
| <30Hz | Visibly jerky. Not recommended. | - |
| >100Hz | Diminishing returns. Control loop is 50Hz anyway. | Wasteful |

**Recommended pattern:**
```python
import time

while not stop_event.is_set():
    target = compute_target()
    mini.set_target(head=target)
    time.sleep(0.01)  # 100Hz
```

Using `time.sleep()` is sufficient. The motor PID controller smooths any jitter between user updates and the 50Hz hardware loop.

---

## Where Latency Matters Most

1. **Control loop period (20ms):** Overruns degrade motion quality. Sustained overruns cause "No response from motor" errors.
2. **IK computation:** Must fit within the control loop budget. AnalyticalKinematics (~1ms) is the safe default.
3. **Zenoh message delivery:** Typically <1ms on localhost. Network mode adds ~1-5ms.
4. **Audio playback latency:** SoundDevice callback-based, ~100ms end-to-end. Not suitable for real-time audio effects.
5. **Camera frame latency:** OpenCV: ~33ms@30fps. GStreamer: lower. WebRTC: +50-100ms network latency.

---

## What Contributors Must Not Break

1. **Never add blocking I/O to the backend `_update()` method.** This is the 50Hz loop. File I/O, network requests, or heavy computation here will cause motor communication timeouts.

2. **Never call `set_target()` from the backend thread.** Targets are set by the asyncio event loop (via Zenoh commands or move playback). The backend thread reads them.

3. **Never use `time.sleep()` in asyncio coroutines.** Use `await asyncio.sleep()`. Blocking sleep in a coroutine blocks the entire event loop.

4. **Never allocate large objects in the control loop.** Numpy array creation, JSON parsing, and string formatting all trigger GC. Pre-allocate buffers.
