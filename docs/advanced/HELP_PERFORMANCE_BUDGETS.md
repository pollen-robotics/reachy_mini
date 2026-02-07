# Reachy Mini -- Performance Budgets

Stop slow creep. Make tradeoffs explicit instead of accidental.

---

## Control Loop Budget (20ms total)

The backend control loop runs at 50Hz. Every component in the loop has a time budget.

```
┌──────────────────────────────────────────────────┐
│               20ms CONTROL LOOP BUDGET            │
├──────────────┬───────┬───────────────────────────┤
│ Component    │ Budget│ Actual (measured)          │
├──────────────┼───────┼───────────────────────────┤
│ Read joints  │  5ms  │ ~2ms (serial I/O)         │
│ FK compute   │  3ms  │ ~1ms (analytical)         │
│ IK compute   │  5ms  │ ~1ms (analytical)         │
│              │       │ ~5ms (NN)                  │
│              │       │ ~50ms (Placo) ← OVERRUN   │
│ Write targets│  2ms  │ ~0.2ms (serial write)     │
│ Publish state│  2ms  │ ~0.5ms (Zenoh JSON)       │
│ Error checks │  1ms  │ ~0.1ms (every 1s only)    │
│ Margin       │  2ms  │ ~15ms (analytical default) │
├──────────────┼───────┼───────────────────────────┤
│ TOTAL        │ 20ms  │ ~5ms (analytical)         │
└──────────────┴───────┴───────────────────────────┘
```

**Overrun behavior:** If the loop exceeds 20ms, a debug warning is logged and the next iteration starts immediately (minimum 1ms sleep). Sustained overruns (>1 second without a response) trigger a fatal error.

**How to monitor:**
```python
# Via SDK
status = mini.client.get_status()
stats = status["backend_status"]["control_loop_stats"]
print(f"Mean frequency: {stats['mean_control_loop_frequency']:.1f} Hz")
print(f"Max interval: {stats['max_control_loop_interval']*1000:.1f} ms")

# Via REST
# GET /api/daemon/status → backend_status.control_loop_stats
```

**Target:** Mean frequency ~50Hz, max interval <25ms.

---

## IK Engine Latency Budget

| Engine | Latency | Fits in 20ms? | Notes |
|--------|---------|---------------|-------|
| AnalyticalKinematics (Rust) | ~1ms | Yes (15ms margin) | Default. Always safe. |
| NNKinematics (ONNX) | ~5ms | Yes (10ms margin) | Good alternative. |
| PlacoKinematics | ~50ms | **No** | Only recomputes when target changes. OK for goto_target. Risky for set_target at high frequency. |

**Rule:** If you are calling `set_target()` at >10Hz, use AnalyticalKinematics. PlacoKinematics is for goto_target (interpolated) use cases where IK is computed infrequently.

---

## Motion Playback Budget (10ms per tick)

The `play_move()` loop runs at 100Hz.

| Component | Budget | Typical |
|-----------|--------|---------|
| `move.evaluate(t)` | 5ms | <1ms (polynomial or lookup) |
| Set targets | 1ms | <0.1ms (attribute assignment) |
| asyncio.sleep overhead | 2ms | ~1ms |
| Margin | 2ms | ~8ms |

**If evaluate() exceeds 5ms:** Motion will visibly stutter. Pre-compute trajectories if possible.

---

## CPU Budget by Subsystem

### Raspberry Pi CM4 (Wireless -- 4-core ARM, 4GB RAM)

| Subsystem | CPU Budget | Rationale |
|-----------|-----------|-----------|
| Backend control loop | 1 core, <25% | Must be deterministic |
| Daemon (FastAPI + async) | 1 core | Request handling, app management |
| GStreamer media pipeline | 1 core | Audio/video encoding |
| User app | 1 core (remaining) | App code runs as subprocess |

**Total:** All 4 cores allocated. On Wireless, heavy AI workloads (LLMs, vision models) should run on the user's laptop, not on the CM4.

### Developer Laptop (Lite / Simulation)

CPU is typically not a constraint. The daemon uses <10% of a modern laptop CPU.

---

## Memory Budget

### Raspberry Pi CM4 (4GB total)

| Component | Budget | Notes |
|-----------|--------|-------|
| OS + system | 500MB | Linux base |
| Daemon | 300MB | Python runtime + libraries |
| GStreamer pipelines | 200MB | Audio/video buffers |
| App subprocess | 500MB | Depends on app |
| App venvs | 1.5GB | Per-app virtual environments |
| Available | 1GB | Headroom |

**Watch out for:**
- ONNX runtime loading (NNKinematics): ~200MB for model + runtime
- Large HuggingFace datasets: Downloaded to disk, loaded incrementally
- Audio buffers: SoundDevice ring buffer capped at 60 seconds (~2MB @ 16kHz stereo)

### Recording Memory

Motion recording stores one frame per control loop tick (50Hz):

```
Per frame: ~200 bytes (7 joints + 2 antennas + 4x4 pose + timestamp)
1 minute: ~600KB
10 minutes: ~6MB
```

This is bounded by the in-memory list. For very long recordings, consider streaming to disk.

---

## Network Bandwidth Budget

### Zenoh (Localhost)

| Channel | Direction | Size per message | Frequency | Bandwidth |
|---------|-----------|-----------------|-----------|-----------|
| joint_positions | Server→Client | ~200B JSON | 50Hz | ~10 KB/s |
| head_pose | Server→Client | ~300B JSON | 50Hz | ~15 KB/s |
| daemon_status | Server→Client | ~500B JSON | 1Hz | ~0.5 KB/s |
| commands | Client→Server | ~200B JSON | Varies | <10 KB/s |
| **Total** | | | | **~35 KB/s** |

Negligible on localhost. Over WiFi, adds <1ms latency typically.

### WebRTC (Wireless Remote)

| Stream | Direction | Bandwidth |
|--------|-----------|-----------|
| Video (H.264) | Robot→Laptop | 2-5 Mbps |
| Audio (Opus) | Robot→Laptop | 64 kbps |
| Audio (Opus) | Laptop→Robot | 64 kbps |
| **Total** | | **2-5 Mbps** |

Requires stable WiFi. Packet loss >5% will cause visible artifacts.

### MuJoCo Rendering (Simulation)

| Stream | Direction | Bandwidth |
|--------|-----------|-----------|
| JPEG frames (UDP) | Daemon→Client | ~2 Mbps @ 25Hz |

Localhost only. No network traversal.

---

## Thermal Budget

### Continuous Operation

| Condition | Duration | Risk |
|-----------|----------|------|
| Active motion (moderate) | Up to 2 hours | Low -- motors share load |
| High-torque hold (static) | Up to 30 minutes | Medium -- single motor may overheat |
| Gravity compensation | Up to 1 hour | Medium -- continuous current draw |
| Motors disabled (idle) | Unlimited | None -- no power draw |

**Thermal protection:** Dynamixel motors have built-in thermal shutdown. If triggered:
1. Motor stops responding
2. Hardware error flag set (Overheating)
3. Power off, wait 5+ minutes, power on

**Prevention:**
- Disable motors during idle periods
- Use gravity compensation instead of position hold for recording sessions
- Avoid sending targets far from current position repeatedly (causes high torque)

### Battery (Wireless)

| Activity | Approximate Runtime |
|----------|-------------------|
| Active (motors + compute + WiFi) | 1-2 hours |
| Idle (daemon running, motors disabled) | 3-4 hours |

No precise battery percentage available. Rely on LED indicator (green → orange → red).

---

## Performance Monitoring Checklist

To verify the system is within budget:

```python
with ReachyMini() as mini:
    status = mini.client.get_status()
    stats = status["backend_status"]["control_loop_stats"]

    freq = stats["mean_control_loop_frequency"]
    max_dt = stats["max_control_loop_interval"]
    errors = stats["nb_error"]

    assert freq > 45, f"Control loop too slow: {freq:.1f} Hz"
    assert max_dt < 0.025, f"Max interval too high: {max_dt*1000:.1f} ms"
    assert errors == 0, f"Communication errors: {errors}"
```

Run this periodically during development to catch performance regressions early.

---

## When Budget is Exceeded

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Control loop < 45Hz | Heavy IK (Placo) or CPU contention | Switch to AnalyticalKinematics |
| Max interval > 50ms | GC pause or blocking I/O in loop | Profile, reduce allocations |
| Motor errors > 0 | Serial timeout from overrun | Reduce control loop work |
| App sluggish on Wireless | CM4 out of CPU | Offload AI to laptop |
| WebRTC video stutters | WiFi bandwidth or latency | Move closer to router, reduce resolution |
| Memory >3.5GB on CM4 | App or model too large | Reduce model size, stream data |
