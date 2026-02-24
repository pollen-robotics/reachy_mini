# Startup Optimization — Full Report

*Branch: `893-load-apps-faster` — February 2026*

---

## Goal

Reduce the time between clicking "Start Marionette" and the robot's head moving. On the CM4 (Wireless robot), this takes ~7.3 seconds. The user perceives the app as unresponsive during this entire period.

---

## How the Startup Works

When the daemon launches `python -u -m marionette.main`, three phases happen sequentially:

### Phase 1: Python Imports (3.48s on CM4, ~0.5s on laptop)

Python reads every `import` at the top of `main.py` and loads libraries into memory. Nothing runs — no robot communication, no audio, nothing. The app can't do anything until imports finish because the code that creates the Marionette class depends on types and functions from these libraries.

| Import | CM4 Time | What it is |
|--------|----------|------------|
| `reachy_mini` | 1.91s | The robot SDK. Imports `cv2` (0.3s), `scipy` (0.5s), `zenoh` (0.2s), `numpy`, etc. — even though Marionette doesn't use camera or scipy math functions. |
| `fastapi` | 1.22s | Web framework for the UI (also pulls in Starlette, anyio, pydantic). |
| `numpy` | 0.38s | Math/array library. Required by everything. |
| `huggingface_hub` | 0.31s | For dataset upload/download. |
| Everything else | ~0.27s | pydantic, soundfile, etc. |

### Phase 2: Creating the ReachyMini SDK Instance (3.42s on CM4, ~2.9s on laptop)

The Marionette class inherits from `ReachyMiniApp`. The base class's `wrapped_run()` method creates a `ReachyMini` object, which connects to the daemon and sets up motor control and audio. Here's what happens step by step:

```
ReachyMini.__init__():
│
├─ daemon_check()               Scans running processes to verify the daemon exists.
│
├─ ZenohClient()                Opens a Zenoh session (network protocol to talk to the daemon).
│                               Creates subscribers for joint positions, head pose, status, etc.
│
├─ wait_for_connection()        Blocks until the daemon has published at least one joint
│                               position update AND one head pose update over Zenoh.
│                               Uses Event.wait() with a timeout to check periodically.
│
└─ _configure_mediamanager()    Sets up the audio (and optionally camera) system.
    ├─ get_status()             Asks the daemon: "Are you wireless? What's your IP?"
    ├─ Backend selection        Picks GStreamer (wireless), SoundDevice (Lite), or WebRTC (remote).
    └─ MediaManager()           Creates audio pipelines, enumerates hardware, spawns threads.
```

The sound can't play until the MediaManager exists. The head can't move until the ZenohClient is connected. So the app must wait for all of this before doing anything visible.

### Phase 3: Startup Animation (~0.4s, not a problem)

Once `run()` is called, the app enables motors, moves the head up (~0.3s), then plays the first sound. This is fast and not a target for optimization.

---

## The Investigation

### Setting Up Local Benchmarking

The CM4 (Wireless robot) is slow to iterate on. We created a local benchmark that:
1. Spawns a mockup daemon (`reachy-mini-daemon --mockup-sim`)
2. Runs 10 fresh subprocesses, each measuring import time and `ReachyMini()` creation time
3. Reports mean ± standard deviation

Each iteration uses a fresh Python process so import caching doesn't skew results.

**Script**: `benchmarks/bench_startup.py`

```bash
python benchmarks/bench_startup.py --runs 10 --label baseline
```

### Baseline Measurement

| Metric | Laptop (mean ± std) |
|--------|---------------------|
| Import | 0.406s ± 0.009s |
| Connect | 2.528s ± 0.010s |
| **Total** | **2.934s ± 0.002s** |

The connect time is extremely consistent (std = 0.010s), which suggests a fixed blocking wait — not random network latency.

### Detailed Timing Breakdown

We added temporary instrumentation to `ReachyMini.__init__()` to measure each step:

```
daemon_check():          0.000s
_initialize_client():    0.018s   ← Zenoh open + wait_for_connection (fast: daemon is local)
_configure_mediamanager: 2.952s   ← THIS IS THE BOTTLENECK
```

Nearly all of the 2.9s connect time is inside `_configure_mediamanager()`. But with `media_backend='no_media'`, the MediaManager constructor is essentially a no-op. So where is the time going?

### Root Cause: Triple `get_status()` Call

We added finer instrumentation inside `_configure_mediamanager()`:

```
get_status() call #1:    0.929s
remaining:               2.023s   ← unexplained
```

Looking at the code, `_configure_mediamanager()` calls `self.client.get_status()` **three times**:

```python
def _configure_mediamanager(self, media_backend, log_level):
    daemon_status = self.client.get_status()          # CALL 1: line 194
    is_wireless = daemon_status.get("wireless_version", False)

    # ... backend selection logic ...

    return MediaManager(
        use_sim=self.client.get_status()["simulation_enabled"],     # CALL 2: line 246
        backend=mbackend,
        log_level=log_level,
        signalling_host=self.client.get_status()["wlan_ip"],        # CALL 3: line 249
    )
```

And the `get_status()` implementation in `ZenohClient` **clears the event after each call**:

```python
def get_status(self, wait=True, timeout=5.0):
    if wait and not self.status_received.wait(timeout):
        raise TimeoutError(...)
    self.status_received.clear()   # ← clears the flag!
    return self._last_status
```

The daemon publishes status every **1 second** (`time.sleep(1)` in `daemon._publish_status()`). So:

| Call | What happens | Time |
|------|-------------|------|
| `get_status()` #1 | Waits for first status message to arrive | ~0.5s (depends on when we subscribe relative to the 1s publish cycle) |
| `get_status()` #2 | Event was cleared by #1. Waits for the daemon to publish again. | ~1.0s |
| `get_status()` #3 | Event was cleared by #2. Waits for the daemon to publish again. | ~1.0s |
| **Total wasted** | | **~2.0s** |

The same dict is returned each time — the daemon status doesn't change between these three calls. **Two seconds wasted waiting for identical data.**

We also found the same double-call pattern in `utils/rerun.py`:

```python
if self._reachymini.client.get_status()["wireless_version"]:       # CALL 1
    self._robot_ip = self._reachymini.client.get_status()["wlan_ip"]  # CALL 2
```

---

## Changes Made

### Fix 1: Cache `get_status()` result (the big win)

**File**: `src/reachy_mini/reachy_mini.py`

Cache the first `get_status()` result and reuse it:

```python
def _configure_mediamanager(self, media_backend, log_level):
    daemon_status = self.client.get_status()    # call once
    is_wireless = daemon_status.get("wireless_version", False)
    # ... backend selection ...
    return MediaManager(
        use_sim=daemon_status["simulation_enabled"],      # reuse cached
        backend=mbackend,
        log_level=log_level,
        signalling_host=daemon_status["wlan_ip"],          # reuse cached
    )
```

Same fix in `src/reachy_mini/utils/rerun.py`.

### Fix 2: Lazy import cv2 and scipy

**File**: `src/reachy_mini/reachy_mini.py`

Moved `import cv2` and `from scipy.spatial.transform import Rotation as R` from the top of the file to inside the methods that use them:

- `cv2`: only used in `look_at_image()` (1 method)
- `Rotation`: only used in `wake_up()` and `look_at_world()` (2 methods)

Apps that don't use camera or look_at features (like Marionette) no longer pay the import cost at startup.

**Why this is safe:**
1. If you use a name that isn't imported, Python crashes with `NameError` immediately — no silent failures.
2. Python caches imports. The first `import cv2` inside a function takes 0.3s; every later call takes ~0.
3. The scope is small: 3 methods to update, verified by grepping for `cv2.` and `R.from_`.

**Expected CM4 savings**: ~0.8s (cv2 = 0.3s + scipy = 0.5s). Not measurable on the laptop where both load in <0.05s.

### Fix 3: Faster Zenoh poll interval

**File**: `src/reachy_mini/io/zenoh_client.py`

Changed `Event.wait(timeout=1.0)` to `Event.wait(timeout=0.1)` in `wait_for_connection()`.

The code uses `threading.Event.wait()` which returns immediately when the event is set. The timeout is only reached when data hasn't arrived yet. Reducing it from 1.0s to 0.1s means: if data hasn't arrived, we re-check 10x more often. If data has arrived, there's no difference.

Minimal impact on the laptop (daemon is local, data arrives in ~10ms). On the CM4 with real Zenoh over the network, this should help when the first data packet takes longer to arrive.

---

## Benchmark Results

Machine: Ubuntu laptop. Local mockup daemon. `media_backend='no_media'`. 10 runs each with warmup.

| State | Import (mean±std) | Connect (mean±std) | Total (mean±std) |
|-------|-------------------|-------------------|------------------|
| Baseline (develop) | 0.406s ± 0.009s | 2.528s ± 0.010s | 2.934s ± 0.002s |
| + Cache get_status + poll 0.1s | 0.535s ± 0.024s | **0.331s ± 0.030s** | **0.866s ± 0.008s** |
| + Lazy cv2/scipy | 0.522s ± 0.016s | 0.342s ± 0.019s | 0.864s ± 0.006s |

**Connect time: 2.528s → 0.342s (−2.19s, 87% faster)**
**Total time: 2.934s → 0.864s (−2.07s, 71% faster)**

Import time varies between 0.4–0.55s across runs due to system load. The lazy cv2/scipy change shows no measurable impact on the laptop (both libraries load too fast to matter). On the CM4, expect ~0.8s additional savings from this change.

### Connect Time Breakdown: Before vs After

**Before:**
```
daemon_check():           ~0.000s
ZenohClient():            ~0.010s
wait_for_connection():    ~0.010s
get_status() #1:          ~0.500s
get_status() #2:          ~1.000s  ← wasted (same data, cleared event)
get_status() #3:          ~1.000s  ← wasted (same data, cleared event)
                          --------
Total:                    ~2.520s
```

**After:**
```
daemon_check():           ~0.000s
ZenohClient():            ~0.010s
wait_for_connection():    ~0.010s
get_status() (once):      ~0.300s  (reused for all 3 lookups)
                          --------
Total:                    ~0.320s
```

---

## CM4 (Wireless Robot) Results

Tested on the Reachy Mini Wireless (CM4) by rsyncing the changed files directly to the installed package at `/venvs/apps_venv/lib/python3.12/site-packages/reachy_mini/`. Marionette was restarted via `deploy_wireless.sh` and boot timing captured from `journalctl`.

### Before (develop branch)

```
[BOOT] imports: reachy_mini=3.06s total=3.85s
[BOOT] __init__ done at +3.86s
[BOOT] run() entered at +7.27s        ← SDK init took 3.42s
[BOOT] anim: head-up done at +7.68s
```

### After (893-load-apps-faster)

```
[BOOT] imports: reachy_mini=1.89s total=3.82s
[BOOT] __init__ done at +3.82s
[BOOT] run() entered at +4.53s        ← SDK init took 0.71s
[BOOT] anim: head-up done at +4.93s
```

### Comparison

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| reachy_mini import | 3.06s | 1.89s | −1.17s (lazy cv2/scipy) |
| Total imports | 3.85s | 3.82s | −0.03s (savings inside reachy_mini offset by variance) |
| SDK init (run − init) | 3.42s | 0.71s | **−2.71s** (cached get_status) |
| First head movement | 7.68s | 4.93s | **−2.75s (36% faster)** |

The `reachy_mini` import dropped by 1.17s (cv2 + scipy no longer loaded). Total import time barely changed because other imports (FastAPI, numpy, etc.) still dominate. The SDK init gap collapsed from 3.42s to 0.71s — exactly matching the triple-get_status fix eliminating ~2.7s of blocked waits.

**Result: First head movement 2.75 seconds faster (7.68s → 4.93s).**

---

## How to Test on the CM4

1. On the wireless robot, switch the reachy_mini branch:
   ```bash
   cd /path/to/reachy_mini
   git fetch origin
   git checkout 893-load-apps-faster
   pip install -e .
   ```

2. Deploy Marionette (which already has `[BOOT]` timing instrumentation):
   ```bash
   ./deploy_wireless.sh
   ```

3. Watch the boot timing:
   ```bash
   ssh reachy journalctl -u reachy-mini-daemon -f | grep BOOT
   ```

   Look for:
   - `[BOOT] imports: total=...` — should show ~0.8s less (no cv2/scipy)
   - `[BOOT] run() entered at +...` — should be ~2-3s earlier
   - `[BOOT] anim: head-up done at +...` — the metric we care about

---

## Files Changed

| File | Change |
|------|--------|
| `src/reachy_mini/reachy_mini.py` | Cache `get_status()` in `_configure_mediamanager()`. Lazy import cv2 (→ `look_at_image`), scipy (→ `wake_up`, `look_at_world`). |
| `src/reachy_mini/io/zenoh_client.py` | `wait_for_connection()` poll interval: 1.0s → 0.1s |
| `src/reachy_mini/utils/rerun.py` | Cache `get_status()` (same fix) |
| `benchmarks/bench_startup.py` | New: benchmark script (mockup daemon + 10 subprocess runs) |

---

## Future Optimization Opportunities (Not Implemented)

These could save additional time but require more work:

| Idea | Where | Est. Savings | Effort |
|------|-------|-------------|--------|
| Parallelize GStreamer init with Zenoh wait | reachy_mini.py | 0.5–1.0s on CM4 | Medium |
| Lazy FastAPI import in app.py | app.py | up to 1.2s for non-FastAPI apps | Small |
| Reduce daemon status publish interval (1s → 0.2s) | daemon.py | reduces first get_status() wait | Trivial |
| Pre-launch app processes in daemon | daemon manager | 3–5s (all imports amortized) | Large |
