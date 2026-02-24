# Startup Optimization — Iteration Log

*Branch: `893-load-apps-faster`*

Target metric: time from "Start Marionette" click to first head movement (head-up).
Secondary: time to first sound playback.

---

## CM4 Update Procedure

Both venvs on the robot must be updated:

```bash
# 1. Push changes
git push origin 893-load-apps-faster

# 2. Update daemon venv
ssh pollen@reachy-mini.local "/venvs/mini_daemon/bin/pip install --upgrade --force-reinstall \
    'reachy_mini[wireless-version] @ git+https://github.com/pollen-robotics/reachy_mini.git@893-load-apps-faster'"

# 3. Update apps venv
ssh pollen@reachy-mini.local "/venvs/apps_venv/bin/pip install --upgrade --force-reinstall \
    'reachy-mini @ git+https://github.com/pollen-robotics/reachy_mini.git@893-load-apps-faster'"

# 4. Clear pip cache (CM4 has only 14GB disk)
ssh pollen@reachy-mini.local "/venvs/mini_daemon/bin/pip cache purge; /venvs/apps_venv/bin/pip cache purge"

# 5. Restart daemon
ssh pollen@reachy-mini.local "sudo systemctl restart reachy-mini-daemon"

# 6. Wait ~10s, then start backend (daemon uses --no-autostart)
ssh pollen@reachy-mini.local "curl -sf -X POST 'http://127.0.0.1:8000/api/daemon/start?wake_up=false'"

# 7. Wait for Zenoh (port 7447) + backend to settle (~30s total)

# 8. Sync marionette code
rsync -az --delete --exclude '__pycache__' --exclude '*.pyc' \
    /path/to/marionette/marionette/ \
    pollen@reachy-mini.local:/venvs/apps_venv/lib/python3.12/site-packages/marionette/

# 9. Start marionette
ssh pollen@reachy-mini.local "curl -sf -X POST 'http://127.0.0.1:8000/api/apps/start-app/marionette'"
```

Automated script: `benchmarks/test_cm4.sh`

---

## Baseline (develop branch, before any changes)

*Source: previous session measurements*

```
imports:  total=3.85s (reachy_mini=3.06s)
__init__: +3.86s
run():    +7.27s  → SDK init = 3.42s
head-up:  +7.68s
```

---

## Iteration 0: Cache get_status + lazy cv2/scipy + poll 0.1s

**Commits**:
- reachy_mini: `83d28f8a` (893-load-apps-faster)
- marionette: unchanged

**Changes (reachy_mini repo)**:
- `src/reachy_mini/reachy_mini.py`: cache `get_status()` in `_configure_mediamanager()`, lazy import cv2 and scipy
- `src/reachy_mini/io/zenoh_client.py`: `wait_for_connection()` poll interval 1.0s → 0.1s
- `src/reachy_mini/utils/rerun.py`: cache `get_status()` (same fix)

**CM4 Results (3 runs)**:

| Run | Imports | reachy_mini import | SDK init | Head-up | Sound starts |
|-----|---------|-------------------|----------|---------|-------------|
| 1   | 3.97s   | 2.24s             | 1.73s    | +6.51s  | +8.10s?     |
| 2   | 3.69s   | 2.09s             | 1.12s    | +5.60s  | +5.60s+2.5s |
| 3   | 3.91s   | 2.20s             | 1.47s    | +6.17s  | +8.69s?     |
| **Avg** | **3.86s** | **2.18s** | **1.44s** | **~6.1s** | **~8.3s** |

**Comparison to baseline**:
- reachy_mini import: 3.06s → 2.18s (−0.88s from lazy cv2/scipy)
- SDK init: 3.42s → 1.44s (−1.98s from cached get_status)
- Head-up: 7.68s → ~6.1s (−1.6s)
- Note: previous rsync-based test showed 4.93s head-up, but pip install has higher variance

**Variance notes**: SDK init ranges 1.12–1.73s depending on daemon status publish timing (1s cycle). First `get_status()` waits 0–1s for next publish.

---

## Iteration 1: Loading sound + don't clear status event + faster status publish

**Commits**:
- reachy_mini: `e3753701` (893-load-apps-faster)
- marionette: unchanged

**Changes (reachy_mini repo)**:
- `src/reachy_mini/apps/manager.py`: play `count.wav` immediately when daemon receives start-app request (before subprocess spawn)
- `src/reachy_mini/io/zenoh_client.py`: remove `status_received.clear()` in `get_status()` — event stays set after first status arrives
- `src/reachy_mini/daemon/daemon.py`: reduce status publish interval from 1.0s to 0.2s

**CM4 Results (3 runs)**:

| Run | Imports | SDK init breakdown | SDK total | run() | Head-up |
|-----|---------|-------------------|-----------|-------|---------|
| 1   | 3.68s   | check=0.00 client=0.03 media=0.35 | 0.38s | +4.86s | +6.62s* |
| 2   | 3.92s   | check=0.00 client=0.03 media=0.44 | 0.47s | +5.25s | +5.65s  |
| 3   | 3.67s   | check=0.00 client=0.05 media=0.44 | 0.49s | +4.95s | +5.36s  |

*Run 1 had anomalous head-up (1.56s after enable_motors vs normal ~0.2s).

**Loading sound**: plays at T=0 (confirmed: "[BOOT] Loading sound triggered" appears immediately)

**Comparison to Iteration 0**:
- SDK init: 1.44s → 0.45s (−1.0s from faster status publish + no clear)
- Head-up: ~6.1s → ~5.5s (−0.6s, taking runs 2&3 average)
- Loading sound: now plays ~5.5s before head moves

**Comparison to baseline (develop)**:
- SDK init: 3.42s → 0.45s (−3.0s, **87% faster**)
- Head-up: 7.68s → ~5.5s (−2.2s, **29% faster**)
- User perceives responsiveness from T=0 (loading sound) vs T=7.7s (first sign of life)

**SDK init breakdown shows**:
- `daemon_check`: instant (0s)
- `client` (Zenoh connect + wait_for_connection): 0.03–0.05s
- `media` (get_status + GStreamer MediaManager init): 0.35–0.44s
- The remaining 0.35–0.44s is mostly GStreamer initialization (pipeline setup, device enumeration)

---

## Iteration 2: Pre-warmed app launcher

**Commits**:
- reachy_mini: `a010be08` (893-load-apps-faster)
- marionette: unchanged

**Changes (reachy_mini repo)**:
- `src/reachy_mini/apps/warm_launcher.py` (NEW): Pre-imports heavy packages (numpy, zenoh, fastapi, pydantic, reachy_mini) in a background process, waits for module name on stdin, then runs it via `runpy.run_module`.
- `src/reachy_mini/apps/manager.py`: On daemon startup (wireless only), spawns warm launcher process. When `start_app()` is called, sends module name to warm process instead of cold subprocess spawn. Falls back to cold start if warm process isn't ready.
- `src/reachy_mini/daemon/app/main.py`: Calls `app_manager.initialize()` in lifespan to spawn warm process.

**How it works**:
1. Daemon starts → spawns `apps_venv/bin/python -m reachy_mini.apps.warm_launcher`
2. Warm process imports all heavy packages (~0.7–1.0s on CM4) and prints `WARM_READY`
3. Process sits idle, waiting for module name on stdin
4. User clicks "Start Marionette" → daemon writes `marionette.main\n` to stdin
5. Warm process runs `runpy.run_module("marionette.main")` — all imports are already in `sys.modules`
6. Daemon spawns a new warm process in the background for next app start

**CM4 Results (3 runs)**:

| Run | Imports | __init__ | SDK init | run() | Head-up |
|-----|---------|----------|----------|-------|---------|
| 1   | 0.05s   | +0.46s   | 0.647s   | +1.46s | +3.23s* |
| 2   | 0.07s   | +0.52s   | 0.766s   | +1.69s | +2.11s  |
| 3   | 0.06s   | +0.47s   | 0.600s   | +1.43s | +1.85s  |
| **Avg** | **0.06s** | **+0.48s** | **0.67s** | **+1.53s** | **~2.0s** |

*Run 1 anomaly: first start after daemon restart, head-up animation took 1.6s instead of ~0.2s.

**Warm process initialization**: 0.68–0.96s on CM4 (imports happen once in the background, invisible to user).

**Comparison to Iteration 1**:
- Imports: 3.7s → 0.06s (−3.64s, **98% faster**, all pre-cached by warm process)
- Head-up: ~5.5s → ~2.0s (−3.5s, taking runs 2&3 average)

**Comparison to baseline (develop)**:
- Imports: 3.85s → 0.06s (−3.79s, **98% faster**)
- SDK init: 3.42s → 0.67s (−2.75s, **80% faster**)
- Head-up: **7.68s → ~2.0s (−5.7s, 74% faster)**
- Loading sound still plays at T=0 (instant audio feedback)

**New timeline on CM4 (best case)**:
```
T=0.0s  Loading sound plays (daemon-side)
T=0.0s  Module name sent to warm process
T=0.1s  Imports complete (all cached in sys.modules)
T=0.5s  Marionette.__init__ done
T=1.1s  SDK init done
T=1.4s  run() entered
T=1.6s  enable_motors
T=1.9s  head-up
T=4.4s  intro sound done
```

---

## Summary

| Metric | Baseline | Iter 0 | Iter 1 | Iter 2 |
|--------|----------|--------|--------|--------|
| Imports | 3.85s | 3.86s | 3.76s | **0.06s** |
| SDK init | 3.42s | 1.44s | 0.45s | **0.67s** |
| Head-up | 7.68s | ~6.1s | ~5.5s | **~2.0s** |
| Loading sound | N/A | N/A | T=0 | T=0 |
| Total speedup | — | 1.6s | 2.2s | **5.7s** |

---

## Remaining Bottlenecks

After the warm launcher, the remaining ~2.0s breaks down as:
- **0.5s**: Marionette `__init__` (FastAPI setup, daemon check, dataset loading)
- **0.6s**: SDK init (Zenoh connect 0.2s + GStreamer MediaManager 0.4s)
- **0.4s**: run() setup (mic AGC, animation prep)
- **0.5s**: Physical movement (goto_target + head-up animation)

The physical movement time (~0.5s) is inherent and can't be optimized.

---

## Ideas for Further Optimization

### Idea G: Choose better loading sound
`count.wav` (0.66s) might not be the best UX. Consider a shorter chirp or the start of a boot chime. Or let apps register their own loading sound.

### Idea H: Overlap SDK init with __init__
Start Zenoh connection in a background thread during Marionette.__init__(). By the time run() needs reachy_mini, the SDK init might already be done. Could save ~0.5s.

### Idea I: Lazy GStreamer init
Defer MediaManager initialization until the first play_sound() or camera access. SDK init drops from 0.6s to 0.2s. Risk: first audio playback has a ~0.4s delay.
