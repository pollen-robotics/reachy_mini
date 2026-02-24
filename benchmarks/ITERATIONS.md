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

## Ideas for Next Iterations

### Idea A: Early "loading" sound from the daemon
Play a sound immediately when the daemon receives the start-app request, before the Python subprocess even starts. The daemon has access to GStreamer. This gives instant feedback (< 0.1s after click).

### Idea B: Reduce daemon status publish interval (1s → 0.2s)
The first `get_status()` waits for the next publish cycle. With 1s interval, expected wait = 0.5s. With 0.2s interval, expected wait = 0.1s. Saves ~0.4s on average.

### Idea C: Lazy FastAPI import in marionette
`fastapi` takes 1.19–1.29s on the CM4. If we defer it to after the `__init__`, the app could start connecting to Zenoh sooner. But FastAPI is needed before `wrapped_run()` because the app's routes are decorated with `@app.get(...)` at class definition time.

### Idea D: Don't clear `status_received` event in `get_status()`
Instead of clearing the event after each call (which forces the next caller to wait), keep it set. The cached value is fine for startup. This eliminates the timing dependency on the publish cycle entirely.

### Idea E: Parallel import + SDK init
Start the ReachyMini connection in a background thread while imports are still loading. Complex but could overlap the 3.9s import with the 1.4s SDK init.

### Idea F: Pre-warm app process in daemon
Daemon could spawn a Python process that pre-imports common packages, then fork it when an app starts. Saves all import time. Large effort.
