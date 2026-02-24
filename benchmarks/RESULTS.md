# Startup Benchmark Results

Machine: Ubuntu laptop (not CM4). Local mockup daemon. `media_backend='no_media'`.
Each measurement is 10 runs with a warmup run.

| State | Import (mean±std) | Connect (mean±std) | Total (mean±std) |
|-------|-------------------|-------------------|------------------|
| Baseline (develop) | 0.406s ± 0.009s | 2.528s ± 0.010s | 2.934s ± 0.002s |
| + Cache get_status + poll 0.1s | 0.535s ± 0.024s | 0.331s ± 0.030s | 0.866s ± 0.008s |
| + Lazy cv2/scipy | 0.522s ± 0.016s | 0.342s ± 0.019s | 0.864s ± 0.006s |

## Key Findings

### Root Cause of Slow Startup: Triple `get_status()` in `_configure_mediamanager`

The biggest bottleneck was in `_configure_mediamanager()` which called `self.client.get_status()` **three times** (lines 194, 246, 249 of reachy_mini.py). Each call clears the `status_received` event, so the next call blocks until the daemon publishes a fresh status message — which happens every **1 second** (`time.sleep(1)` in `daemon._publish_status`).

Result: every `ReachyMini()` creation wasted ~2 seconds waiting for redundant status updates.

**Fix**: Cache the result of the first `get_status()` call and reuse it for all three lookups.

### Connect Time Breakdown (Baseline)

```
daemon_check():           ~0.000s  (process scan)
ZenohClient():            ~0.010s  (zenoh.open + subscribers)
wait_for_connection():    ~0.010s  (events set quickly with local daemon)
get_status() #1:          ~0.500s  (wait for first status message)
get_status() #2:          ~1.000s  (wait for SECOND status — previous cleared the event!)
get_status() #3:          ~1.000s  (wait for THIRD status — same problem)
                          --------
Total:                    ~2.520s
```

### After Fix

```
daemon_check():           ~0.000s
ZenohClient():            ~0.010s
wait_for_connection():    ~0.010s
get_status() (once):      ~0.300s  (wait for first status, reuse for all 3 lookups)
                          --------
Total:                    ~0.320s
```

### Lazy cv2/scipy

Not measurable on the laptop (both load in <0.05s). On the CM4 where cv2 takes ~0.3s and scipy ~0.5s, this should save ~0.8s from the import phase.

### Poll Interval Change

Changed `Event.wait(timeout=1.0)` to `Event.wait(timeout=0.1)` in `wait_for_connection()`. This makes the Zenoh connection check 10x more responsive when data hasn't arrived yet. Minimal impact on the laptop (data arrives almost instantly from local daemon), but should help on the CM4 where Zenoh message delivery is slower.

## Changes Made

1. **`src/reachy_mini/reachy_mini.py`**:
   - Cached `get_status()` result in `_configure_mediamanager()` — used for all 3 lookups
   - Moved `import cv2` from module level to inside `look_at_image()`
   - Moved `from scipy.spatial.transform import Rotation as R` from module level to inside `wake_up()` and `look_at_world()`

2. **`src/reachy_mini/io/zenoh_client.py`**:
   - Reduced `wait_for_connection()` poll interval from 1.0s to 0.1s

3. **`src/reachy_mini/utils/rerun.py`**:
   - Cached `get_status()` result (same pattern fix as reachy_mini.py)

## Expected CM4 Impact

On the CM4, the daemon also publishes status every 1 second. The triple get_status() fix should save ~2 seconds there as well. Combined with the lazy cv2/scipy imports (~0.8s), the total startup should improve by ~2.8s.

Current CM4 timeline: first head movement at T+7.3s
Expected after fix: first head movement at ~T+4.5s
