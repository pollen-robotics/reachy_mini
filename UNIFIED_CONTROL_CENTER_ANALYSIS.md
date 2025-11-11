# Unified Control Center - Architectural Analysis

**Date:** October 30, 2025
**Status:** 🔴 Fundamental architecture problems
**Recommendation:** Complete refactor required

---

## TL;DR - What Went Wrong

**The control center is trying to replicate the conversation app's internal architecture instead of using the daemon's REST API.**

This is like writing your own TCP/IP stack instead of using sockets. The daemon exists specifically to provide a clean REST API for robot control, but the control center is bypassing it and trying to run the same threading/IK/movement code that already lives in the daemon.

---

## The Correct Architecture

### How It Should Work

```
┌─────────────────────────────────────────────────────┐
│  UNIFIED CONTROL CENTER (ImGui Desktop App)         │
│                                                      │
│  • Pure REST API client                             │
│  • No SDK imports                                   │
│  • No threading (except UI)                         │
│  • No IK calculations                               │
│  • No direct robot control                          │
│                                                      │
│  Just sends HTTP requests:                          │
│  - GET /api/camera/frame                            │
│  - POST /api/move/set_target                        │
│  - GET /api/state/full                              │
└─────────────────────────────────────────────────────┘
                        ↓ HTTP
┌─────────────────────────────────────────────────────┐
│  DAEMON (FastAPI Server)                            │
│                                                      │
│  • Owns the camera                                  │
│  • Runs face tracking (if enabled)                  │
│  • Handles IK calculations                          │
│  • Controls robot hardware                          │
│  • Manages all threads                              │
│                                                      │
│  REST API Endpoints:                                │
│  - /api/kinematics/look_at_image                    │
│  - /api/camera/frame                                │
│  - /api/move/set_target                             │
│  - /api/state/full                                  │
└─────────────────────────────────────────────────────┘
                        ↓ SDK
┌─────────────────────────────────────────────────────┐
│  HARDWARE (MuJoCo Sim or Physical Robot)            │
└─────────────────────────────────────────────────────┘
```

### What They Actually Built

```
┌─────────────────────────────────────────────────────┐
│  UNIFIED CONTROL CENTER                             │
│                                                      │
│  • Imports ReachyMini SDK (wrong!)                  │
│  • Creates its own robot instance (wrong!)          │
│  • Runs CameraWorker thread (wrong!)                │
│  • Runs MovementManager thread (wrong!)             │
│  • Does its own IK calculations (wrong!)            │
│  • Tries to share camera with daemon (wrong!)       │
│                                                      │
│  Threading Chaos:                                   │
│  - CameraWorker (30Hz face detection)               │
│  - MovementManager (100Hz control loop)             │
│  - Tracking UI loop (10Hz updates)                  │
│  - Breathing idle motion thread                     │
│  - Mood loop threads                                │
│  - HTTP server for mood triggers (port 5002)        │
└─────────────────────────────────────────────────────┘
       ↓ SDK (bypassing daemon!)
┌─────────────────────────────────────────────────────┐
│  DAEMON (idle, unused except for camera frames)     │
│                                                      │
│  • Has all the needed endpoints                     │
│  • Nobody is using them properly                    │
└─────────────────────────────────────────────────────┘
```

---

## Specific Problems

### 1. Unnecessary SDK Usage

**File:** `tracking_panel.py` lines 184-196

```python
# Phase 2: Use shared robot instance from sdk_manager (don't create duplicate)
if not self.sdk_manager.is_ready():
    self._set_status("SDK not ready", is_error=True)
    logger.error("Cannot start tracking: SDK not initialized")
    return

self.robot = self.sdk_manager.reachy
```

**Problem:** Control center should NEVER import or use ReachyMini SDK. That's the daemon's job.

**Solution:** Remove all SDK imports. Use only `requests` to call daemon REST API.

---

### 2. Reinventing CameraWorker/MovementManager

**File:** `tracking_panel.py` lines 206-226

```python
# Initialize CameraWorker with wrapped media_manager
self.camera_worker = CameraWorker(
    media_manager=media_manager,
    head_tracker=self.head_tracker,
    daemon_client=ik_client
)

# Initialize movement manager with CameraWorker
self.movement_manager = MovementManager(
    current_robot=self.robot,
    camera_worker=self.camera_worker
)

# Start CameraWorker thread
self.camera_worker.start()

# Start MovementManager thread
self.movement_manager.start()
```

**Problem:** These classes belong in the conversation app, not the control center. They're designed for standalone apps that need direct SDK control.

**Solution:** Remove entirely. Face tracking should be a daemon feature, not a control center feature.

---

### 3. Unnecessary Wrapper Classes

**File:** `tracking_panel.py` lines 18-76

```python
class IKClientWrapper:
    """Wrapper to calculate IK via daemon REST API."""
    def look_at_image(self, u: int, v: int):
        # ... 30 lines of code that just calls daemon endpoint ...

class MediaManagerWrapper:
    """Wrapper to provide frames from camera_panel to CameraWorker."""
    def get_frame(self):
        return self.camera_panel.get_latest_frame()
```

**Problem:** These wrappers exist only to make the control center's SDK-based code work. If the control center wasn't using the SDK, these wouldn't exist.

**Solution:** Remove wrappers. Call daemon endpoints directly.

---

### 4. Camera Sharing Nightmare

**File:** `camera_panel.py` lines 89-116

```python
def update_camera_feed(self):
    """Update the camera feed by fetching from daemon API."""
    try:
        response = requests.get(
            f"{self.daemon_url}/api/camera/frame",
            timeout=0.1
        )
        # ... decode frame ...
        self.latest_frame = frame
```

**Then:** `tracking_panel.py` lines 202

```python
# Create media manager wrapper that gets frames from camera_panel
media_manager = MediaManagerWrapper(self.camera_panel)
```

**Problem:** Camera panel fetches frames from daemon, then passes them to MediaManagerWrapper, which passes them to CameraWorker, which then does face detection and calls the daemon IK endpoint. This is a circular dependency nightmare.

**Solution:** Daemon owns the camera. If you want face tracking, tell the daemon to enable face tracking via REST API. Don't try to get frames and do your own tracking.

---

### 5. The Daemon Already Has Everything

**Daemon endpoint that exists:** `/api/kinematics/look_at_image`

```python
@router.get("/look_at_image")
async def look_at_image(
    u: int,
    v: int,
    use_pose_matrix: bool = False,
    backend: Backend = Depends(get_backend),
) -> Any:
    """Calculate head pose to look at a pixel position in camera image."""
    # ... full IK calculation with camera intrinsics ...
    return as_any_pose(target_head_pose, use_pose_matrix)
```

**What the control center does instead:**

1. Gets frame from daemon
2. Runs its own face detection (YOLO)
3. Calls daemon's look_at_image endpoint
4. Converts the result to 4x4 matrix
5. Inverts the yaw (for coordinate system fix)
6. Interpolates smoothly
7. Converts back to SDK format
8. Sends to robot via SDK

**This is insane.** The daemon already has camera intrinsics, IK solver, and can do all of this internally.

---

### 6. Threading Explosion

**Control center currently runs:**

- `CameraWorker` thread (30Hz) - face detection + IK
- `MovementManager` thread (100Hz) - breathing + face blend
- `tracking_thread` (10Hz) - UI state updates
- `http_server` thread - mood triggers on port 5002
- `mood_loop_thread` - random emotion player
- Plus all the ImGui/OpenGL rendering threads

**Daemon should handle:**

- Camera capture
- Face detection
- IK calculations
- Movement blending
- All robot control

**Control center should have:**

- Single UI thread (ImGui)
- Periodic REST API polling (low frequency)

---

## Coordinate System Issues

**From TRACKING_ISSUES.md:**

> When head is on left side of screen, bounding box appears on right side.

**Root Cause:**

They're mixing two different coordinate systems:

1. **Daemon coordinate system:** Camera frames with specific intrinsics, expects pixel coordinates in daemon's format
2. **SDK coordinate system:** ReachyMini's internal coordinate system

The control center is trying to use both simultaneously, leading to double inversions:

- `HeadTracker` flips x-coordinate (for "robot perspective")
- `CameraWorker` inverts yaw (for "camera mirrored")
- Overlay drawing doesn't compensate properly

**Solution:** Use ONLY daemon REST API. Daemon handles all coordinate transforms internally.

---

## What the Conversation App Does Differently

**Conversation app architecture (correct for standalone app):**

```
ConversationApp.py
  ↓ direct SDK import
ReachyMini SDK
  ↓ owns camera
MediaManager
  ↓ provides frames
CameraWorker (face tracking)
  ↓ calculates IK
MovementManager (breathing + tracking blend)
  ↓ sends commands
Robot Hardware
```

**Why this works:** The conversation app is a standalone application that needs full control of the robot. It's not designed to coexist with a daemon.

**Why control center can't copy this:** The control center is designed to work WITH the daemon, not replace it. The daemon is the single point of robot control.

---

## The Correct Implementation

### Face Tracking Feature (Example)

**Option 1: Daemon Has Face Tracking (recommended)**

If daemon has built-in face tracking:

```python
# In control center's tracking_panel.py
def start_tracking(self):
    """Start face tracking via daemon."""
    response = requests.post(
        f"{self.daemon_url}/api/tracking/enable",
        json={"tracker": "yolo" if self.use_yolo else "mediapipe"}
    )
    if response.status_code == 200:
        self._set_status("Tracking enabled", is_error=False)
    else:
        self._set_status("Failed to enable", is_error=True)

def stop_tracking(self):
    """Stop face tracking via daemon."""
    requests.post(f"{self.daemon_url}/api/tracking/disable")

def render(self):
    """Render tracking panel - just UI, no logic."""
    if imgui.button("Start Tracking"):
        self.start_tracking()
    if imgui.button("Stop Tracking"):
        self.stop_tracking()
```

**That's it.** ~10 lines instead of 674 lines.

**Option 2: Daemon Doesn't Have Face Tracking Yet**

Add face tracking to the daemon:

```python
# In daemon's app/routers/tracking.py (new file)
from reachy_mini_conversation_app.camera_worker import CameraWorker
from reachy_mini_conversation_app.moves import MovementManager

router = APIRouter(prefix="/tracking")

# Global instances
camera_worker: CameraWorker | None = None
movement_manager: MovementManager | None = None

@router.post("/enable")
async def enable_tracking(tracker: str, backend: Backend = Depends(get_backend)):
    """Enable face tracking with specified tracker (yolo/mediapipe)."""
    global camera_worker, movement_manager

    # Initialize tracker
    if tracker == "yolo":
        from reachy_mini_conversation_app.vision.yolo_head_tracker import HeadTracker
        head_tracker = HeadTracker()
    else:
        from reachy_mini_toolbox.vision import HeadTracker
        head_tracker = HeadTracker()

    # Start workers
    camera_worker = CameraWorker(
        media_manager=backend.media_manager,
        head_tracker=head_tracker,
        daemon_client=None  # Use backend directly
    )
    camera_worker.start()

    movement_manager = MovementManager(
        current_robot=backend.robot,
        camera_worker=camera_worker
    )
    movement_manager.start()

    return {"status": "tracking enabled"}

@router.post("/disable")
async def disable_tracking():
    """Disable face tracking."""
    global camera_worker, movement_manager

    if camera_worker:
        camera_worker.stop()
        camera_worker = None

    if movement_manager:
        movement_manager.stop()
        movement_manager = None

    return {"status": "tracking disabled"}

@router.get("/status")
async def get_tracking_status():
    """Get current tracking status."""
    return {
        "enabled": camera_worker is not None,
        "face_detected": camera_worker.face_detected if camera_worker else False,
        "face_position": camera_worker.get_last_face_position() if camera_worker else None
    }
```

**Now the control center can use it with simple REST calls.**

---

## Migration Path

### Phase 1: Add Tracking Endpoints to Daemon

1. Create `src/reachy_mini/daemon/app/routers/tracking.py`
2. Import CameraWorker and MovementManager from conversation app
3. Add `/api/tracking/enable`, `/api/tracking/disable`, `/api/tracking/status` endpoints
4. Test with curl:
   ```bash
   curl -X POST http://localhost:8100/api/tracking/enable -d '{"tracker":"yolo"}'
   curl http://localhost:8100/api/tracking/status
   curl -X POST http://localhost:8100/api/tracking/disable
   ```

### Phase 2: Simplify Control Center

1. Remove all SDK imports from `tracking_panel.py`
2. Remove `IKClientWrapper` class
3. Remove `MediaManagerWrapper` class
4. Remove `CameraWorker` and `MovementManager` initialization
5. Replace with simple REST API calls:
   ```python
   def start_tracking(self):
       requests.post(f"{self.daemon_url}/api/tracking/enable",
                    json={"tracker": "yolo" if self.use_yolo else "mediapipe"})

   def stop_tracking(self):
       requests.post(f"{self.daemon_url}/api/tracking/disable")

   def update_ui(self):
       response = requests.get(f"{self.daemon_url}/api/tracking/status")
       data = response.json()
       self.face_detected = data["enabled"] and data["face_detected"]
   ```

### Phase 3: Remove Dead Code

1. Delete wrappers
2. Remove threading logic
3. Remove SDK manager dependency
4. Remove mood HTTP server (port 5002) - move to daemon if needed
5. Simplify camera panel - just display frames from daemon, no capture logic

---

## File Size Comparison

### Current (Broken)

- `tracking_panel.py`: 674 lines (threading, workers, wrappers, HTTP server)
- `camera_panel.py`: 160 lines (camera capture, frame management)
- Dependencies: reachy_mini, reachy_mini_conversation_app, cv2, numpy, scipy

### After Refactor (Correct)

- `tracking_panel.py`: ~50 lines (REST API calls, UI rendering)
- `camera_panel.py`: ~30 lines (display frames from daemon)
- Dependencies: requests, imgui

**90% code reduction.**

---

## Lessons Learned

### What Went Wrong

1. **Copied code without understanding architecture** - CameraWorker/MovementManager are conversation app internals, not reusable components
2. **Mixed abstraction layers** - SDK and REST API don't mix
3. **Didn't check daemon capabilities** - Assumed daemon couldn't do things it can do
4. **Tried to share resources** - Camera can't be owned by two processes

### Design Principles

1. **Daemon owns hardware** - Control center is just a UI
2. **REST API is the interface** - No SDK imports in clients
3. **Keep it simple** - If you're writing threading code in a UI, something is wrong
4. **Check existing endpoints first** - Before writing 100 lines, check if there's a 1-line API call

---

## Recommendation

**Stop.** Don't try to fix coordinate inversions or add error handling. The entire architecture is wrong.

**Start over with this pattern:**

```python
# tracking_panel.py (complete rewrite)
class TrackingPanel:
    def __init__(self, daemon_url):
        self.daemon_url = daemon_url
        self.tracking_active = False

    def start_tracking(self):
        response = requests.post(f"{self.daemon_url}/api/tracking/enable")
        self.tracking_active = (response.status_code == 200)

    def stop_tracking(self):
        requests.post(f"{self.daemon_url}/api/tracking/disable")
        self.tracking_active = False

    def render(self):
        if imgui.button("Start" if not self.tracking_active else "Stop"):
            self.start_tracking() if not self.tracking_active else self.stop_tracking()
```

**If daemon doesn't have `/api/tracking/enable` yet, add it to the daemon, not the control center.**

---

## Questions to Answer

Before continuing development:

1. **Does the daemon need face tracking as a built-in feature?**
   - If yes: Add tracking router to daemon
   - If no: Face tracking belongs in standalone apps (conversation app), not control center

2. **What is the control center's purpose?**
   - Manual control? (sliders, buttons)
   - Monitoring? (display current state)
   - Testing? (play moves, check responses)

3. **Should control center enable features or just trigger them?**
   - Enable: Control center decides when tracking starts (current approach)
   - Trigger: Daemon has modes, control center just switches modes (better approach)

---

*This analysis written to prevent further development on a fundamentally broken architecture. Fix the foundation before adding features.*
