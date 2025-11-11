# Laura Carson - Reachy Mini Development Notes

**Created:** October 17, 2025
**Context:** Beta testing Reachy Mini robot from Pollen Robotics
**Primary Focus:** Web-based choreography builder and simulator interface

---

## What We've Accomplished

### 1. MuJoCo Simulator Video Streaming (October 16-17, 2025)

**Problem:** Needed to integrate the MuJoCo 3D simulator visualization into a web interface for remote monitoring and control.

**Solution:** Added offscreen rendering capability to the MuJoCo backend that streams MJPEG video at 20fps via HTTP multipart streaming.

**Key Files Modified:**
- `src/reachy_mini/daemon/backend/mujoco/backend.py` - Added `viewer_rendering_loop()` method
- `src/reachy_mini/daemon/app/routers/video.py` - Created new video streaming endpoint
- `src/reachy_mini/daemon/app/main.py` - Registered video router

**Technical Details:**
- Offscreen renderer captures 640x480 external 3D view
- Camera positioned at: distance=0.8, azimuth=160°, elevation=-20°, lookat=[0,0,0.15]
- JPEG compression at quality 75 for bandwidth efficiency
- Thread-safe frame access using Lock for concurrent reads
- MJPEG served via `multipart/x-mixed-replace; boundary=frame` HTTP response

**Installation Lesson:** Discovered package was installed in site-packages (not editable mode). Fixed with:
```bash
pip install -e ".[mujoco]"
```
This ensures code changes take effect without reinstalling.

---

### 2. Web Interface Architecture Evolution

**Phase 1: Control Center Concept**
- Initial idea: All-in-one interface with move library, video feed, and controls
- Used 100+ radio buttons for individual move selection
- Video feed was interactive (click-to-look on video itself)

**Phase 2: Separation of Concerns (Critical Pivot)**
- **Key Realization:** MuJoCo passive viewer needs native camera controls (left-click rotate, right-click pan, scroll zoom)
- **Solution:** Separated click-to-look into dedicated "Quick Look Direction" pad
- Preserved MuJoCo's interactive 3D viewer functionality
- Video becomes pure visualization, not an interaction surface

**Phase 3: Choreography Builder Focus (Current)**
- **Purpose Shift:** Web UI as choreography creation tool, not full control center
- Replaced 100+ radio buttons with 2 dropdown menus (dance/emotion)
- Added routine sequencing system for building multi-move choreographies
- Consolidated all tools into clean 2-column layout

**Current Layout:**
```
[Video Feed + Look Pad + Head Pose] | [Choreography Builder]
                440px                |      flexible width
```

---

### 3. Routine Sequencing System

**Functionality:**
- Add moves from either library (dance or emotion) to build sequences
- Display numbered routine list with move types (🕺 dance / 😊 emotion)
- Remove individual moves from sequence
- Play entire routine with seamless flow (0ms delay between moves)
- Quick presets: Happy Greeting, Confused Look, Dance Party, Sleepy Time

**API Pattern:**
```javascript
// Sequential execution with async/await
for (let i = 0; i < routine.length; i++) {
    const res = await fetch(`${serverUrl()}/api/move/play/recorded-move-dataset/${library}/${move}`, {
        method: 'POST'
    });
    await waitForMoveComplete(); // Polls /api/move/running endpoint
    // No delay - moves flow immediately
}
```

**Key Technical Decision:** User wanted 0ms delay between moves for seamless choreography flow (originally had 500ms pause).

---

### 4. Live Data Visualization

**WebSocket Integration:**
- Real-time head pose streaming via `/api/state/ws/full`
- Chart.js bar chart displaying 6 DOF (x, y, z, roll, pitch, yaw)
- Updates with `chart.update('none')` for better performance (no animation)

**Click-to-Look Pad:**
- 150px interactive grid with visual feedback
- Coordinate mapping: normalized (-1 to 1) → angle ranges (yaw: ±40°, pitch: ±20°)
- Animated orange circle indicator on click
- Sends `/api/move/goto` with 0.3s minjerk interpolation

---

## Common Questions & Patterns

### Questions About Architecture
- **"Do we need two interfaces?"** - Yes, separation of concerns is critical. MuJoCo viewer for 3D visualization with camera controls, web UI for choreography building.
- **"Can we integrate X into the video feed?"** - Generally no, preserve native MuJoCo controls. Create separate UI elements instead.
- **"How do we handle move timing?"** - Currently 0ms delay, moves flow immediately. Can be adjusted per-routine if needed.

### Questions About Move Libraries
- **"Where are moves stored?"** - Hugging Face datasets:
  - `pollen-robotics/reachy-mini-dances-library` (20 moves)
  - `pollen-robotics/reachy-mini-emotions-library` (83+ moves, including 35 electric_shocked variations)

### Questions About Development Workflow
- **"Why aren't my changes working?"** - Check if package is editable: `pip install -e ".[mujoco]"`
- **"What port is the daemon running on?"** - Default 8100, adjust with `--fastapi-port 8100` flag

---

## Key Concepts Moving Forward

### 1. **Choreography System Architecture** (Not Yet Explored)
User mentioned "after you get done compacting I'll have you read through choreography system again"

**What to expect:**
- JSON choreography files (example: `examples/choreographies/another_one_bites_the_dust.json`)
- More advanced sequencing beyond simple move chaining
- Potentially BPM/timing synchronization (user mentioned "that and the bpm")
- May involve move duration control, transitions, or parallel movements

**Files to investigate:**
- `/examples/choreographies/*.json` - Example choreography structures
- `/src/reachy_mini/motion/recorded_move.py` - Move loading implementation
- `/docs/rest-api.md` - Full API documentation

### 2. **Interpolation Methods**
Four interpolation modes available for smooth motion:
- **linear** - Constant velocity
- **minjerk** - Minimum jerk (smooth acceleration, preferred for natural movement)
- **ease** - Ease-in/ease-out
- **cartoon** - Exaggerated motion profiles

### 3. **REST API Endpoints**
**Move execution:**
```
POST /api/move/play/recorded-move-dataset/{dataset}/{move_name}
POST /api/move/goto (body: {head_pose, antennas, duration, interpolation})
POST /api/move/stop (body: {uuid})
GET /api/move/running (returns array of active move UUIDs)
```

**Video streaming:**
```
GET /api/video/stream (MJPEG multipart stream)
```

**Live data:**
```
WS /api/state/ws/full (WebSocket for real-time pose updates)
```

### 4. **MuJoCo Physics Details**
- Physics timestep: 2ms (500Hz simulation)
- Control decimation: 10x (50Hz control loop)
- Rendering: 25Hz (camera feed), 20Hz (3D viewer stream)
- Scene options: empty, minimal (specified via `--scene` flag)

### 5. **Daemon Startup Command**
```bash
cd /Users/lauras/Desktop/laura/reachy_mini
source /Users/lauras/Desktop/laura/venv/bin/activate
mjpython -m reachy_mini.daemon.app.main --sim --scene minimal --fastapi-port 8100
```

---

## Development Workflow Best Practices

### 1. **Editable Installation**
Always use editable mode for active development:
```bash
pip install -e ".[mujoco]"
```

### 2. **Testing Workflow**
1. Start daemon with simulator
2. Open `move_controller.html` in browser
3. Verify video stream connects
4. Verify WebSocket connects (live pose updates)
5. Test individual moves via dropdowns
6. Build and test routine sequences

### 3. **Debugging Tips**
- Check daemon logs for API errors
- Use browser console (F12) for JavaScript errors
- Verify daemon status: `curl http://localhost:8100/api/daemon/status`
- Check running moves: `curl http://localhost:8100/api/move/running`

### 4. **Port Conflicts**
If port 8100 is in use:
```bash
lsof -ti:8100 | xargs kill
```

---

## Upcoming Exploration Areas

### Immediate Next Steps
1. **Read choreography system documentation** - Understand JSON structure and advanced sequencing
2. **BPM integration** - Add tempo/timing controls to routine builder
3. **Move duration display** - Show estimated time for each move/routine

### Potential Enhancements
- Move preview thumbnails or descriptions
- Save/load custom routines (JSON export/import)
- Keyboard shortcuts for quick move triggering
- Real-time move editing (adjust speed, timing, interpolation)
- Multi-robot synchronization (if testing with multiple Reachy Minis)

### ImGui Native Interface (Separate Project)
- Desktop app for direct MuJoCo viewer integration
- Mouse controls: left-click rotate, right-click pan, scroll zoom
- Overlay controls without web browser
- Libraries: Python + ImGui bindings

---

## Notes on Laura's Development Style

**Preferences:**
- **Concise, direct solutions** - No over-explanation, get to the point
- **Separation of concerns** - Focused tools over complex monoliths
- **Clean UI** - Dropdown menus over radio buttons, organized sections
- **Iterative refinement** - Build, test, adjust based on real usage
- **Practical focus** - "Let's just make it a dance routine maker for now"

**Communication Pattern:**
- Visual feedback important (shares screenshots)
- Hands-on testing ("i took care of it myself, take a look at the html interface")
- Specific feedback ("move X to bottom", "0ms delay")
- Clear priorities ("the moves list and routine builder are the key things, that and the bpm")

**Technical Background:**
- Comfortable with Python, web technologies, robotics
- Understands physics simulation concepts (MuJoCo)
- Familiar with REST APIs, WebSockets, async patterns
- Prefers seeing working examples over theoretical explanations

---

## Related Documentation

- `/Users/lauras/Desktop/laura/reachy_mini/CLAUDE.md` - Full project context and overview
- `/examples/recorded_moves_example.py` - Python version of move playback
- `/examples/choreographies/another_one_bites_the_dust.json` - Example choreography
- `/docs/rest-api.md` - Complete REST API reference
- `/src/reachy_mini/motion/recorded_move.py` - Move loading implementation

---

*These notes capture the development context, technical decisions, and lessons learned during initial Reachy Mini beta testing. Update as new patterns emerge.*
