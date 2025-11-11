# Mobile Deployment Guide for MuJoCo Viewer

**Created:** October 18, 2025
**Purpose:** Strategic planning for eventual iPhone/iOS deployment

---

## Current Desktop Viewer

### What Works Now (macOS/Desktop)
- **Platform:** macOS desktop application
- **Graphics:** OpenGL 2.1 (fixed pipeline)
- **UI:** ImGui with GLFW
- **Controls:** Keyboard (F11, Backspace), Mouse (orbit, pan, zoom)
- **Features:**
  - Scene selection (`--scene minimal`)
  - Fullscreen toggle (F11 or button)
  - 3D MuJoCo rendering
  - Real-time physics simulation

### Running Desktop Viewer

```bash
# Windowed mode
python3 desktop_viewer.py --scene minimal

# Fullscreen mode
python3 desktop_viewer.py --scene minimal --fullscreen

# Toggle fullscreen: Press F11 or click button
```

---

## iPhone/iOS Deployment Options

### ❌ What WON'T Work on iPhone

**Direct Desktop Port:**
- ❌ OpenGL is deprecated on iOS (replaced by Metal)
- ❌ GLFW doesn't run on iOS
- ❌ ImGui desktop renderer won't work
- ❌ Python scripts can't run natively on iOS (sandboxing restrictions)

**Bottom line:** The current `desktop_viewer.py` cannot be directly ported to iPhone.

---

## ✅ Recommended Approach: Web-Based Interface

### Architecture

```
┌─────────────────┐
│  iPhone Safari  │  ← User interacts here
│   (Web Browser) │
└────────┬────────┘
         │ HTTP/WebSocket
         ▼
┌─────────────────┐
│  FastAPI Server │  ← Runs on Mac/Pi
│  (port 8100)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ MuJoCo Backend  │  ← Headless rendering
│  (Simulation)   │
└─────────────────┘
```

### Why This Works

1. **Server runs on powerful hardware** (Mac Mini, Raspberry Pi, cloud)
2. **MuJoCo renders headlessly** (no OpenGL window needed)
3. **Video streamed as MJPEG** to browser
4. **Controls via HTML/JS** (touch-friendly)
5. **Works on ANY device** (iPhone, iPad, Android, desktop)

### Implementation Path

#### Phase 1: Enhanced Web UI (Next Step)

Update your existing `move_controller.html`:

```html
<!-- Add 3D viewer stream -->
<img src="http://your-server:8100/api/camera/stream.mjpg" />

<!-- Add scene selector -->
<select id="scene-select">
  <option value="empty">Empty Scene</option>
  <option value="minimal">Minimal Scene</option>
  <option value="theater_stage">Theater Stage</option>
</select>

<!-- Touch controls for camera -->
<div id="camera-controls">
  <button>Orbit</button>
  <button>Pan</button>
  <button>Zoom</button>
</div>
```

**Benefits:**
- Works on iPhone immediately
- No App Store approval needed
- Easy to iterate and update
- Cross-platform (iPhone, iPad, Mac, etc.)

#### Phase 2: Progressive Web App (PWA)

Convert web UI to installable app:

```javascript
// service-worker.js - enables offline mode and home screen install
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open('mujoco-viewer-v1').then(cache => {
      return cache.addAll(['/move_controller.html', '/styles.css']);
    })
  );
});
```

**Features:**
- Install to iPhone home screen
- Fullscreen mode (hides Safari UI)
- Offline capability for UI
- Push notifications (for scene manager events)

#### Phase 3: Native iOS App (Long-term)

If you eventually need native performance:

**Option A: React Native + WebView**
- UI in React Native (native feel)
- 3D view in WebView (MJPEG stream)
- Fastest development time

**Option B: Swift + Metal**
- Full native iOS app
- Port MuJoCo rendering to Metal
- Maximum performance, huge effort

**Option C: Unity + MuJoCo**
- Use Unity engine for iOS
- MuJoCo has Unity bindings
- Good for interactive experiences

---

## Recommended Technology Stack

### For iPhone Web App

```yaml
Frontend (iPhone):
  - HTML5 + CSS3 + JavaScript
  - Touch event handling
  - Canvas API for overlays
  - WebSocket for real-time state
  - MJPEG streaming for video

Backend (Server):
  - FastAPI (already have this!)
  - MuJoCo headless rendering
  - WebSocket server
  - MJPEG encoder

Deployment:
  - Ngrok (quick testing)
  - Tailscale (private network)
  - VPS (production)
```

### Example: Touch-Friendly Camera Control

```javascript
// Pinch to zoom
let touchDistance = 0;
canvas.addEventListener('touchmove', (e) => {
  if (e.touches.length === 2) {
    const touch1 = e.touches[0];
    const touch2 = e.touches[1];
    const newDistance = Math.hypot(
      touch2.pageX - touch1.pageX,
      touch2.pageY - touch1.pageY
    );

    if (touchDistance > 0) {
      const delta = newDistance - touchDistance;
      fetch('/api/camera/zoom', {
        method: 'POST',
        body: JSON.stringify({ delta })
      });
    }
    touchDistance = newDistance;
  }
});

// Swipe to orbit
let touchStart = { x: 0, y: 0 };
canvas.addEventListener('touchstart', (e) => {
  touchStart.x = e.touches[0].pageX;
  touchStart.y = e.touches[0].pageY;
});

canvas.addEventListener('touchend', (e) => {
  const dx = e.changedTouches[0].pageX - touchStart.x;
  const dy = e.changedTouches[0].pageY - touchStart.y;

  fetch('/api/camera/orbit', {
    method: 'POST',
    body: JSON.stringify({ dx, dy })
  });
});
```

---

## API Endpoints Needed

### Extend FastAPI Server

```python
# In reachy_mini/daemon/app/routers/viewer.py

@router.post("/api/scene/load")
async def load_scene(scene_name: str):
    """Dynamically load a different MuJoCo scene."""
    backend.reload_scene(scene_name)
    return {"status": "loaded", "scene": scene_name}

@router.post("/api/camera/orbit")
async def orbit_camera(dx: float, dy: float):
    """Orbit camera based on touch/mouse input."""
    # Update MuJoCo camera
    return {"status": "updated"}

@router.post("/api/camera/zoom")
async def zoom_camera(delta: float):
    """Zoom camera in/out."""
    # Update MuJoCo camera
    return {"status": "updated"}

@router.get("/api/scene/list")
async def list_scenes():
    """Return available scene files."""
    scenes = ["empty", "minimal", "theater_stage"]
    return {"scenes": scenes}
```

---

## Network Considerations

### Local Network (Development)

```bash
# Server on Mac Mini (192.168.1.100)
# iPhone on same WiFi
# Access via: http://192.168.1.100:8100
```

### Remote Access (Production)

**Option 1: Ngrok (Quick Testing)**
```bash
ngrok http 8100
# Access from anywhere: https://abc123.ngrok.app
```

**Option 2: Tailscale (Private VPN)**
```bash
# Both devices on Tailscale network
# Secure, no port forwarding needed
```

**Option 3: VPS Deployment**
```bash
# Deploy to DigitalOcean/AWS/Linode
# Run MuJoCo server in cloud
# Access via: https://yourdomain.com
```

---

## Performance Considerations

### Video Streaming

**MJPEG (Current):**
- ✅ Simple, works everywhere
- ✅ Low latency
- ❌ High bandwidth (~5-10 Mbps)

**WebRTC (Future):**
- ✅ Adaptive bitrate
- ✅ Better compression
- ❌ More complex setup

**Recommendations:**
- Start with MJPEG at 720p, 24fps
- Reduce to 480p on mobile networks
- Upgrade to WebRTC if bandwidth is issue

### Latency

**Target: < 100ms end-to-end**
- Touch input → Server: ~20ms (WiFi)
- Server processing: ~10ms
- MuJoCo render: ~40ms (25 fps)
- Video encode: ~10ms
- Stream to iPhone: ~20ms

**Optimizations:**
- Use WebSocket for controls (faster than HTTP)
- Render at lower resolution for mobile
- Increase frame rate at cost of resolution

---

## Scene Manager Integration

### Mobile-First Scene Control

```javascript
// iPhone sends scene manager command
async function triggerScene(context) {
  const response = await fetch('/api/scene-manager/trigger', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      context: context,
      device: 'iphone',
      capabilities: ['touch', 'accelerometer', 'camera']
    })
  });

  const { scene } = await response.json();

  // Automatically load the appropriate MuJoCo scene
  await fetch('/api/scene/load', {
    method: 'POST',
    body: JSON.stringify({ scene_name: scene.environment })
  });
}

// Example: Conversation context triggers theater scene
triggerScene({
  type: 'debate',
  participants: ['claude', 'laura'],
  mood: 'serious'
});
// → Loads theater_stage.xml with dual positions
```

---

## Development Roadmap

### Immediate (This Week)
- ✅ Desktop viewer with fullscreen
- ✅ Scene selection working
- ⏭️ Test on iPhone via existing web UI

### Short-term (Next 2 Weeks)
- [ ] Add 3D viewer stream to web UI
- [ ] Implement touch camera controls
- [ ] Scene selector dropdown
- [ ] Test on iPhone Safari

### Medium-term (Next Month)
- [ ] Convert to Progressive Web App
- [ ] Add home screen install
- [ ] Implement WebSocket for real-time updates
- [ ] Optimize video streaming for mobile

### Long-term (Future)
- [ ] Native iOS app (if needed)
- [ ] Accelerometer controls (tilt to pan)
- [ ] ARKit integration (AR mode)
- [ ] Multi-user sync (collaborative viewing)

---

## Testing Checklist

### iPhone Compatibility

- [ ] Safari 15+ (iOS 15+)
- [ ] Touch gestures work
- [ ] Video stream loads
- [ ] WebSocket connects
- [ ] Fullscreen mode (PWA)
- [ ] Landscape orientation
- [ ] Works on cellular data (not just WiFi)

### Performance Targets

- [ ] < 100ms input latency
- [ ] 24+ fps video stream
- [ ] < 50 MB/min bandwidth
- [ ] Battery drain < 10%/hour
- [ ] No UI lag on swipes

---

## Example: Minimal Web UI for iPhone

```html
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
    <style>
        * { margin: 0; padding: 0; touch-action: manipulation; }
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; }
        #viewer { width: 100vw; height: 60vh; object-fit: contain; background: #000; }
        #controls { padding: 20px; }
        button {
            font-size: 18px;
            padding: 15px 30px;
            margin: 5px;
            border-radius: 10px;
            background: #007AFF;
            color: white;
            border: none;
        }
    </style>
</head>
<body>
    <img id="viewer" src="http://192.168.1.100:8100/api/camera/stream.mjpg" />

    <div id="controls">
        <select id="scene-select" style="font-size: 18px; padding: 10px;">
            <option value="empty">Empty</option>
            <option value="minimal">Minimal</option>
        </select>

        <button onclick="toggleFullscreen()">Fullscreen</button>
        <button onclick="resetCamera()">Reset Camera</button>
    </div>

    <script>
        const viewer = document.getElementById('viewer');
        const sceneSelect = document.getElementById('scene-select');

        // Scene switching
        sceneSelect.addEventListener('change', async (e) => {
            await fetch('http://192.168.1.100:8100/api/scene/load', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ scene_name: e.target.value })
            });
        });

        // Fullscreen
        function toggleFullscreen() {
            if (!document.fullscreenElement) {
                viewer.requestFullscreen();
            } else {
                document.exitFullscreen();
            }
        }

        // Touch controls for camera
        let touchStart = null;
        viewer.addEventListener('touchstart', (e) => {
            touchStart = { x: e.touches[0].clientX, y: e.touches[0].clientY };
        });

        viewer.addEventListener('touchend', async (e) => {
            if (!touchStart) return;
            const dx = e.changedTouches[0].clientX - touchStart.x;
            const dy = e.changedTouches[0].clientY - touchStart.y;

            await fetch('http://192.168.1.100:8100/api/camera/orbit', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ dx: dx / window.innerWidth, dy: dy / window.innerHeight })
            });

            touchStart = null;
        });
    </script>
</body>
</html>
```

---

## Summary

**Desktop (Now):**
- Use `desktop_viewer.py` for development and testing
- Fullscreen mode with F11
- Scene selection via command-line

**iPhone (Next Step):**
- Enhance web UI with 3D stream
- Add touch controls
- Make it a Progressive Web App

**Future:**
- Consider native iOS app only if web version has limitations
- Focus on web-first approach for maximum flexibility
- MuJoCo server can run anywhere (Mac, Pi, cloud)

**Key Insight:** Don't port the desktop app to iPhone. Build a better mobile-first interface that leverages web technologies while keeping MuJoCo on the server.

