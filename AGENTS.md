# Reachy Mini Development Guide for AI Agents

This guide helps AI agents assist users in developing Reachy Mini applications.

---

## Agent Behavior

### FIRST: Check for agents.local.md

**Before doing anything else**, search for `agents.local.md` in the current directory:

```
IF agents.local.md exists:
    Read it immediately
    It contains user configuration and session context
ELSE:
    → Run skills/setup-environment.md to set up the environment
```

This file stores the user's robot type, preferences, and setup status. Always check it first.

### Be a Teacher

Unless the user explicitly requests otherwise:
- Explain concepts as you go
- Encourage questions ("Let me know if you'd like more detail on any of this")
- Guide non-technical users through each step
- Don't assume prior knowledge

### Two App Flavours — Default to JS, Python for developers

Reachy Mini supports two app types. **Default to a JS (Live/Web) app** unless the user explicitly wants on-robot Python, or has a need that only Python can cover (heavy on-robot compute, rich hardware access, deterministic real-time control loops, or bundled offline LAN tooling).

| Flavour | Default for | Where it runs |
|---|---|---|
| **JS app (recommended)** | End-users, shareable-by-URL experiences, anyone who wants "open a link, use the robot". Remote launch, zero-install, streaming A/V UIs. | Static HF Space; reaches the robot over WebRTC via the central signaling server. |
| **Python app** | Developer tools, on-robot control loops, heavy motion sequencing, offline/LAN. | Robot owner's machine (laptop / CM4), optionally with a bundled web UI. |

Both coexist — a Python app can bundle a browser UI, and a JS app can call the Python daemon's REST API.

**When unsure, start JS.** If the user later discovers they need on-robot compute they can graduate to a Python app; the reverse migration is rarely needed.

Confirm the choice with the user up front, then jump to the corresponding section:
- JS path → [Live/Web/JS Apps](#livewebjs-apps)
- Python path → instructions below

---

**Python path (for developers):**

- **NEVER create app folders manually** — use `reachy-mini-app-assistant`.
- **If a command fails**, ask the user to run it in their terminal — don't attempt complex workarounds.

```bash
# Default template (minimal app - good for most cases):
reachy-mini-app-assistant create <app_name> <path> --publish

# Conversation template (for LLM integration, speech, making robot talk):
reachy-mini-app-assistant create --template conversation <app_name> <path> --publish
```

Python apps put web UIs in `static/`. See `skills/create-app.md` for details.

### Always Create plan.md Before Coding

Before implementing any app:
1. Create `plan.md` in the app directory
2. Write your understanding of what the user wants
3. List your technical approach
4. Ask clarifying questions and provide answer fields inside `plan.md`
5. Wait for answers before coding

### Keep Notes in agents.local.md

Use `agents.local.md` to store:
- User's robot type (Lite/Wireless)
- Environment preferences
- Useful context for future sessions
- Keep it concise

---

## Robot Basics

**Reachy Mini** is a small expressive robot:

| Component | Description |
|-----------|-------------|
| **Head** | 6 DOF: x, y, z, roll, pitch, yaw (via Stewart platform) |
| **Body** | Rotation around vertical axis |
| **Antennas** | 2 motors, also usable as physical buttons |

**Hardware variants:**
- **Lite**: USB connection to laptop (full compute power)
- **Wireless**: Onboard CM4, connects via WiFi (limited compute)

---

## SDK Essentials

### Connection

```python
from reachy_mini import ReachyMini

with ReachyMini() as mini:
    # Your code here
```

### Two Motion Methods

| Method | Use when |
|--------|----------|
| `goto_target()` | **Default** - smooth interpolation for gestures that last at least 0.5s each |
| `set_target()` | Real-time control loops (e.g. tracking) at 10Hz+ |

### Basic Example

See and run `examples/minimal_demo.py` - demonstrates connection, head motion, and antenna control.

### Before Writing Code

- Read `docs/source/SDK/python-sdk.md` for API overview
- Skim `src/reachy_mini/reachy_mini.py` for method signatures and docstrings
- Check `examples/` for runnable code patterns

---

## Live/Web/JS Apps

> ## START HERE: clone `webrtc_example` and modify it.
>
> **Live Space (run it, read the source):** https://huggingface.co/spaces/cduss/webrtc_example
>
> This is the **canonical reference implementation** 
>
 **Mimicking what is done in this example is the fastest path to a working app.**

Don't necessarily clone it verbatim — feel free to trim panels, remove features, and tweak the UI as needed. But the core patterns (SDK wiring, event handling, session management, media flow, responsive mobile layout) are already solved there; start by understanding how it works before scaffolding something new. **Default to a mobile-friendly responsive design** (most users open Spaces on phones) — see the section below.

Browser apps that control a Reachy Mini remotely over WebRTC. **The app IS a static Hugging Face Space**, not something installed on the robot. Any HF-authenticated user can open the Space URL from anywhere and reach any robot they have access to, through the central signaling server.

> **Full walkthrough**: [`docs/source/SDK/javascript-sdk.md`](docs/source/SDK/javascript-sdk.md) — architecture diagram, end-to-end deployment, complete SDK reference. The section below is a quick-start; read the full doc when scaffolding something non-trivial.

**What this flavour unlocks:**
- **Remote launch** — no LAN, no USB, no local daemon on the end-user side.
- **Zero-install sharing** — the Space URL *is* the product.
- **Off-robot compute** — work lives in the browser (static Space) or in the Space backend (Gradio/Docker variants). The robot stays a pure IO device.
- **Bidirectional media** — robot camera/mic → browser; optionally user's mic → robot speaker.
- **Clean version pinning** — each app imports the SDK from a specific GitHub tag; stable even as the SDK advances.

### Design for mobile first — most users will open the Space on a phone

Unless the user explicitly asks for a desktop-only / kiosk / dev-tool UI, **assume the app will be opened on a smartphone** and design accordingly. The Space URL is shareable, and "open this link on your phone and play with the robot" is the most common end-user flow.

Practical rules:
- Include `<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no" />` in `<head>` (the `webrtc_example` already does).
- Use fluid widths and `rem`/`vh`/`vw` units, not fixed pixel layouts. Stack panels vertically by default; reserve side-by-side layouts for `@media (min-width: 768px)`.
- Touch-friendly hit targets: ≥ 44×44 px buttons, sliders with comfortable thumbs.
- No hover-only affordances. Anything reachable by mouse-hover must also be reachable by tap.
- Test in Chrome devtools' phone emulation before declaring the app done.
- Bidirectional audio + camera permissions on iOS Safari are the most fragile path — verify on a real iPhone if the app uses them.

The `webrtc_example` is already responsive; cloning it gets you the mobile baseline for free. Don't undo it by hardcoding desktop widths in your edits.

### SDK import — strongly prefer an immutable ref (tag or commit SHA)

For anything you'd want to be stable, pin to either a `v*` **tag** (preferred for releases) or a 40-char **commit SHA** (when you need an unreleased SDK feature, like the live `webrtc_example` does). Both are immutable on jsDelivr. Pinning to a branch like `@main` works, but means the app can break on any SDK change — fine for early prototyping, risky for anything you share.

```html
<script type="module">
  // Tagged release — preferred
  import { ReachyMini } from "https://cdn.jsdelivr.net/gh/pollen-robotics/reachy_mini@v1.6.4/js/reachy-mini.js";

  // Or a specific commit SHA — for unreleased SDK changes
  // import { ReachyMini } from "https://cdn.jsdelivr.net/gh/pollen-robotics/reachy_mini@e8ba0b3dfcd48af528a8223eb2f2d58986204c94/js/reachy-mini.js";

  const robot = new ReachyMini({ appName: "my-app" });
</script>
```

Single ES module from jsDelivr; no build step, no npm.

**When scaffolding a new app**, resolve the latest `v*` tag and hardcode it:

```bash
git ls-remote --tags --refs --sort=-v:refname \
  https://github.com/pollen-robotics/reachy_mini.git \
  | head -1 | awk -F/ '{print $NF}'
```

**When extending an existing app**, re-check for a newer tag and bump only if the app needs it. Stale pins are fine. Branch pins (`@main`, etc.) work but trade stability for freshness — use them deliberately, not by default.

### Create the Space from scratch

> **Reminder**: don't actually scaffold "from scratch." Start by **copying [`../hfspace/webrtc_example/`](https://huggingface.co/spaces/cduss/webrtc_example)** (or the live Space) and trimming. The steps below are for the rare case where the example doesn't apply.

The app *is* the Space repo. Workflow:

**0. Preconditions**

```bash
hf --version           # pip install -U huggingface_hub if missing
hf auth whoami         # run `hf auth login` if not authenticated
git --version
```

**1. Gather** app name (slug, lowercase, underscores/hyphens), one-line title, feature list. Username comes from `hf auth whoami`.

**2. Write `plan.md` first** (same rule as Python apps). Wait for user approval before scaffolding.

**3. Scaffold three files locally** in `<app-name>/`:

- **`README.md`** — the Space YAML header is where `sdk`, OAuth, and discovery tags live (all required):

```yaml
---
title: <one-line title>
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
hf_oauth: true
hf_oauth_expiration_minutes: 480
tags:
  - reachy_mini
  - reachy_mini_js_app
---
```

- **`index.html`** — **copy from [`../hfspace/webrtc_example/index.html`](https://huggingface.co/spaces/cduss/webrtc_example)** and trim panels you don't need. Do not write from scratch.
- **`style.css`** — copy from the same example, then tweak.

**4. Create + push:**

```bash
hf repos create <app-name> --repo-type space --space-sdk static
git clone https://huggingface.co/spaces/<username>/<app-name>
cp <scaffold-dir>/{README.md,index.html,style.css} <app-name>/
cd <app-name> && git add . && git commit -m "Initial app scaffold" && git push
```

Static Spaces go live in ~10 s. **The live URL is `https://<username>-<app-name>.static.hf.space/`** — not `huggingface.co/spaces/<username>/<app-name>/`. The latter is the file browser; only the `.static.hf.space/` origin actually serves the app (and only that origin can complete the HF OAuth redirect). Return the `.static.hf.space/` URL to the user.

### SDK cheatsheet

State: `disconnected` → `connect()` → `connected` → `startSession(robotId)` → `streaming`.

Commands: `setHeadRpyDeg(r°,p°,y°)`, `setAntennasDeg(right°, left°)`, `setBodyYawDeg(yaw°)`, `setTarget({ head?: number[16], antennas?: [rRad, lRad], body_yaw?: rad })` (atomic raw-units update), `playSound(file)`, `setAudioMuted(bool)`, `setMicMuted(bool)`, `sendRaw(obj)`, `getVersion()`. Math utilities exported from the module: `rpyToMatrix`, `matrixToRpy`, `degToRad`, `radToDeg`.

Speaker / mic volume: `getVolume()` → 0-100, `setVolume(0-100)`, `getMicrophoneVolume()`, `setMicrophoneVolume(0-100)`. All return a `Promise`; `setVolume` resolves with the value the server actually applied (may be clamped/rounded).

Torque / wake: `setMotorMode("enabled"|"disabled"|"gravity_compensation")`, `wakeUp()`, `gotoSleep()`, `isAwake()`, `ensureAwake()`. `robot.robotState.motor_mode` reflects the live state.

Lifecycle: `authenticate()` / `login()` / `connect()` / `startSession()` / `stopSession()` / `disconnect()` / `logout()`. Video: `attachVideo(<video>)` returns a detach fn.

**Always call `await robot.ensureAwake()` right after `startSession()`** — otherwise if the robot is in the sleep pose (torque off, head on the base) every `setHeadRpyDeg` / `setAntennasDeg` is silently ignored and the app looks broken. `ensureAwake()` is a no-op when the robot is already awake (including gravity-compensation mode), so it's safe to call unconditionally at startup.

Events: `connected`, `disconnected`, `robotsChanged`, `streaming`, `sessionStopped`, `sessionRejected` (robot busy — inspect `e.detail.activeApp`), `state` (every ~500 ms), `videoTrack`, `micSupported`, `error`.

**Full API:** read the top ~90 lines of [`js/reachy-mini.js`](js/reachy-mini.js) — the file header is a complete reference.

### Media flow — use WebRTC, not `getUserMedia`, for robot media

The `<video>` element you pass to `attachVideo()` receives the **robot's** camera and microphone over WebRTC. Do **not** call `navigator.mediaDevices.getUserMedia()` to read robot media — that grabs the user's *own* laptop camera.

Bidirectional audio is automatic: with `enableMicrophone: true` (default), the SDK acquires the user's mic via `getUserMedia`, negotiates it into the peer connection, and exposes `setMicMuted()`. Check `micSupported` after session start before showing a mic UI.

Rule of thumb: robot IO flows through WebRTC; the user's own laptop IO flows through browser APIs; the SDK wires them together.

### Reference implementation

**`webrtc_example`** — canonical JS app: login, robot picker, video stream, head/antenna sliders, sound presets, latency overlay. ~500 lines.

- Live Space: https://huggingface.co/spaces/cduss/webrtc_example
- Source (in sibling repo): `../hfspace/webrtc_example/`

Always start from this when scaffolding anything non-trivial. Integration patterns (event wiring, slider debouncing, state sync, session-rejection handling) are already solved there.

### Local development (without pushing to HF every iteration) (advanced user)

OAuth only works from the deployed `*.static.hf.space` origin — clicking "Sign in with HF" on `localhost` or `file://` silently fails because HF auto-provisions the OAuth client against your Space's origin. HF Spaces **Dev Mode is not available for static Spaces**, so the manual-token loop below is the only low-latency option:

1. Create a read-only token at https://huggingface.co/settings/tokens.
2. In dev, skip `authenticate()` / `login()` and pass the token directly to `connect()`:
   ```js
   const token = new URLSearchParams(location.search).get("hf")
                 || localStorage.getItem("hf_dev_token");
   await robot.connect(token);    // SDK accepts a manually-pasted HF token
   ```
3. Serve with `python3 -m http.server 8765` (not `file://` — the SDK is an ES module from jsDelivr and needs a real HTTP origin). Open `http://localhost:8765/?hf=hf_…` once, then rely on `localStorage`.
4. Before pushing to the Space, restore the OAuth path and **never commit the token**.

Rule of thumb: for UI-heavy iteration (CSS, slider behaviour, state machine wiring), local reload ≪ push-to-HF (~10-30 s). For a one-off tweak, just push.

### Common gotchas

| Symptom | Cause |
|---|---|
| `robot.login()` fails / redirects to `about:blank` | Running via `file://` or localhost — OAuth only works on the live Space domain. |
| Head doesn't move | `setHeadRpyDeg` called before `startSession()` resolved. Check the return value. |
| Head doesn't move, session *is* streaming | Robot is asleep (torque off). Call `await robot.ensureAwake()` after `startSession()`. |
| Audio stays silent after `setAudioMuted(false)` | Browser requires unmute inside a user-gesture handler. |
| `sessionRejected` | Robot is locked by another app. Surface `e.detail.activeApp` in the UI. |
| Stream laggy | See buffer-lag overlay in `webrtc_example/index.html`; > 500 ms jitter = network issue. |
| UI broken on phone | Hardcoded pixel widths or desktop-only layout. Use the `webrtc_example` viewport meta + fluid units; test in Chrome devtools phone emulation. |
| Volume slider floods the data channel | Wire `setVolume` on the slider's `'change'` event (release) — not `'input'` (per-pixel). Sync once on streaming-start with `await robot.getVolume()` to reflect the robot's real level, then reflect the value that `setVolume` resolves with back into the slider. |

---

## REST API

The daemon exposes an HTTP/WebSocket API at `http://{daemon-ip}:8000/api`.

> REST and the JS SDK's WebRTC data channel are **sibling transports** into the same `process_command()` backend on the daemon — WebRTC is a JSON subset (motion, audio, state). Commands you send from JS (`setHeadRpyDeg`, `setAntennasDeg`, …) reach the same handler as the corresponding REST endpoints; picking one transport or the other is a deployment choice, not a functional one.

- **Lite**: `localhost:8000` (daemon runs on your machine)
- **Wireless**: `reachy-mini.local:8000` or the robot's IP address

**Use REST API for:** Web UIs, non-Python clients, remote control, AI/LLM integration via HTTP. => Note: for the app to be discoverable, it must be a python app for now, this will change in a future release.

**Interactive docs:** `http://{daemon-ip}:8000/docs` (when daemon is running)

See `skills/rest-api.md` for details.

---

## Platform Compatibility

| Setup | Compute | Camera | Notes |
|-------|---------|--------|-------|
| **Lite** | Full (laptop) | Direct USB | Most flexible, best for dev |
| **Wireless (local)** | Limited (CM4) | Direct | Memory/CPU constrained |
| **Wireless (streamed)** | Full (laptop) | Via network | Some tracking quality loss |
| **Simulation** | Full | N/A | Can't test camera features |

---

## Safety Limits

| Joint | Range |
|-------|-------|
| Head pitch/roll | [-40, +40] degrees |
| Head yaw | [-180, +180] degrees |
| Body yaw | [-160, +160] degrees |
| Yaw delta (head - body) | Max 65° difference |

Gentle collisions with body are safe. SDK clamps values automatically.

For coordinate systems and architecture details, see `docs/source/SDK/core-concept.md`.

---

## Example Apps

| App | Key Patterns | Source |
|-----|--------------|--------|
| **reachy_mini_conversation_app** | AI integration, control loops, LLM tools | [GitHub](https://github.com/pollen-robotics/reachy_mini_conversation_app) |
| **marionette** | Recording motion, safe torque, HF dataset | [HF Space](https://huggingface.co/spaces/RemiFabre/marionette) |
| **fire_nation_attacked** | Head-as-controller, leaderboards, games | [HF Space](https://huggingface.co/spaces/RemiFabre/fire_nation_attacked) |
| **spaceship_game** | Head-as-joystick, antenna buttons | [HF Space](https://huggingface.co/spaces/apirrone/spaceship_game) |
| **reachy_mini_radio** | Antenna interaction pattern | [HF Space](https://huggingface.co/spaces/pollen-robotics/reachy_mini_radio) |
| **reachy_mini_simon** | No-GUI pattern (antenna to start) | [HF Space](https://huggingface.co/spaces/apirrone/reachy_mini_simon) |
| **hand_tracker_v2** | Camera-based control loop | [HF Space](https://huggingface.co/spaces/pollen-robotics/hand_tracker_v2) |
| **reachy_mini_dances_library** | Symbolic motion definition | [GitHub](https://github.com/pollen-robotics/reachy_mini_dances_library) |

---

## Documentation

Full SDK documentation is in `docs/source/`:

| Topic | File |
|-------|------|
| Quickstart | `docs/source/SDK/quickstart.md` |
| Python SDK | `docs/source/SDK/python-sdk.md` |
| **JavaScript SDK & Web Apps** | `docs/source/SDK/javascript-sdk.md` |
| Core concepts | `docs/source/SDK/core-concept.md` |
| Media architecture (WebRTC / GStreamer) | `docs/source/SDK/media-architecture.md` |
| AI integration | `docs/source/SDK/integration.md` |
| Troubleshooting | `docs/source/troubleshooting.md` |

For platform-specific guides (Lite, Wireless, Simulation), see `docs/source/platforms/`.

---

## Skills Reference

Read these files in `skills/` when you need detailed knowledge:

| Skill | When to use |
|-------|-------------|
| **setup-environment.md** | First session, no `agents.local.md` exists |
| **create-app.md** | Creating a new app with `reachy-mini-app-assistant` |
| **control-loops.md** | Building real-time reactive apps (tracking, games) |
| **motion-philosophy.md** | Choosing between `goto_target` and `set_target` |
| **safe-torque.md** | Enabling/disabling motors without jerky motion |
| **ai-integration.md** | Building LLM-powered apps |
| **symbolic-motion.md** | Defining motion mathematically (dances, rhythms) |
| **interaction-patterns.md** | Using antennas as buttons, head as controller |
| **debugging.md** | App crashes, connectivity issues, basic checks |
| **testing-apps.md** | Testing before delivery (sim vs physical) |
| **rest-api.md** | HTTP/WebSocket API for non-Python clients |
| **deep-dive-docs.md** | When to read full SDK documentation |

---

## Quick Reference

**Motor names:** `body_rotation`, `stewart_1-6`, `right_antenna`, `left_antenna`

**Interpolation methods:** `linear`, `minjerk` (default), `ease_in_out`, `cartoon`

**Emotions library:**
```python
from reachy_mini.motion.recorded_move import RecordedMoves
moves = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
mini.play_move(moves.get("happy"), initial_goto_duration=1.0)
```

---

## Community

- **App guide**: https://huggingface.co/blog/pollen-robotics/make-and-publish-your-reachy-mini-apps
- **Source code**: https://github.com/pollen-robotics/reachy_mini
- **Community apps**: https://huggingface.co/spaces?q=reachy_mini
- **Discord**: https://discord.gg/Y7FgMqHsub
