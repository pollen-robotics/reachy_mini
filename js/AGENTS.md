# Reachy Mini JS App Development Guide for AI Agents

This guide is for building **browser-based apps** that control a Reachy Mini robot live over WebRTC — no Python install, no local daemon on the user's machine. The user's browser talks to a central signaling server (on Hugging Face) which brokers a WebRTC connection to the robot running at the owner's home/office.

If the user wants a Python app running on/next to the robot, stop and read `../AGENTS.md` instead.

---

## What this lets you build

- **Dashboards & remote controls** — anyone with an HF account can log in and drive the robot.
- **Live interactive experiences** — receive the robot's camera stream, push head/antenna commands back, play sounds, stream microphone audio.
- **Zero-install web apps** — published as a static Hugging Face Space, loads the SDK from a CDN.

Typical end-user flow: open the Space in a browser → "Sign in with Hugging Face" → pick a robot from the live list → start streaming.

---

## The SDK in one import

```html
<script type="module">
  import { ReachyMini } from "https://cdn.jsdelivr.net/gh/pollen-robotics/reachy_mini@main/js/reachy-mini.js";
  const robot = new ReachyMini();
</script>
```

No build step, no npm. The SDK is a single ES module served from jsDelivr (backed by the `pollen-robotics/reachy_mini` GitHub repo). Pin to a branch/tag by replacing `@main` with `@v1.5.1` or a specific commit SHA.

The SDK's only runtime dep is `@huggingface/hub` for OAuth, loaded from its own CDN inside the module.

---

## The starter example — fork this

The canonical reference implementation:

- **Live Space:** https://huggingface.co/spaces/cduss/webrtc_example
- **Source (locally, sibling repo):** `../../hfspace/webrtc_example/` → `index.html`, `style.css`, `README.md`

~500 lines total — login, robot picker, video stream, head/antenna sliders, sound presets, latency overlay. The integration patterns (event wiring, slider debouncing, state sync) are already solved there. Copy it and delete what you don't need.

Always build a fresh Space from scratch using the playbook below — don't ask the user to click around the HF UI.

---

## State machine

```
'disconnected' ──connect()──▸ 'connected' ──startSession(robotId)──▸ 'streaming'
      ▴ disconnect()                  ▴ stopSession()
      └────────────────────────────────┘
```

- **disconnected** — no SSE, no WebRTC. Must call `authenticate()` or `login()` first.
- **connected** — SSE to signaling server open, `robotsChanged` events arriving. Not yet talking to a robot.
- **streaming** — WebRTC session active with one robot. Commands go through, video/audio flow.

`stopSession()` returns to `connected`; `disconnect()` returns to `disconnected` but keeps auth. `logout()` also clears HF credentials.

---

## Minimum working app (copy-paste)

```html
<!doctype html>
<video id="v" autoplay playsinline muted></video>
<button onclick="login()">Login</button>
<button onclick="start()">Connect & Stream</button>

<script type="module">
  import { ReachyMini } from "https://cdn.jsdelivr.net/gh/pollen-robotics/reachy_mini@main/js/reachy-mini.js";
  const robot = new ReachyMini({ appName: "my-demo" });
  window.login = () => robot.login();

  window.start = async () => {
    if (!(await robot.authenticate())) return robot.login();
    robot.attachVideo(document.getElementById('v'));
    await robot.connect();
    robot.addEventListener('robotsChanged', async (e) => {
      const first = e.detail.robots[0];
      if (first) await robot.startSession(first.id);
    });
  };

  // Drive the robot
  setInterval(() => robot.setHeadPose(0, 10*Math.sin(Date.now()/1000), 0), 100);
</script>
```

On first load the user sees a login button. After HF OAuth redirect, `authenticate()` returns true and the stream starts.

---

## API surface

### Constructor

```js
new ReachyMini({
  signalingUrl:     string,   // default "https://cduss-reachy-mini-central.hf.space"
  enableMicrophone: boolean,  // default true — prompt for mic permission during startSession
  clientId:         string,   // optional HF OAuth client id (uses Space default otherwise)
  appName:          string,   // default "unknown" — shown on the robot dashboard
})
```

### Commands (fire-and-forget over data channel — return `true` if sent)

| Method | Units / notes |
|---|---|
| `setHeadPose(roll, pitch, yaw)` | degrees — use for real-time sliders / loops (10 Hz+) |
| `setAntennas(right, left)` | degrees, range ~[-175, +175] |
| `playSound(file)` | filename that exists on the robot (e.g. `"wake_up.wav"`) |
| `setAudioMuted(bool)` | mute robot → your speaker (default muted, browser requires gesture) |
| `setMicMuted(bool)` | mute your mic → robot speaker (requires `micSupported`) |
| `sendRaw(obj)` | escape hatch — arbitrary JSON over the data channel |

### Async methods

| Method | Returns |
|---|---|
| `authenticate()` | `Promise<bool>` — true if valid HF token found |
| `login()` | redirects to HF OAuth page |
| `logout()` | clears token + disconnects |
| `connect(token?)` | resolves when SSE welcomed, robot list starts arriving |
| `startSession(robotId)` | resolves when video + data channel both ready |
| `stopSession()` | returns to `connected` |
| `disconnect()` | returns to `disconnected` |
| `getVersion()` | daemon version string (or `null`) |
| `attachVideo(<video>)` | returns cleanup function; binds srcObject when track arrives |

### Read-only properties

`state`, `robots`, `robotState` (live head/antennas in degrees), `username`, `isAuthenticated`, `micSupported`, `micMuted`, `audioMuted`

### Events (`EventTarget` — use `addEventListener`)

| Event | `e.detail` |
|---|---|
| `connected` | `{ peerId }` |
| `disconnected` | `{ reason }` |
| `robotsChanged` | `{ robots: [{ id, meta: { name } }, ...] }` |
| `streaming` | `{ sessionId, robotId }` |
| `sessionStopped` | `{ reason }` |
| `sessionRejected` | `{ reason, activeApp }` — robot is locked by someone else |
| `state` | `{ head: {roll,pitch,yaw}, antennas: {right,left} }` — every ~500 ms while streaming |
| `videoTrack` | `{ track, stream }` — usually handled by `attachVideo()` |
| `micSupported` | `{ supported }` — emitted after SDP negotiation |
| `error` | `{ source: "signaling"\|"webrtc"\|"robot", error }` |

### Exported helpers

`rpyToMatrix(r, p, y)` / `matrixToRpy(m)` / `degToRad(x)` / `radToDeg(x)` — for apps that need to pre-compute poses.

---

## Creating the Space from scratch — the full playbook

Reachy Mini JS apps live **on Hugging Face**, not on GitHub. The app IS the Space repo. When the user says *"create me an app that does X"*, the end-to-end flow is:

### 0. Preconditions (check once, at the start of the session)

```bash
hf --version                    # needs the `hf` CLI (pip install -U huggingface_hub, or brew install huggingface-cli)
hf auth whoami                  # must print a username; if not, run `hf auth login`
git --version                   # needs git
```

If `hf` is missing, ask the user to `pip install -U huggingface_hub` in their terminal and come back — don't try clever workarounds.

### 1. Gather the inputs

Ask the user (or read from `agents.local.md`):
- **App name** — slug for the Space, e.g. `reachy_dance_controller`. Lowercase, underscores/hyphens, no spaces.
- **One-line description** — becomes the Space title.
- **What the app should do** — feature list you'll turn into `plan.md`.

HF username comes from `hf auth whoami`. Don't ask twice.

### 2. Write `plan.md` first

Before any code, write `plan.md` in the working dir with:
- Restatement of what the app does
- Which SDK methods you'll call (`setHeadPose`, `attachVideo`, sound presets, etc.)
- UI layout (panels / controls)
- Open questions with answer fields

Wait for the user to approve `plan.md` before scaffolding.

### 3. Scaffold locally

Create a working directory `<app-name>/` containing three files:

**`README.md`** — the Space configuration, auth, and tags all live here:

```yaml
---
title: <one-line description>
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

# <App name>

<what it does, how to use it>
```

Required YAML fields and why:
| Field | Why |
|---|---|
| `sdk: static` | Plain HTML/JS/CSS, no build, no runtime. |
| `hf_oauth: true` | Enables `robot.login()` / `robot.authenticate()`. Without this, auth redirects fail. |
| `tags: [reachy_mini, reachy_mini_js_app]` | Makes the Space discoverable in the Reachy Mini app browser. |

**`index.html`** — start from the minimum working app above, or copy `../../hfspace/webrtc_example/index.html` and strip panels you don't need.

**`style.css`** — copy from `webrtc_example/style.css` and tweak, or write from scratch if the UI is trivial.

### 4. Create the empty Space on Hugging Face

```bash
hf repos create <app-name> --repo-type space --space-sdk static
```

This creates `https://huggingface.co/spaces/<username>/<app-name>` with an empty repo and default README.

### 5. Push the code

```bash
git clone https://huggingface.co/spaces/<username>/<app-name>
cp <working-dir>/{README.md,index.html,style.css} <app-name>/
cd <app-name>
git add README.md index.html style.css
git commit -m "Initial app scaffold"
git push
```

The Space rebuilds automatically on push. Status turns "Running" within ~10s for static Spaces.

### 6. Report back

Give the user the Space URL (`https://huggingface.co/spaces/<username>/<app-name>`) and remind them that OAuth only works on the live Space domain — testing locally via `python -m http.server` **will** break the HF login flow.

### Iterating after first push

Edit files locally, `git add / commit / push`. The Space redeploys on push. For a tighter feedback loop, use the "Files" tab in the Space UI to edit `index.html` in-browser (changes hot-reload).


---

## Agent behaviour for JS apps

### Before writing code

1. Confirm the user actually wants a **web** app — if they want to run Python locally, redirect to `../AGENTS.md` + `reachy-mini-app-assistant`.
2. Run the preconditions check (`hf --version`, `hf auth whoami`, `git --version`). Stop and ask the user to fix their environment before going further.
3. Check for `agents.local.md` one level up (project notes, robot variant, HF username, etc.).
4. Write `plan.md` next to the future `index.html` — describe what the app does, UI layout, which SDK methods you'll call, and any open questions. Wait for the user to answer before scaffolding.

### While writing code

- **Start from `webrtc_example/`**, don't build from scratch. Delete the panels you don't need.
- **Use `goto_target` semantics in your head? No** — the JS SDK only exposes `set_target` (real-time). For smooth prerecorded gestures you'd need a Python app.
- **Throttle sliders.** `setHeadPose` at every `input` event is fine (tiny payloads, ~60 Hz tops), but don't fire in a `requestAnimationFrame` loop unless you need to.
- **Listen for `sessionRejected`** — robot may be locked by another app. Show the `activeApp` name.
- **Keep the UI usable pre-auth.** The login button must work before `authenticate()` has resolved.
- **Respect browser audio rules.** Audio starts muted; only unmute inside a click handler.
- **No new abstractions for "what if the user has two robots"** — pick the first from `robotsChanged` unless the user asked for a picker.

### Testing

You can't run this without a live robot or at least a browser. Flow:
1. Open the HTML file via any static server (`python -m http.server`, `npx serve`, etc.) — **not** `file://` (OAuth will refuse).
2. Register the local URL as an OAuth redirect in your HF Space settings, or deploy to the Space early and iterate there.
3. For local dev without a robot, verify the flow up to `robotsChanged` with an empty list. Real hardware tests require a robot running the daemon and connected to the central signaling server.

---

## Common gotchas

| Symptom | Cause |
|---|---|
| `robot.login()` redirects to `about:blank` | Loaded via `file://` — use an http:// origin. |
| "No token — call authenticate() first" on connect | Tab reload lost the redirect params and sessionStorage — call `authenticate()` before `connect()`. |
| Audio stays silent even after `setAudioMuted(false)` | Browser requires the unmute to happen inside a user gesture (click handler). |
| `startSession` hangs | Robot may be busy / locked — wire a `sessionRejected` listener, and/or add a timeout on your side. |
| Head doesn't move | Data channel not open yet. `startSession()` already waits for both ICE and DC, but if you call `setHeadPose` before it resolves, the command is dropped (`returns false`). |
| Stream is laggy | Check the buffer lag overlay pattern in `webrtc_example` — jitter buffer > 500 ms means network trouble. |

---

## Quick reference

**Canonical SDK import:**
`https://cdn.jsdelivr.net/gh/pollen-robotics/reachy_mini@main/js/reachy-mini.js`

**Default signaling server:**
`https://cduss-reachy-mini-central.hf.space`

**Head joint ranges (same as Python SDK):**
roll/pitch ±40°, yaw ±180° (but body yaw delta capped at 65°). Antennas ±175°. The SDK does not clamp — the daemon does.

**Source:** [`js/reachy-mini.js`](reachy-mini.js) — single file, ~900 lines, read the top 90 lines for the full API reference.
