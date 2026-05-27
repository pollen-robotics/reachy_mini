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

> ## START HERE: clone one of the three reference apps under `pollen-robotics/*` on Hugging Face and trim.
>
> | Reference app | Stack | Use it for |
> |---|---|---|
> | [`pollen-robotics/reachy_mini_minimal_conversation`](https://huggingface.co/spaces/pollen-robotics/reachy_mini_minimal_conversation) | **Vanilla TS + Vite** | Smallest runtime, zero framework. |
> | [`pollen-robotics/reachy_mini_emotions`](https://huggingface.co/spaces/pollen-robotics/reachy_mini_emotions) | React 19 + MUI 7 + Vite | UI-rich apps (rich components, theming, deep links). |
> | [`pollen-robotics/reachy_mini_telepresence`](https://huggingface.co/spaces/pollen-robotics/reachy_mini_telepresence) | React 19 + MUI 7 + Vite | Camera / media-stream apps. |
>
> These are kept in lockstep with every SDK release. **Mimicking them is the fastest path to a working app.**

Browser apps that drive a Reachy Mini over WebRTC, deployed as **Hugging Face Spaces with `sdk: static`** (a Vite build pushed to the Space's CDN). Any HF-authenticated user can open the Space URL from anywhere and reach any robot they have access to, through the central signaling server.

> ⚠️ **`sdk: static` is the recommended default.** Static Spaces scale better (cold-start free, served from HF's CDN, no container costs) and HF replaces `__OAUTH_CLIENT_ID__` at file-serve time, so you don't need a Docker entrypoint. The three reference apps below currently still ship a `Dockerfile`; you can drop it on your fork. A short `sdk: docker` fallback is documented at the end of this section for cases where you can't run the Vite build locally.

> **Full walkthroughs**: [`ts/host/APP_AUTHOR_GUIDE.md`](ts/host/APP_AUTHOR_GUIDE.md) (app-author recipe), [`ts/host/SPEC.md`](ts/host/SPEC.md) (host ↔ embed contract + invariants), [`docs/source/SDK/javascript-sdk.md`](docs/source/SDK/javascript-sdk.md) (SDK + media architecture). The section below is the quick-start; read those before scaffolding anything non-trivial.

**What this flavour unlocks:**
- **Remote launch** — no LAN, no USB, no local daemon on the end-user side.
- **Zero-install sharing** — the Space URL *is* the product.
- **Off-robot compute** — work lives in the browser or the Space backend. The robot stays a pure IO device.
- **Bidirectional media** — robot camera/mic → browser; optionally user's mic → robot speaker.
- **Free OAuth + robot picker + top bar + leave flow** via the host shell (see next section).

### What the host shell does for you

The npm package [`@pollen-robotics/reachy-mini-sdk`](https://www.npmjs.com/package/@pollen-robotics/reachy-mini-sdk) exposes two subpath entries (`./host/auto` for the shell, `./host/embed` for the iframe) that, together, take care of:

- **Hugging Face OAuth** — sign-in screen, redirect, token storage, sign-out menu.
- **Robot discovery + picker** — live online / offline / busy list, click-to-pick.
- **Connecting overlay** — 3-step "Connecting / Starting session / Waking up" view.
- **Top bar** — app icon, app name, robot status, "End session", OAuth menu.
- **Dark / light theme** — from `prefers-color-scheme` or HF settings, propagated to the iframe.
- **Teardown** — `onLeave` callbacks fire reliably on user "End session", `requestLeave()`, or page unload.

You only write the app's UI. **Use any framework you want inside the iframe** (React, Svelte, Vue, vanilla TS) — the host doesn't care. Tech freedom is a [hard design rule, not an accident](ts/host/SPEC.md#tech-freedom-is-a-core-design-principle). **The host shell is always there, even in static deployments** — it owns OAuth + picker + top bar in every flavour of the app.

### App-author contract

Every Reachy Mini app ships exactly **three source files** (+ a `public/icon.svg`, a `package.json` for Vite, and the Space `README.md` frontmatter — see below):

| File | Purpose |
|---|---|
| `index.html` | Theme bootstrap + `<link rel="icon">` + `<script src=dispatch>` |
| `src/dispatch.ts` | Two-branch boot (host shell vs embed) |
| `src/embed.{ts,tsx}` | The app itself; calls `connectToHost()` once |

A 4th file — `Dockerfile` — is only needed if you fall back to `sdk: docker` (see the "Optional: docker fallback" subsection below). For `sdk: static`, you build locally with Vite and push the resulting `dist/` to the Space.

#### `index.html`

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>My App</title>
    <link rel="icon" href="/icon.svg" type="image/svg+xml" />

    <!-- Theme bootstrap: paint the right palette before CSS lands -->
    <script>
      (function () {
        var params = new URLSearchParams(window.location.search);
        var raw = params.get("theme");
        var mode = (raw === "dark" || raw === "light") ? raw
          : (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
        document.documentElement.setAttribute("data-theme", mode);
      })();
    </script>

    <!-- HF Spaces substitutes __OAUTH_CLIENT_ID__ (file-serve time in
         static Spaces, container-start time in docker Spaces) when
         `hf_oauth: true` is set in the README frontmatter. -->
    <script>
      (function () {
        var clientId = "__OAUTH_CLIENT_ID__";
        if (clientId && clientId.indexOf("__") !== 0) {
          window.huggingface = { variables: { OAUTH_CLIENT_ID: clientId } };
        }
      })();
    </script>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/dispatch.ts"></script>
  </body>
</html>
```

#### `src/dispatch.ts`

```ts
const params = new URLSearchParams(window.location.search);

if (params.get("embedded") === "1") {
  void import("./embed");
} else {
  void import("@pollen-robotics/reachy-mini-sdk/host/auto").then(({ mountHost }) => {
    mountHost({
      appName: "My App",
      appIconUrl: "/icon.svg",
      enableMicrophone: false,
    });
  });
}
```

#### `src/embed.ts`

```ts
import { connectToHost } from "@pollen-robotics/reachy-mini-sdk/host/embed";

const handle = await connectToHost();
const { reachy, theme, config, onLeave } = handle;

reachy.setHeadRpyDeg(0, 10, 0);

onLeave(async () => {
  // ~1.5–2 s budget. Cancel streams, release audio, flush telemetry.
});
```

By the time `connectToHost()` resolves: SDK loaded, session started, robot awake. **Don't reimplement `connect()` / `startSession()` / `ensureAwake()`** — let the host do it.

### `public/icon.svg` — the single app-identity asset

**One SVG file** powers three independent surfaces:

| Surface | What it shows | How it gets there |
|---|---|---|
| Host top bar (inside iframe) | App logo | You pass `appIconUrl: "/icon.svg"` to `mountHost()` |
| Hugging Face mobile catalog | App tile icon | The catalog API lists the Space's `siblings` and matches the repo path `public/icon.svg` (or `public/icon.png`) |
| Browser tab / OS | Favicon | `<link rel="icon" href="/icon.svg" type="image/svg+xml" />` in `index.html` |

Ship one **`public/icon.svg`** in your repo. Vite copies it verbatim to `dist/icon.svg`, so it's served at `https://<space>.hf.space/icon.svg`. **PNG fallback** (`public/icon.png`) is accepted; SVG wins when both exist. Without an icon, the mobile catalog falls back to the frontmatter `emoji` — pass the same emoji to `mountHost({ appEmoji: "🤖" })` so the top-bar fallback matches.

Design hint: square `viewBox`, readable at 16 px, inline all colours, target ≤ 30 KB. Full guidance: [`ts/host/APP_AUTHOR_GUIDE.md` §6](ts/host/APP_AUTHOR_GUIDE.md#6-visual-identity-icon-name-emoji).

### HF Space frontmatter

```yaml
---
title: <one-line title>
emoji: 🤖
colorFrom: yellow
colorTo: red
sdk: static
pinned: false
hf_oauth: true
short_description: One-line description shown in the mobile catalog.
tags:
  - reachy_mini
  - reachy_mini_js_app   # mandatory: mobile-catalog discovery filters on this exact string
---
```

- `sdk: static` is the recommended default. HF serves whatever you push as the Space root, replaces `__OAUTH_CLIENT_ID__` placeholders in HTML at file-serve time when `hf_oauth: true` is set, and gives you free CDN caching.
- The **`reachy_mini_js_app` tag is mandatory** to appear in the mobile catalog. The catalog API filters on this exact string.
- `hf_oauth: true` makes HF auto-provision an OAuth client and inject its ID into the served `index.html`.
- Apps published under the `pollen-robotics/*` namespace are automatically tagged as "official" in the catalog ([SPEC §1: Official vs community apps](ts/host/SPEC.md#official-vs-community-apps)); no extra config.

### Design for mobile first — most users will open the Space on a phone

Unless the user explicitly asks for a desktop-only / kiosk / dev-tool UI, **assume the app will be opened on a smartphone**. The Space URL is shareable, and "open this link on your phone and play with the robot" is the most common end-user flow.

Practical rules:
- Include `<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no" />` in `<head>`.
- Use fluid widths and `rem` / `vh` / `vw` units, not fixed pixel layouts. Stack panels vertically by default; reserve side-by-side layouts for `@media (min-width: 768px)`.
- Touch-friendly hit targets: ≥ 44×44 px buttons, sliders with comfortable thumbs.
- No hover-only affordances. Anything reachable by mouse-hover must also be reachable by tap.
- Test in Chrome devtools' phone emulation before declaring the app done.
- Bidirectional audio + camera permissions on iOS Safari are the most fragile path — verify on a real iPhone if the app uses them.

The reference apps are already responsive; cloning them gets you the mobile baseline for free. Don't undo it by hardcoding desktop widths in your edits.

### Create + deploy the Space

```bash
hf --version           # pip install -U huggingface_hub if missing
hf auth whoami         # run `hf auth login` if not authenticated
git --version
node --version         # need a recent Node (≥ 20) for the Vite build
```

1. **Pick a reference app** (`reachy_mini_minimal_conversation` for vanilla TS, `reachy_mini_emotions` for React + MUI, `reachy_mini_telepresence` for media) and clone its repo locally. The reference apps currently ship a `Dockerfile` — for the static path, you'll drop it.
2. **Write `plan.md` first** (same rule as Python apps). Wait for user approval before scaffolding.
3. Customise: `package.json` `name`, README frontmatter (`title`, `emoji`, `short_description`, **`sdk: static`** — drop `sdk: docker` and `app_port` if you cloned a docker-based reference), `public/icon.svg`, `src/embed.{ts,tsx}` (your app code). You can also delete the cloned `Dockerfile` and `docker-entrypoint.d/` folder.
4. Build locally:

```bash
npm install
npm run build      # produces ./dist/ — the artifact you'll ship to HF
```

5. Create the Space and push the built `dist/` (plus `README.md` and a copy of `public/`) as the Space root:

```bash
hf repos create <app-name> --repo-type space --space-sdk static
git clone https://huggingface.co/spaces/<username>/<app-name> ../<app-name>-space

# Space frontmatter
cp README.md ../<app-name>-space/

# Built app (served by HF's static CDN at the Space root)
cp -R dist/. ../<app-name>-space/

# Source-side public/ duplicated at the repo path the mobile catalog API matches.
# Vite already copied icon.svg into dist/, so it's served at /icon.svg for the
# top bar + favicon; this second copy lives only so `siblings` listing matches
# `public/icon.svg` for the catalog probe.
mkdir -p ../<app-name>-space/public
cp public/icon.svg ../<app-name>-space/public/
# (also cp public/icon.png if you ship a raster fallback)

cd ../<app-name>-space
git add -A && git commit -m "Initial deploy" && git push
```

The contract: **only built artifacts + `README.md` + `public/icon.svg` live in the HF Space repo**. Source stays in your dev folder (or a separate GitHub repo) and you re-run `npm run build && cp -R dist/.` for each deploy. Tip: most teams wire this into a GitHub Action that builds on push and mirrors to HF — pick any HF "sync" action that runs `npm ci && npm run build` and force-pushes `dist/` + `README.md` + `public/` to the Space's `main`.

**The live URL is `https://<username>-<app-name>.static.hf.space/`** (not `huggingface.co/spaces/<username>/<app-name>/`, which is the file browser). Return the `.static.hf.space/` URL to the user.

**Sanity check OAuth substitution**: open the live URL, view source, and search for `__OAUTH_CLIENT_ID__`. If the placeholder is still there, HF didn't substitute it — verify `hf_oauth: true` is in your frontmatter and that you didn't accidentally minify the placeholder out of `index.html`. If you're stuck, fall back to `sdk: docker` (next subsection).

**Cache busting**: if a push doesn't update the live bundle, push an empty commit (`git commit --allow-empty -m "chore: bust cache" && git push`).

### Optional: `sdk: docker` fallback

Use this when you can't run the Vite build locally (e.g. CI-less environment, agent without Node, deployment from a tablet, etc.) — let HF build the Dockerfile for you instead.

1. Keep the `Dockerfile` + `docker-entrypoint.d/10-inject-hf-vars.sh` you cloned from the reference app.
2. Restore `sdk: docker` and `app_port: 7860` in the README frontmatter.
3. Push the **source** (not the build) to a `--space-sdk docker` Space:
   ```bash
   hf repos create <app-name> --repo-type space --space-sdk docker
   git remote add space git@hf.co:spaces/<username>/<app-name>
   git push space main
   ```

HF Spaces builds the `Dockerfile`, runs the entrypoint to inject HF vars, and serves nginx on `app_port: 7860`. The live URL is `https://<username>-<app-name>.hf.space/`.

Trade-off vs. static: the Space spins up a container on each cold start, costs more, and is slower to deploy. Prefer static whenever the build can run upstream.

### `mountHost()` and `connectToHost()` — API quick reference

```ts
import { mountHost } from "@pollen-robotics/reachy-mini-sdk/host/auto";

mountHost({
  appName: "My App",        // REQUIRED: passed to the SDK + shown in top bar
  appIconUrl: "/icon.svg",  // optional: top-bar logo
  appEmoji: "🤖",           // optional: fallback when no icon.svg
  enableMicrophone: false,  // false unless your app uses user-mic input
  clientId: undefined,      // optional; defaults to window.huggingface.variables.OAUTH_CLIENT_ID
  devToken: undefined,      // optional: { token, userName } — dev shortcut, see local dev below
});
```

```ts
import { connectToHost } from "@pollen-robotics/reachy-mini-sdk/host/embed";

interface MyConfig { startingEmotion?: string; }

const handle = await connectToHost<MyConfig>();
const { reachy, theme, config, onLeave, onThemeChange, onConfigChange,
        setAppState, requestLeave, reportError } = handle;
```

By the time `connectToHost()` resolves, `reachy` is a live SDK instance with a session running and the robot awake. **`config` is `unknown` at runtime** — cast is not enough, validate it (an attacker controlling the URL can shape config freely).

The host owns all teardown: don't call `reachy.stopSession()` yourself, register an `onLeave` callback instead. Full surface: [`ts/host/APP_AUTHOR_GUIDE.md` §4-§5](ts/host/APP_AUTHOR_GUIDE.md#4-mounthost-api).

### SDK cheatsheet (motion + media + daemon-side playback)

The `reachy` instance you get from `connectToHost()` is the same `ReachyMini` object app authors used to instantiate by hand. Once connected, you drive it the same way regardless of which boot path you came from.

Commands: `setHeadRpyDeg(r°, p°, y°)`, `setAntennasDeg(right°, left°)`, `setBodyYawDeg(yaw°)`, `setTarget({ head?: number[16], antennas?: [rRad, lRad], body_yaw?: rad })` (atomic raw-units update), `playSound(file)`, `setAudioMuted(bool)`, `setMicMuted(bool)`, `sendRaw(obj)`, `getVersion()`. Math utilities exported from the module: `rpyToMatrix`, `matrixToRpy`, `degToRad`, `radToDeg`.

Recorded-move playback (daemon-side, single-clock A/V sync): `playMove(motion, { audioBlob?, audioLeadMs? })` uploads motion + optional WAV and plays both on the daemon's local clock; resolves with `{finished|cancelled|error}`. `cancelMove()` stops mid-play. For record-time flows that need the SAME audio pipeline at capture and replay (so pipeline latency cancels), `uploadAudio(blob)` returns an `uploadId`, then `playUploadedAudio(uploadId)` resolves on the daemon's `started` broadcast (sync anchor), and `cancelAudio()` stops. Default `audioLeadMs` is `-100` — empirical system-wide constant covering combined motor + GStreamer playbin warmup; tune only if you've measured a different value on your hardware. Use these instead of hand-rolling `sendRaw` chunked uploads. Full reference: [`docs/source/SDK/javascript-sdk.md`](docs/source/SDK/javascript-sdk.md#daemon-side-recorded-move-playback).

Speaker / mic volume: `getVolume()` → 0-100, `setVolume(0-100)`, `getMicrophoneVolume()`, `setMicrophoneVolume(0-100)`. All return a `Promise`; `setVolume` resolves with the value the server actually applied (may be clamped / rounded).

Torque / wake: `setMotorMode("enabled"|"disabled"|"gravity_compensation")`, `wakeUp()`, `gotoSleep()`, `isAwake()`, `ensureAwake()`. `robot.robotState.motor_mode` reflects the live state.

Lifecycle primitives (only needed if you opt out of the host shell): `authenticate()` / `login()` / `connect()` / `startSession()` / `stopSession()` / `disconnect()` / `logout()` / `autoConnect()` / `ensureAwake()`. Video: `attachVideo(<video>)` returns a detach fn.

Events: `connected`, `disconnected`, `robotsChanged`, `streaming`, `sessionStopped`, `sessionRejected` (robot busy — inspect `e.detail.activeApp`), `state` (every ~500 ms), `videoTrack`, `micSupported`, `error`.

**Full API:** read the top ~60 lines of [`ts/reachy-mini-sdk.ts`](ts/reachy-mini-sdk.ts) — the file header is a complete reference.

### Media flow — use WebRTC, not `getUserMedia`, for robot media

The `<video>` element you pass to `reachy.attachVideo()` receives the **robot's** camera and microphone over WebRTC. Do **not** call `navigator.mediaDevices.getUserMedia()` to read robot media — that grabs the user's *own* laptop camera.

Bidirectional audio is automatic: with `enableMicrophone: true` on `mountHost`, the SDK acquires the user's mic via `getUserMedia`, negotiates it into the peer connection, and exposes `setMicMuted()`. Check `micSupported` after session start before showing a mic UI.

Rule of thumb: robot IO flows through WebRTC; the user's own laptop IO flows through browser APIs; the SDK wires them together.

### Iframe-embedded apps (mobile-shell handoff)

Reachy Mini apps run in two boot modes (cf. [SPEC.md §2](ts/host/SPEC.md#2-two-boot-modes-one-url-surface)):

| Mode | URL shape | What happens |
|---|---|---|
| **A. Standalone** | `https://<space>.hf.space/` | Full host shell (OAuth → picker → iframe with app). |
| **B. Mobile handoff** | `https://<space>.hf.space/?embedded=1#creds=<base64(CredsBundle)>` | Skip shell; embed boots directly, creds from hash. |

The dispatcher routes between them. **The same `embed.ts` works in both modes** — `connectToHost()` reads creds from the URL hash in Mode B, waits for `host:init` in Mode A, and resolves identically. App authors don't parse URLs, don't manage `sessionStorage`, don't gate on embed-vs-standalone.

The mobile app (`reachy_mini_mobile_app`) is the canonical Mode-B host: it pre-authenticates, pre-picks a robot, then opens the Space in a WebView with creds in the URL hash. Reference implementation: [`reachy_mini_mobile_app/src/ui/panels/apps-list/AppIframeOverlay.tsx`](https://github.com/pollen-robotics/reachy_mini_mobile_app).

For host-shell authors (writing a NEW iframe parent), the wire protocol lives in [`ts/host/src/lib/protocol.ts`](ts/host/src/lib/protocol.ts) and the contract is in [`ts/host/SPEC.md` §4-§6](ts/host/SPEC.md#4-mode-b-mobile-handoff-flow).

### Local development

Two options, switched via `mountHost()` props. Reference apps support both via `.env.local`.

**Option A: personal access token (no OAuth).** Fastest for local dev:

1. Get a token at <https://huggingface.co/settings/tokens> (read scope is enough).
2. Create `.env.local` (gitignored):
   ```
   VITE_HF_TOKEN=hf_xxx
   VITE_HF_USERNAME=your-handle
   ```
3. In `dispatch.ts`, forward to `mountHost`:
   ```ts
   mountHost({
     appName: "My App",
     devToken: import.meta.env.VITE_HF_TOKEN && import.meta.env.VITE_HF_USERNAME
       ? { token: import.meta.env.VITE_HF_TOKEN, userName: import.meta.env.VITE_HF_USERNAME }
       : undefined,
   });
   ```
4. `npm run dev` → you're signed in on page load. **Never commit the token.**

**Option B: real OAuth client ID.** Use when you're touching the OAuth / logout paths:

1. Create one at <https://huggingface.co/settings/applications/new> (Homepage + Redirect = `http://localhost:5173`, scopes `openid` + `profile`).
2. `VITE_HF_OAUTH_CLIENT_ID=...` in `.env.local`.
3. `mountHost({ appName: "My App", clientId: import.meta.env.VITE_HF_OAUTH_CLIENT_ID })`.

For testing Mode B locally (the mobile-handoff URL shape), see [`APP_AUTHOR_GUIDE.md` §11](ts/host/APP_AUTHOR_GUIDE.md#11-faq-and-common-pitfalls).

### Common gotchas

| Symptom | Cause |
|---|---|
| Pushed to the Space but nothing changed | Forgot to re-run `npm run build` before copying `dist/` over. The Space serves whatever's at the repo root, not whatever's in your local `src/`. |
| Live page shows raw `__OAUTH_CLIENT_ID__` in the HTML | HF didn't substitute the placeholder. Check `hf_oauth: true` is in the frontmatter, the placeholder is still the literal string `__OAUTH_CLIENT_ID__` (Vite / minifier didn't rename it), and you're on `sdk: static`. |
| `connectToHost()` hangs forever | Dispatcher isn't routing to embed correctly. Check `?embedded=1` is being passed; check console for postMessage origin errors. |
| Top bar shows the emoji fallback | `appIconUrl` not resolvable. Open `https://<space>.static.hf.space/icon.svg` directly; confirm `public/icon.svg` is in your source tree and Vite is copying it into `dist/`. |
| App missing from mobile catalog | `reachy_mini_js_app` tag missing from README frontmatter, or `icon.svg` / `icon.png` not at `public/` in your **source** repo (the catalog API reads from the Space repo's `siblings`, which means you may need to also commit `public/icon.svg` alongside the build artifacts depending on your deploy layout). |
| Head doesn't move | `setHeadRpyDeg` called before `connectToHost()` resolved. Always `await` the handle. |
| Head doesn't move, session *is* streaming | Robot is asleep (torque off). With the host shell, this should not happen (`ensureAwake()` runs as part of boot) — file a bug. Without it, call `await reachy.ensureAwake()` after `startSession()`. |
| Audio stays silent after `setAudioMuted(false)` | Browser requires unmute inside a user-gesture handler. |
| `Robot is busy: <our own appName>` | Two SDK instances on the same `appName`. Likely a Strict-Mode double mount; ensure your embed runs `connectToHost()` exactly once. The host enforces "single SDK per tab" ([SPEC §8.1](ts/host/SPEC.md#81-single-live-sdk-per-tab)). |
| `Robot is busy` (other peer) | Robot is locked by another app. Surface `e.detail.activeApp` in the UI. |
| Stream laggy | Network jitter > 500 ms. Check the latency overlay in the reference apps. |
| UI broken on phone | Hardcoded pixel widths or desktop-only layout. Reference apps use the responsive viewport meta; don't undo it. |
| Volume slider floods the data channel | Wire `setVolume` on the slider's `'change'` event (release) — not `'input'` (per-pixel). Sync once on streaming-start with `await robot.getVolume()` to reflect the robot's real level, then reflect the value that `setVolume` resolves with back into the slider. |
| Vite warns about React installed twice | Add `react`, `react-dom`, `@emotion/react`, `@emotion/styled`, `@mui/material` to `resolve.dedupe` in `vite.config.ts`. Full snippet: [`APP_AUTHOR_GUIDE.md` §11](ts/host/APP_AUTHOR_GUIDE.md#11-faq-and-common-pitfalls). |

### Legacy: minimal CDN-only path (`webrtc_example`)

Before the host shell, JS apps were `sdk: static` HF Spaces with a single `index.html` importing the SDK directly from jsDelivr (`https://cdn.jsdelivr.net/gh/pollen-robotics/reachy_mini@vTAG/js/reachy-mini.js`) and reimplementing OAuth + picker + session lifecycle by hand. The canonical example is [`cduss/webrtc_example`](https://huggingface.co/spaces/cduss/webrtc_example) (~500 lines).

**Use this only** for one-off prototypes that don't need the host shell's surface (no top bar, no picker, no theme propagation, no mobile-catalog tile). For anything you'd share, **start from a reference app instead** — you get OAuth, picker, mobile-catalog discovery, mode-B handoff, and the entire `connectToHost()` API for free.

If you do go this route, pin the SDK to `v1.7.2` or newer (earlier versions default `signalingUrl` to a decommissioned HF Space) and use `robot.autoConnect({ pickRobot })` to handle both standalone and embed modes from the URL.

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
