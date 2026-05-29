# Skill: Create JS App

## When to Use

- User wants to build a Reachy Mini app that runs in the browser (open a
  link, drive the robot from a phone or laptop, share by URL).
- User wants to "vibe code" an app and doesn't care about the toolchain.
- User asks for a web UI / game / visualizer / controller for the robot.
- Anything that should be a shareable Hugging Face Space rather than an
  on-robot Python app. (For on-robot Python apps, use `create-app.md`.)

This skill gets a working app to a deployed Space fast, for **any author
profile**, by giving you one golden path, hard boundaries, a copy-paste
scaffold, and a definition of done. The deep reference is
[`ts/APP_CREATION_GUIDE.md`](../ts/APP_CREATION_GUIDE.md) - read it only
when this skill points you there.

## Mental Model (read this first)

You are building a **self-contained app**. After `connectToHost()`
resolves you receive a `handle`:

- `handle.reachy` - a live `ReachyMini` instance, **already authenticated,
  robot picked, session streaming, robot awake**. Drive the robot through
  this and nothing else.
- `handle.theme` - `'dark' | 'light'`. Mirror it in your UI.
- `handle.config` - optional external config (deep-link / mobile). Cast
  and **validate** it; never trust it.
- `handle.onLeave(cb)` / `onThemeChange(cb)` / `onConfigChange(cb)` -
  lifecycle hooks.

You **never** write auth, robot picking, session lifecycle, signaling, or
peer-connection code. The host shell owns all of that, outside your
iframe. Your job is the UI and the robot choreography. That is the entire
contract - your app is agnostic of everything else.

## Boundaries

### Always

- Go through the host shell: `mountHost()` in `dispatch.ts`,
  `connectToHost()` in `embed.ts`. Free OAuth + picker + top bar + leave.
- Pin the **exact** SDK build everywhere it appears (npm dep AND any CDN
  URL): `@pollen-robotics/reachy-mini-sdk@1.8.0-rc1-main.fd4354c`.
- Register `onLeave(() => safelyReturnToPose(handle.reachy))` so the robot
  returns to a safe rest pose when the user leaves.
- Validate `handle.config` at the boundary before using it.
- Mobile-first: `viewport-fit=cover`, fluid units, touch targets >= 44px,
  no hover-only affordances.
- Ship `public/icon.svg` and the `reachy_mini_js_app` tag (catalog needs both).

### Ask First

- `sdk: docker` instead of `sdk: static` (only if you need a server,
  secrets, websockets, or Python compute - see the guide).
- Microphone / camera (`enableMicrophone: true`, `attachVideo`).
- npm dependencies other than the SDK.
- A desktop-only / kiosk UI (default is mobile-first).

### Never

- Roll your own OAuth, sign-in screen, or robot picker.
- Call `reachy.stopSession()` yourself - the host tears down; use `onLeave`.
- Reach into host internals or private SDK fields (e.g. `reachy._pc`).
  Use the public API (`attachVideo`, `enableMicrophone`).
- Carry degrees through motion code below the UI layer - speak radians /
  magic-mm; convert at the UI boundary with `degToRad` / `radToDeg`.
- Mutate `INIT_POSE` or `DEFAULT_SCALED_DURATION_PRESET` (deep-frozen).

## Procedure

### Step 1: Write `plan.md` first

Before any code, create `plan.md` in the app directory: what the user
wants, your robot choreography (which gestures map to which states), the
UI, and any open questions. Wait for answers before scaffolding. (Same
rule as Python apps.)

### Step 2: Pick the path

| Path | Use when | Deploy |
|------|----------|--------|
| **TS + Vite (default)** | Almost always. Production apps, TypeScript, npm deps. | `sdk: static` + `app_build_command`, HF builds on push |
| Bare HTML + CDN | Trivial prototype, no Node toolchain wanted | `sdk: static`, no build. See guide §11.5 |
| Docker | Need a server / secrets / Python | Ask first. `sdk: docker`. See guide |

Default to **TS + Vite** unless the user explicitly wants otherwise.

### Step 3: Scaffold (TS + Vite golden path)

Create these files. They are the complete integration surface - the app
is **framework-agnostic inside `embed.ts`**: swap the body for React,
Svelte, Vue, or vanilla. The host doesn't care.

`package.json`:

```json
{
  "name": "my-reachy-app",
  "private": true,
  "type": "module",
  "scripts": { "dev": "vite", "build": "tsc -b && vite build", "preview": "vite preview" },
  "dependencies": { "@pollen-robotics/reachy-mini-sdk": "1.8.0-rc1-main.fd4354c" },
  "devDependencies": { "typescript": "^5.5.4", "vite": "^5.4.10" }
}
```

`README.md` frontmatter (the deploy contract):

```yaml
---
title: My Reachy App
emoji: 🤖
colorFrom: purple
colorTo: indigo
sdk: static
app_build_command: npm ci && npm run build
app_file: dist/index.html
pinned: false
hf_oauth: true
short_description: One line, max 60 chars.
tags:
  - reachy_mini
  - reachy_mini_js_app
---
```

`index.html`, `vite.config.ts`, `tsconfig.json`, `.gitignore` (with
`node_modules/` and `dist/`): copy verbatim from guide §3.1 + §11.
`src/dispatch.ts`: copy from guide §3.2 (sets `window.ReachyMini`,
branches on `?embedded=1`).

`src/embed.ts` - **your app, with animation best-practices already wired**:

```ts
import { connectToHost } from "@pollen-robotics/reachy-mini-sdk/host/embed";
import { safelyReturnToPose } from "@pollen-robotics/reachy-mini-sdk/animation";

async function main() {
  const handle = await connectToHost();
  const { reachy } = handle;

  document.documentElement.setAttribute("data-theme", handle.theme);
  handle.onThemeChange((t) => document.documentElement.setAttribute("data-theme", t));

  // Render your UI here. Drive the robot via `reachy`:
  //   reachy.setHeadRpyDeg(roll, pitch, yaw)
  //   reachy.setAntennasDeg(right, left)
  //   reachy.setBodyYawDeg(yaw)
  // For long / audio moves use reachy.playMove(...). See the SDK reference.

  // Canonical safe teardown - ALWAYS register this:
  handle.onLeave(() => safelyReturnToPose(reachy));
}

void main().catch((err) => {
  console.error("[my-app] boot failed", err);
  window.parent.postMessage(
    { source: "reachy-mini", type: "embed:error", version: 1, message: String(err), fatal: true },
    window.location.origin,
  );
});
```

`public/icon.svg`: a square `viewBox="0 0 24 24"` SVG, readable at 16px,
colors inlined.

### Step 4: Animation best-practices

- Gestures use degrees at the UI boundary, radians/magic-mm below it.
- For smooth keyframed gestures, tween client-side with `requestAnimationFrame`
  and feed `setHeadRpyDeg` / `setAntennasDeg` each frame.
- For long or audio-synced moves, prefer daemon-side `reachy.playMove(motion, { audioBlob })`.
- Use `scaledDuration(current, target)` from `/animation` to size move
  durations instead of hardcoding. Full recipe: guide §14.

### Step 5: Local dev

`.env.local` (gitignored) with `VITE_HF_TOKEN` + `VITE_HF_USERNAME` to
skip OAuth, then `npm install && npm run dev` -> http://localhost:5173.
See guide §9.

### Step 6: Deploy

```bash
npm ci && npm run build          # sanity-check the build HF will run
hf repos create <app-name> --repo-type space --space-sdk static
git init -b main
git remote add space git@hf.co:spaces/<username>/<app-name>
git add -A && git commit -m "feat: initial deploy"
git push -u space main
```

Push **source only**; HF runs `app_build_command` and serves `app_file`.
Never commit `dist/`. Full deploy details + FAQ: guide §11.

## Definition of Done

- [ ] `plan.md` written and approved before coding.
- [ ] `npm run build` passes clean (no TS errors).
- [ ] App goes through `mountHost` / `connectToHost` - zero auth/picker code.
- [ ] `onLeave` registered with `safelyReturnToPose`.
- [ ] SDK pinned to the exact build string in `package.json`.
- [ ] `handle.config` validated before use (if used).
- [ ] Theme mirrored from `handle.theme` + `onThemeChange`.
- [ ] Mobile-first: tested in phone emulation, touch targets >= 44px.
- [ ] `public/icon.svg` committed; frontmatter has `reachy_mini_js_app` tag
      and `short_description` <= 60 chars.
- [ ] Deployed; standalone Space URL loads, sign-in -> picker -> app works.

## Reference

- Deep guide (scaffold details, deploy, host contract, FAQ): [`ts/APP_CREATION_GUIDE.md`](../ts/APP_CREATION_GUIDE.md)
- Runtime SDK API (methods, events, daemon-side playback): [`docs/source/SDK/javascript-sdk.md`](../docs/source/SDK/javascript-sdk.md)
- Animation helpers (`safelyReturnToPose`, `scaledDuration`, `INIT_POSE`): guide §14
