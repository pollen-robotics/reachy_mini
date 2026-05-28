# App Creation Guide

> ### Use `@pollen-robotics/reachy-mini-sdk@1.8.0-rc1` today
>
> Every reference app currently pins the same SDK release line:
> **`1.8.0-rc1`** (precisely `1.8.0-rc1-main.fd4354c`). The host shell,
> the embed adapter, the SDK runtime, and the daemon on the robot
> are validated end-to-end against that build. Mixing versions
> across these boundaries causes silent protocol drift - see
> [§10 SDK version pinning](#10-sdk-version-pinning).
>
> Add this to your `package.json`:
>
> ```json
> { "dependencies": { "@pollen-robotics/reachy-mini-sdk": "1.8.0-rc1-main.fd4354c" } }
> ```

**This is the single source of truth for building a Reachy Mini JS
app.** [`../AGENTS.md`](../AGENTS.md) at the repo root points here;
the JS SDK reference at [`../docs/source/SDK/javascript-sdk.md`](../docs/source/SDK/javascript-sdk.md)
documents the runtime API (motion, events, daemon-side playback) once
`connectToHost()` resolves. Everything else - scaffolding, deploy,
host shell, gotchas - lives here.

How to ship a Hugging Face Space that runs on a Reachy Mini robot,
using `@pollen-robotics/reachy-mini-sdk/host` for OAuth, robot picking, session
lifecycle, and a top bar - so your code stays focused on **your
app's UI** and nothing else.

| Doc        | Purpose                              | Audience                          |
|------------|--------------------------------------|-----------------------------------|
| **This**   | **Single source of truth for app authors** (scaffold, deploy, host contract, invariants) | **You, building a new app**       |
| [`../docs/source/SDK/javascript-sdk.md`](../docs/source/SDK/javascript-sdk.md) | Runtime SDK API reference (methods, events, state machine, daemon-side playback) | App authors after `connectToHost()` resolves |
| [`host/README.md`](./host/README.md) | One-page tour of the `@pollen-robotics/reachy-mini-sdk/host` package layout | First-time visitors to the host source |

## Table of contents

1. [What you get for free](#1-what-you-get-for-free)
2. [Quickstart: clone a reference app](#2-quickstart-clone-a-reference-app)
3. [The app-author contract](#3-the-app-author-contract)
4. [`mountHost()` API](#4-mounthost-api)
5. [`connectToHost()` API](#5-connecttohost-api)
6. [Visual identity: icon, name, emoji](#6-visual-identity-icon-name-emoji)
7. [Receiving an external config (deep-link, mobile)](#7-receiving-an-external-config)
8. [Cleaning up on leave](#8-cleaning-up-on-leave)
9. [Local dev: HF token vs OAuth redirect](#9-local-dev)
10. [SDK version pinning](#10-sdk-version-pinning)
11. [Deploying to Hugging Face Spaces](#11-deploying-to-hugging-face-spaces)
    1. [Required frontmatter](#111-required-frontmatter)
    2. [Build and push](#112-build-and-push)
    3. [What HF does on build + serve](#113-what-hf-does-on-build--serve)
    4. [Cache busting](#114-cache-busting)
    5. [Alternative: bare HTML + CDN, no bundler](#115-alternative-bare-html--cdn-no-bundler)
12. [FAQ and common pitfalls](#12-faq-and-common-pitfalls)
13. [Architecture reference (host ↔ embed contract)](#13-architecture-reference-host--embed-contract)
    1. [Roles: app · host · embed](#131-roles-app--host--embed)
    2. [App identity & official apps](#132-app-identity--official-apps)
    3. [Two boot modes, one URL surface](#133-two-boot-modes-one-url-surface)
    4. [Host phase state machine + handoff sequence](#134-host-phase-state-machine--handoff-sequence)
    5. [Engineering invariants](#135-engineering-invariants)
    6. [Protocol v1 messages](#136-protocol-v1-messages)
    7. [Non-goals](#137-non-goals)
    8. [Threat model](#138-threat-model)
14. [Robotics best practices](#14-robotics-best-practices) - `@pollen-robotics/reachy-mini-sdk/animation`
    1. [Pose types: `Pose` vs `PartialPose`](#141-pose-types-pose-vs-partialpose)
    2. [Distance & scaled duration](#142-distance--scaled-duration)
    3. [Safe return to home pose (`safelyReturnToPose`)](#143-safe-return-to-home-pose-safelyreturntopose)
    4. [Standalone exit hooks (`installShutdownHandler`)](#144-standalone-exit-hooks-installshutdownhandler)
    5. [Daemon parity warning](#145-daemon-parity-warning)
    6. [Anti-patterns](#146-anti-patterns)

---

## 1. What you get for free

By integrating `@pollen-robotics/reachy-mini-sdk/host`, your app **does not have to
write**:

- **Hugging Face OAuth**: sign-in screen, redirect handling,
  token storage, sign-out menu - all in the host shell.
- **Robot discovery and picker**: list of online robots, live
  online/offline/busy updates, click-to-pick.
- **Connection overlay**: the 3-step "Connecting / Starting
  session / Waking up" view rendered on top of your iframe.
- **End-session button and tear-down**: a "Back to apps" affordance
  in the top bar that cleanly closes the WebRTC session.
- **Dark / light theme switching**: respected from
  `prefers-color-scheme` or HF settings, propagated to your iframe.

What **you** write:

- `index.html` (~30 lines of theme bootstrap + OAuth placeholders).
- `src/dispatch.ts` (~20 lines, picks shell vs embed mode and self-
  assigns `window.ReachyMini`).
- `src/embed.{ts,tsx}` - **your app's actual code**. You receive a
  live `ReachyMini` SDK handle and render whatever you want.
- `public/icon.svg` - one SVG that powers the top bar, the mobile
  catalog tile, and the favicon.

You can use **any framework** inside your `embed` entry: React,
Svelte, Vue, vanilla TS. The host runs outside your iframe and
doesn't care.

---

## 2. Quickstart: clone a reference app

Pick the reference closest to your needs and clone its repo from Hugging Face:

| Reference app                       | Stack                       | Use it for                                  |
|-------------------------------------|-----------------------------|---------------------------------------------|
| [`pollen-robotics/reachy_mini_minimal_conversation`](https://huggingface.co/spaces/pollen-robotics/reachy_mini_minimal_conversation) | **Vanilla TS + Vite**       | Smallest bundled runtime, no framework      |
| [`pollen-robotics/reachy_mini_emotions`](https://huggingface.co/spaces/pollen-robotics/reachy_mini_emotions)                         | React 19 + MUI 7 + Vite     | UI-rich app with rich components / theming  |
| [`pollen-robotics/reachy_mini_telepresence`](https://huggingface.co/spaces/pollen-robotics/reachy_mini_telepresence)                 | React 19 + MUI 7 + Vite     | App with camera / media streams             |

```bash
# Example: start from the vanilla TS template
git clone https://huggingface.co/spaces/pollen-robotics/reachy_mini_minimal_conversation my_new_app
cd my_new_app
# Edit package.json `name`, README frontmatter (`title`, `emoji`,
# `short_description`), public/icon.svg, and src/embed.ts to your app.
npm install
npm run dev
# → http://localhost:5173
```

All three reference apps pin `@pollen-robotics/reachy-mini-sdk` to
the same RC version in their `package.json` - see [§10 SDK version pinning](#10-sdk-version-pinning).

> **Prefer a no-build path?** You can also ship the app as a single
> `index.html` with the SDK loaded from a CDN. No `package.json`, no
> build step. See [§11.5](#115-alternative-bare-html--cdn-no-bundler).

---

## 3. The app-author contract

Exactly **three source files** are the entire integration surface,
plus `public/icon.svg`, a `package.json` (for the Vite build + the
SDK npm dep), and the Space `README.md` frontmatter
(see [§11 Deploying](#11-deploying-to-hugging-face-spaces)).

Don't add mandatory files outside this list - keep apps
interchangeable.

### 3.1 `index.html`

The reference apps' `index.html` no longer loads the SDK from a CDN
`<script>` tag - `src/dispatch.ts` imports the SDK from npm and self-
assigns `window.ReachyMini` before any host code runs. This removes
the jsDelivr branch / cache-purge friction we used to live with.

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover" />
    <title>My App</title>
    <link rel="icon" href="/icon.svg" type="image/svg+xml" />

    <!-- Theme bootstrap: paints the right palette BEFORE CSS lands so
         there's no flash. Priority: `?theme=dark|light` query param
         (set by the host when iframing us), then `prefers-color-scheme`. -->
    <script>
      (function () {
        try {
          var params = new URLSearchParams(window.location.search);
          var raw = params.get("theme");
          var mode;
          if (raw === "dark" || raw === "light") {
            mode = raw;
          } else if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches === false) {
            mode = "light";
          } else {
            mode = "dark";
          }
          document.documentElement.setAttribute("data-theme", mode);
        } catch (_) {
          document.documentElement.setAttribute("data-theme", "dark");
        }
      })();
    </script>

    <!-- HF Spaces helper variables.
         - In production, HF substitutes these placeholders at file-
           serve time when `hf_oauth: true` is set on the Space.
         - In `npm run dev`, placeholders stay untouched; we detect
           that and drop `window.huggingface` so the SDK falls back to
           the clientId you supplied via `mountHost({ clientId })`. -->
    <script>
      (function () {
        var clientId = "__OAUTH_CLIENT_ID__";
        var scopes = "__OAUTH_SCOPES__";
        var spaceHost = "__SPACE_HOST__";
        var spaceId = "__SPACE_ID__";
        var looksSubstituted = clientId && clientId.indexOf("__") !== 0;
        if (looksSubstituted) {
          window.huggingface = window.huggingface || {};
          window.huggingface.variables = {
            OAUTH_CLIENT_ID: clientId,
            OAUTH_SCOPES: scopes && scopes.indexOf("__") !== 0 ? scopes : "openid profile",
            SPACE_HOST: spaceHost && spaceHost.indexOf("__") !== 0 ? spaceHost : "",
            SPACE_ID: spaceId && spaceId.indexOf("__") !== 0 ? spaceId : "",
          };
        }
      })();
    </script>
  </head>
  <body>
    <div id="root"></div>
    <!-- Single dispatcher. Picks the host shell (standalone visit) or
         the embedded app (host's iframe) based on `?embedded=1`. -->
    <script type="module" src="/src/dispatch.ts"></script>
  </body>
</html>
```

### 3.2 `src/dispatch.ts`

The dispatcher does three things, in order:

1. Imports `ReachyMini` from npm and self-assigns `window.ReachyMini` -
   both the host shell and the embed wait for that global.
2. Dispatches a `reachymini:ready` event so the embed's wait loop
   takes its fast path.
3. Branches on `?embedded=1` (legacy alias `?embed=1` also accepted):
   embed bundle vs host shell bundle.

```ts
import { ReachyMini } from "@pollen-robotics/reachy-mini-sdk";

(window as unknown as { ReachyMini: typeof ReachyMini }).ReachyMini = ReachyMini;
window.dispatchEvent(new Event("reachymini:ready"));

const params = new URLSearchParams(window.location.search);
const isEmbed =
  params.get("embedded") === "1" || params.get("embed") === "1";

if (isEmbed) {
  void import("./embed");
} else {
  void import("@pollen-robotics/reachy-mini-sdk/host/auto").then(({ mountHost }) => {
    mountHost({
      appName: "My App",
      appIconUrl: "/icon.svg",
      appEmoji: "🤖",
      enableMicrophone: false,
      // Optional: forward dev shortcuts from .env.local. See §9.
      devToken:
        import.meta.env.VITE_HF_TOKEN && import.meta.env.VITE_HF_USERNAME
          ? {
              token: import.meta.env.VITE_HF_TOKEN as string,
              userName: import.meta.env.VITE_HF_USERNAME as string,
            }
          : undefined,
      clientId: import.meta.env.VITE_HF_OAUTH_CLIENT_ID as string | undefined,
    });
  });
}
```

### 3.3 `src/embed.ts` (vanilla example)

```ts
import { connectToHost } from '@pollen-robotics/reachy-mini-sdk/host/embed';

interface MyConfig {
  startingEmotion?: string;
}

async function main() {
  const handle = await connectToHost<MyConfig>();
  const { reachy, theme, config, onLeave } = handle;

  document.body.innerHTML = '<h1>Connected!</h1>';

  reachy.setHeadRpyDeg(0, 10, 0);

  onLeave(async () => {
    document.body.innerHTML = '<p>Bye!</p>';
  });
}

void main().catch((err) => {
  console.error('[my-app] boot failed', err);
  window.parent.postMessage(
    { source: 'reachy-mini', type: 'embed:error', version: 1,
      message: String(err), fatal: true },
    window.location.origin,
  );
});
```

### 3.4 `public/icon.svg`

A single 24×24-friendly SVG at `public/icon.svg` powers **all three**
identity surfaces (host top bar, mobile catalog tile, browser
favicon). See [§6 Visual identity](#6-visual-identity-icon-name-emoji)
for the resolution path and PNG fallback notes.

### 3.5 `package.json`

`npm`-managed; only mandatory bits are the Vite build script and the
SDK pin. See [§10 SDK version pinning](#10-sdk-version-pinning) for
the exact version string.

---

## 4. `mountHost()` API

Called once from `dispatch.ts` when the URL is **not** in embed mode.
Renders the shell into `#root`. The shell's visual theme (MUI
light/dark) is bundled and not overridable - the host owns its
look, apps own theirs inside the iframe.

```ts
import { mountHost } from '@pollen-robotics/reachy-mini-sdk/host/auto';

mountHost({
  appName: 'My App',          // REQUIRED: passed to the SDK + shown in top bar
  appIconUrl: '/icon.svg',    // optional: top-bar logo (see §6)
  appEmoji: '🤖',             // optional: fallback when no icon.svg
  enableMicrophone: false,    // false unless you need WebRTC audio in
  clientId: undefined,        // optional: HF OAuth client ID; defaults to window.huggingface.variables.OAUTH_CLIENT_ID
  devToken: undefined,        // optional: { token, userName } - dev shortcut, see §9
  target: undefined,          // optional: HTMLElement | string CSS selector; default '#root'
});
```

**Required**: `appName`. Everything else has sensible defaults.

**Return**: `{ dispose(): void }` - call to unmount cleanly. You
usually never need this; the page lifecycle handles it.

---

## 5. `connectToHost()` API

Called once from `embed.{ts,tsx}` to get a live SDK handle.

```ts
import { connectToHost } from '@pollen-robotics/reachy-mini-sdk/host/embed';

interface MyConfig { /* whatever your app accepts */ }

const handle = await connectToHost<MyConfig>();
```

Awaiting `connectToHost()` blocks until:

1. The URL hash creds are parsed and wiped.
2. The SDK script loaded (`window.ReachyMini` ready).
3. The host posted `host:init` (Mode A only; Mode B times out
   after 8 s and proceeds from hash alone).
4. The SDK connected, started a session, and woke the robot.

The resolved `handle` exposes:

```ts
interface ConnectedHandle<TConfig> {
  // Live state at boot
  reachy: ReachyMiniInstance;        // SDK instance, session live, robot awake
  theme: 'dark' | 'light';
  config: TConfig | null;
  appName: string;
  hostName: string;
  userName: string | null;

  // Subscribe to live updates from the host (returns an unsub fn)
  onLeave(cb: () => void | Promise<void>): () => void;
  onThemeChange(cb: (theme: 'dark' | 'light') => void): () => void;
  onConfigChange(cb: (config: TConfig | null) => void): () => void;

  // Push state / requests back to the host
  setAppState(s: { phase, connectingStep?, message? }): void;
  requestLeave(): void;                                      // ask host to end the session
  reportError(msg: string, opts?: { fatal?, detail? }): void;
}
```

The API is intentionally minimal. If you need a custom channel
between host and embed, file a feature request - we'll add it as
a typed message rather than expose a free-form sink.

### Typing your config

`connectToHost<T>()` types the `config` field; runtime validation
is **your job**. An attacker controlling the URL can shape config
freely - cast is not enough.

```ts
const handle = await connectToHost<MyConfig>();
const config = isValidConfig(handle.config) ? handle.config : null;
```

---

## 6. Visual identity: icon, name, emoji

> **App identity is your HF Space ID** (`owner/space`). An app
> published at `huggingface.co/spaces/your-name/cool-thing` has
> identity `your-name/cool-thing` everywhere downstream (mobile
> catalog, "last opened", official badge). Apps in the
> `pollen-robotics/*` namespace are automatically tagged as
> official in the catalog; no extra config.

Your app's identity surfaces in **three independent places**, all
fed from the same sources but each by its own resolution path:

| Surface                       | What it shows                          | How it gets there                                                                                  |
|-------------------------------|----------------------------------------|----------------------------------------------------------------------------------------------------|
| Host top bar (in your iframe) | App logo + app name                    | You pass `appName` + `appIconUrl` + `appEmoji` to `mountHost()` in `dispatch.ts`                   |
| Hugging Face mobile catalog   | App tile (icon + title + description)  | The mobile reads the catalog API; the API reads HF Spaces' frontmatter + probes `public/icon.svg` |
| Browser tab / OS              | Favicon                                | Your `index.html` `<link rel="icon" href="/icon.svg">`                                             |

The host shell itself **does not discover or list other apps**.
It renders only the app it lives in. The catalog is owned by the
mobile and the website API; see §11 FAQ for the API spec if you
need to validate your app shows up.

### Single source of truth: one `icon.svg`, one Space frontmatter

You ship **one** SVG file and **one** README frontmatter; both
the host and the mobile catalog read from them:

1. **`public/icon.svg`** in your repo. Vite copies it verbatim to
   `dist/icon.svg`, where nginx serves it at the root URL of your
   Space. Reference it from `index.html`:

   ```html
   <link rel="icon" href="/icon.svg" type="image/svg+xml" />
   ```

   Pass it to `mountHost()` so the top bar renders it without a
   probe:

   ```ts
   mountHost({ appName: 'My App', appIconUrl: '/icon.svg' });
   ```

   The catalog API also picks it up by listing the Space's
   files (`siblings`) and matching `public/icon.svg` (or
   `public/icon.png`), no live probe required.

2. **HF Space frontmatter** in your `README.md`:

   ```yaml
   ---
   title: My App
   emoji: 🤖
   colorFrom: yellow
   colorTo: red
   sdk: static
   pinned: false
   hf_oauth: true
   short_description: One-line description shown in the catalog.
   tags:
     - reachy_mini
     - reachy_mini_js_app
   ---
   ```

   See [§11.1](#111-required-frontmatter) for the full annotated
   frontmatter.

   - `title` is the app name in the mobile catalog (the in-iframe
     top bar uses `appName` from `mountHost()` instead).
   - `emoji` is the fallback logo when no `icon.svg` is shipped.
     Pass the same value to `mountHost({ appEmoji: '🤖' })` so the
     top bar's fallback matches.
   - `short_description` shows under the app tile in the catalog.
   - The **`reachy_mini_js_app` tag is mandatory** to appear in
     the mobile catalog. The catalog API filters on this exact
     string. Don't remove it.
   - `hf_oauth: true` makes HF auto-provision an OAuth client and
     inject the ID at file-serve time.

### Icon design recommendations

Your icon renders at three different sizes; design for all three:

- **Host top bar inside the iframe**: ~24x24 px square.
- **Mobile catalog tile**: ~64x64 px square card.
- **Browser tab favicon**: 16x16 px.

Practical guidance:

- Use a **square viewBox** (e.g. `viewBox="0 0 24 24"`) so the
  three target sizes all crop identically.
- Keep the icon **readable at 16 px**: thick strokes, simple
  silhouette, max 2-3 distinct shapes.
- Inline all colours; **don't reference external CSS**, the icon
  is served standalone.
- Respect dark and light backgrounds: an icon that vanishes on
  light should provide a `<style>` tag with `@media (prefers-color-scheme)`
  rules or, simpler, use a neutral mid-tone palette.
- **Optimise the SVG**: target ~30 KB or less. Tools: `svgo`,
  Figma's "Export SVG → optimise".

### PNG fallback

If you can't ship SVG (heavy raster art, exported portrait, ...),
the catalog API also accepts `public/icon.png`. SVG wins when both
exist. The host top bar only renders the SVG variant - if your
`mountHost({ appIconUrl })` points at a PNG, it works, but you
lose the crisp upscale on hi-DPI screens.

---

## 7. Receiving an external config

The host accepts a base64-encoded JSON `config` from two sources:

1. **URL parameter**: `https://<space>.hf.space/?config=eyJlbW90aW9uIjoiam95In0=`
   (decoded once, passed verbatim).
2. **Mobile handoff**: the mobile app embeds your Space with
   `?embedded=1#creds=<base64-bundle-including-config>`.

Your app receives `config` typed as `unknown`; cast and validate.

```ts
interface MyConfig { startingEmotion?: string; }

function isMyConfig(v: unknown): v is MyConfig {
  return v != null && typeof v === 'object' && (
    (v as MyConfig).startingEmotion === undefined ||
    typeof (v as MyConfig).startingEmotion === 'string'
  );
}

const handle = await connectToHost<MyConfig>();
const initial: MyConfig = isMyConfig(handle.config) ? handle.config : {};

handle.onConfigChange((next) => {
  if (isMyConfig(next)) /* react to it */;
});
```

If your app's UI state changes in a way the mobile would want to
remember (e.g. user picked a different emotion), persist it in
your app's storage. The host does **not** propagate state
upstream - apps don't push config to the host in v1.

---

## 8. Cleaning up on leave

The host fires a tear-down sequence in three scenarios:

- User clicks "End session" / "Back to apps" in the top bar.
- Your app calls `handle.requestLeave()`.
- The page is unloaded (`pagehide`, e.g. user closes the tab).

In all three cases your `onLeave` callbacks fire. You have **~1.5-2 s**
before the host force-unmounts the iframe; use that to:

```ts
import { safelyReturnToPose } from "@pollen-robotics/reachy-mini-sdk/animation";

handle.onLeave(async () => {
  safelyReturnToPose(handle.reachy); // canonical safe-rest one-liner, see §14.3
  player.cancel();                   // stop streaming motion frames
  audioCtx?.close();                 // release audio
  ws?.close();                       // close any side channels
  await flushTelemetry();            // your async hooks
});
```

`safelyReturnToPose` is the recommended first call in every `onLeave` -
it re-enables torque, computes a scaled duration, and dispatches a
goto to `INIT_POSE`. Full recipe in
[§14.3](#143-safe-return-to-home-pose-safelyreturntopose).

You do **not** need to call `reachy.stopSession()` yourself - the
host does. You also don't need to navigate away; the iframe is
unmounted by the host.

---

## 9. Local dev

You have two options, picked by the `devToken` and `clientId`
props passed to `mountHost()`. Reference apps support both via
`.env.local`.

### Option A: personal access token (no OAuth)

Fastest for local dev. Skips the OAuth redirect entirely.

1. Get a token at <https://huggingface.co/settings/tokens> (read
   scope is enough).
2. Create `.env.local`:

   ```
   VITE_HF_TOKEN=hf_xxx
   VITE_HF_USERNAME=your-handle
   ```

3. In `dispatch.ts`, forward both to `mountHost`:

   ```ts
   mountHost({
     appName: 'My App',
     devToken: import.meta.env.VITE_HF_TOKEN && import.meta.env.VITE_HF_USERNAME
       ? { token: import.meta.env.VITE_HF_TOKEN, userName: import.meta.env.VITE_HF_USERNAME }
       : undefined,
   });
   ```

4. `npm run dev` → you're signed in on page load.

`.env.local` must be gitignored. **Never commit the token.**

### Option B: real OAuth client ID

Use this when you're touching the OAuth / logout paths.

1. Go to <https://huggingface.co/settings/applications/new>.
2. Homepage URL: `http://localhost:5173` · Redirect URIs:
   `http://localhost:5173` · Scopes: at least `openid`, `profile`.
3. Copy the client ID into `.env.local`:

   ```
   VITE_HF_OAUTH_CLIENT_ID=...
   ```

4. Forward to `mountHost`:

   ```ts
   mountHost({
     appName: 'My App',
     clientId: import.meta.env.VITE_HF_OAUTH_CLIENT_ID,
   });
   ```

---

## 10. SDK version pinning

Every reference app pins the same exact SDK version in `package.json`.
Pin yours the same way - mixing versions across `@pollen-robotics/reachy-mini-sdk`,
`@pollen-robotics/reachy-mini-sdk/host`, and the daemon on the robot
produces hard-to-debug protocol drift.

The current pinned version across all three reference apps:

```json
{
  "dependencies": {
    "@pollen-robotics/reachy-mini-sdk": "1.8.0-rc1-main.fd4354c"
  }
}
```

This is the `1.8.0-rc1` release line, with the `-main.fd4354c` suffix
identifying the commit-tagged prerelease build that's been validated
end-to-end against the host shell + daemon. **Use the same string in
your `package.json`** unless you're explicitly tracking a newer RC.

> When a newer RC is published, the source of truth is whichever
> string is currently shared by [`reachy_mini_minimal_conversation`'s
> `package.json`](https://huggingface.co/spaces/pollen-robotics/reachy_mini_minimal_conversation/blob/main/package.json),
> [`reachy_mini_emotions`'s `package.json`](https://huggingface.co/spaces/pollen-robotics/reachy_mini_emotions/blob/main/package.json),
> and [`reachy_mini_telepresence`'s `package.json`](https://huggingface.co/spaces/pollen-robotics/reachy_mini_telepresence/blob/main/package.json).
> If those three diverge, fall back to whatever this guide says.

### Why pin a specific build (not `^1.8.0` or a major like `@1`)?

The host shell, the embed adapter (`connectToHost`), the SDK, and the
robot daemon negotiate over a versioned WebRTC data-channel protocol.
A patch bump on one side that crosses a protocol boundary will
silently fall back (or noisily fail) at runtime - well past the type
checker.

Pin to the exact build string the reference apps use, upgrade
intentionally, and re-test against a live robot before shipping.

---

## 11. Deploying to Hugging Face Spaces

> Reachy Mini JS apps ship as **`sdk: static`** Hugging Face Spaces.
> Two equally valid paths, pick based on your needs:
>
> 1. **Bundled (Vite + TypeScript, HF-side build)** - §11.1-§11.4. The
>    default for production. You push source only; HF runs
>    `app_build_command` on its builder, serves the resulting `app_file`
>    from its CDN, and replaces `__OAUTH_CLIENT_ID__` and friends at
>    file-serve time (because `hf_oauth: true`).
> 2. **Bare HTML + CDN (no bundler, no Node toolchain)** - §11.5. The
>    fastest path for prototypes, learners, and single-file UIs. One
>    `index.html`, SDK imported from jsDelivr, three files total.
>
> Both end up as `sdk: static` Spaces with `hf_oauth: true`. The host
> shell, the `connectToHost()` API, and the OAuth substitution work
> identically in both. Pick whichever fits your project; the rest of
> this guide focuses on the bundled path because it's what production
> apps standardise on.

### 11.1 Required frontmatter

```yaml
---
title: My Reachy Mini App
emoji: 🤖
colorFrom: yellow
colorTo: red
sdk: static
app_build_command: npm ci && npm run build   # HF runs this on its builder
app_file: dist/index.html                     # HF serves the build output as entry
pinned: false
hf_oauth: true
short_description: One-line description shown in the mobile catalog.
tags:
  - reachy_mini
  - reachy_mini_js_app   # mandatory: mobile-catalog discovery filters on this exact string
---
```

- `sdk: static` is the HF Space type. Combined with `app_build_command`
  it triggers a one-shot build container on every push.
- `app_build_command` is the shell command HF runs on its builder. The
  canonical value is `npm ci && npm run build`. Build logs are visible
  in the Space's "Logs" tab.
- `app_file` tells HF which file to serve as the entry point **after the
  build completes**. For a Vite-built SPA that's `dist/index.html`.
- `hf_oauth: true` is what triggers `__OAUTH_CLIENT_ID__` substitution
  inside HTML files in the served output (post-build).
- The **`reachy_mini_js_app` tag is mandatory** for mobile-catalog
  discovery. The catalog API filters on this exact string.
- Apps in the `pollen-robotics/*` namespace are automatically tagged
  as "official" in the catalog (see [§13.2 App identity & official apps](#132-app-identity--official-apps)); no extra config.

> **`short_description` is hard-capped at 60 characters** by the Hub's
> YAML validator. The pre-receive hook rejects the push with a clear
> error if you overshoot; trim and re-commit.

**Minimum `.gitignore` for a Space repo.** Because the build runs on
HF, the source tree must NOT commit build artifacts:

```
node_modules/
dist/
.env
.env.local
.DS_Store
```

> The legacy two-folder, "build-locally-and-push-dist" pattern (one
> source repo + one separate `*-space` clone) is **deprecated**.
> Source-only push is now the supported path. The reference apps
> already follow it.

### 11.2 Build and push

Your source tree IS your Space repo. One folder, one git history.

```bash
# 1. Sanity-check that the build works locally before pushing.
#    HF will replay this exact command on its builder.
npm ci
npm run build
# → dist/ contains the build output. It stays gitignored.

# 2. Create the Space (idempotent; skip on "already exists").
hf repos create <app-name> --repo-type space --space-sdk static

# 3. Add the Space as a git remote and push the source tree.
git init -b main             # if your source tree isn't a git repo yet
git remote add space git@hf.co:spaces/<username>/<app-name>
git add -A
git commit -m "feat: initial deploy"
git push -u space main
```

What you just pushed:

- `README.md` (the frontmatter, including `app_build_command` + `app_file`).
- `index.html` (the source entry, references `/src/dispatch.ts`).
- `package.json`, `package-lock.json`, `tsconfig.json`, `vite.config.ts`.
- `src/` (your `dispatch.ts`, `embed.{ts,tsx}`, styles, helpers).
- `public/icon.svg` (and optional `public/icon.png` fallback).

What HF does on receive:

1. Runs `npm ci && npm run build` in a one-shot build container.
2. Captures the build output at `dist/`.
3. Substitutes `__OAUTH_CLIENT_ID__` and friends in any `.html` inside the
   served output (because `hf_oauth: true`).
4. Serves `dist/index.html` as the entry point at the Space root URL;
   serves the rest of `dist/` (the hashed bundles in `dist/assets/`) as
   static assets via the CDN.

Build logs are streamed to the Space's "Logs" tab. If the build fails,
the Space stays on the previous successful build.

> **Don't commit `dist/`.** If you push a stale local build, HF still
> runs `app_build_command` and overwrites it - but you've polluted the
> git history with megabytes of build artifacts. Keep `dist/` gitignored.

### 11.3 What HF does on build + serve

On `git push`, HF runs (in this order):

1. **Build step**: spins up a one-shot Node container, runs
   `app_build_command` (the canonical `npm ci && npm run build`) on
   the committed source tree. The container is discarded on completion;
   only the build output survives, captured at the path your build tool
   emits (`dist/` for Vite).
2. **OAuth placeholder substitution**: in every `.html` file inside the
   served output, replaces `__OAUTH_CLIENT_ID__`, `__OAUTH_SCOPES__`,
   `__SPACE_HOST__`, and `__SPACE_ID__` with the values HF provisioned
   for the Space (because `hf_oauth: true`).
3. **CDN serve**: serves `app_file` (typically `dist/index.html`) as
   the Space root URL; serves the rest of the build output as static
   assets (immutable caching honours the hashed filenames Vite emits).
4. **Catalog indexing**: indexes the committed `siblings` (the source
   files in git, NOT the build output) for the mobile-catalog probe.
   That's why `public/icon.svg` must be committed at the source repo
   path - the catalog API lists git-tracked files, not the CDN.

Build failures don't take down a working Space: HF keeps serving the
previous successful build until a new build succeeds.

### 11.4 Cache busting

If a push doesn't take effect, push an empty commit to force HF to
re-resolve the Space:

```bash
git commit --allow-empty -m "chore: bust HF Spaces cache" && git push
```

### 11.5 Alternative: bare HTML + CDN, no bundler

If you don't want a Node toolchain, prefer plain JS, or are shipping a
one-off prototype, you can deploy a single `index.html` that imports
the SDK from a CDN. No `package.json`, no `app_build_command`, no
build step. HF serves the repo root as-is.

**Trade-off summary**

| | Bundled (default, §11.1-§11.4) | Bare HTML + CDN |
|---|---|---|
| Toolchain on your machine | Node + npm | Browser only |
| TypeScript | Yes | No (or self-hosted `tsc` build) |
| Cold start | Faster (pre-bundled, hashed CDN cache) | Slower (jsDelivr bundles JS on first hit) |
| Iteration loop in prod | `git push` -> wait for HF build (~30-60 s) | `git push` -> live immediately |
| Best for | Production, multi-file TS apps, npm deps | Prototypes, learners, single-file UIs, demos |

**Two sub-variants of the bare-HTML path**

There are two patterns in the wild, both `sdk: static`, both no-build:

| Sub-variant | SDK import | Host shell | OAuth handled by |
|---|---|---|---|
| **Modern (recommended for new apps)** | `cdn.jsdelivr.net/npm/@pollen-robotics/reachy-mini-sdk@<sha>/+esm` - same npm package as the bundled path, just served from the CDN | Yes (`mountHost` + `connectToHost`) | The host shell (sign-in, picker, top bar, end-session - all free) |
| **Legacy single-file (pre-host-shell)** | `cdn.jsdelivr.net/gh/pollen-robotics/reachy_mini@<tag>/js/reachy-mini.js` - single-file SDK bundle from a GitHub tag | No | The SDK itself (`robot.login()` / `robot.authenticate()`) plus your own picker and gate UI |

Both sub-variants share **everything that matters at runtime** - the
motion API (`setTarget`, `setMotorMode`, `gotoTarget`, `setHeadRpyDeg`,
`setAntennasDeg`, etc.), the WebRTC media flow, the OAuth scopes -
because they target the same robot daemon over the same data-channel
protocol. They differ only in who renders the sign-in screen, the
robot picker, and the top bar: the host shell does it for you on the
modern sub-variant; you draw it yourself on the legacy sub-variant.

For **new** apps, default to the modern sub-variant: less code, free
top bar, mobile-catalog ready out of the box. The legacy sub-variant
remains valid if you want full control over the pre-session UI
(custom welcome screen, animated splash, branded gate, etc.) and are
willing to maintain that surface yourself.

**Minimum frontmatter** (drop `app_build_command` and `app_file`):

```yaml
---
title: My Bare App
emoji: 🤖
colorFrom: yellow
colorTo: red
sdk: static
pinned: false
hf_oauth: true
short_description: Reachy Mini app, no bundler.
tags:
  - reachy_mini
  - reachy_mini_js_app
---
```

**Minimum file layout** (3 files):

```
my-bare-app/
├── README.md         # frontmatter above
├── index.html        # your app
└── style.css         # optional
```

**Importing the SDK from a CDN**

Pin to an exact build SHA - **the same string you would use in
`package.json`** (see [§10 SDK version pinning](#10-sdk-version-pinning)).
jsDelivr's `/+esm` suffix tells the CDN to bundle the package to ESM at
the edge:

```html
<script type="module">
  import { ReachyMini } from "https://cdn.jsdelivr.net/npm/@pollen-robotics/reachy-mini-sdk@1.8.0-rc1-main.fd4354c/+esm";
  import { mountHost } from "https://cdn.jsdelivr.net/npm/@pollen-robotics/reachy-mini-sdk@1.8.0-rc1-main.fd4354c/host/dist/entry/auto.js";
  import { connectToHost } from "https://cdn.jsdelivr.net/npm/@pollen-robotics/reachy-mini-sdk@1.8.0-rc1-main.fd4354c/host/dist/entry/embed.js";

  window.ReachyMini = ReachyMini;
  window.dispatchEvent(new Event("reachymini:ready"));

  const params = new URLSearchParams(window.location.search);
  if (params.get("embedded") === "1") {
    const handle = await connectToHost();
    /* render your app, use handle.reachy.setHeadRpyDeg(...) etc. */
  } else {
    mountHost({ appName: "My Bare App", appEmoji: "🤖" });
  }
</script>
```

Pair with `<link rel="modulepreload" href="...">` tags in `<head>` for
both module URLs to warmup the CDN fetch during HTML parse. This shaves
a few hundred ms off cold-start on mobile networks.

The `__OAUTH_CLIENT_ID__` substitution block from
[§3.1](#31-indexhtml) is **still required** in your `index.html` -
that's what wires HF OAuth into the SDK at runtime. The host shell
(when reached via `mountHost`) handles the rest of the OAuth flow
identically to the bundled path.

**Scaling up without a bundler**

When a single-file `index.html` becomes hard to navigate, split your
code into local ES modules under `lib/` and/or `views/` and let the
browser resolve the `import` graph natively. Common, lightweight
patterns that scale well at this size:

- A tiny custom state store (`state.js` exposing a mutable object plus
  a `render()` pointer the views call after mutating it) - avoids
  pulling in a framework just for re-render.
- Factory functions that close over a shared SDK handle
  (`createMotorSubsystem(robot)`, `createGate({ robot })`, etc.) so
  views don't have to thread the handle through props.
- `IndexedDB` for any persistence the app needs (recorded moves,
  user settings); no Node deps required.
- The `@huggingface/hub` ESM build from jsDelivr if you want Hub
  dataset upload / download without npm.

No transpile, no bundle, just static `import` statements between
sibling files. The architecture transfers cleanly between the two
sub-variants - swap the SDK import URL and route through
`connectToHost` instead of instantiating the SDK directly (see the
migration recipe below).

**Migrating a legacy single-file SDK app to the modern host shell**

If your existing app imports the SDK as
`https://cdn.jsdelivr.net/gh/pollen-robotics/reachy_mini@<tag>/js/reachy-mini.js`
and rolls its own sign-in and picker UI, you can graduate to the host
shell without touching your motion code. The four-step recipe:

1. **Swap the SDK import URL.** Replace your `/gh/.../js/reachy-mini.js`
   import with the npm-CDN equivalent, and add the two host imports:

   ```js
   // BEFORE (legacy single-file)
   import { ReachyMini } from "https://cdn.jsdelivr.net/gh/pollen-robotics/reachy_mini@v1.7.2/js/reachy-mini.js";

   // AFTER (modern host shell, same SDK runtime API)
   import { ReachyMini } from "https://cdn.jsdelivr.net/npm/@pollen-robotics/reachy-mini-sdk@1.8.0-rc1-main.fd4354c/+esm";
   import { mountHost } from "https://cdn.jsdelivr.net/npm/@pollen-robotics/reachy-mini-sdk@1.8.0-rc1-main.fd4354c/host/dist/entry/auto.js";
   import { connectToHost } from "https://cdn.jsdelivr.net/npm/@pollen-robotics/reachy-mini-sdk@1.8.0-rc1-main.fd4354c/host/dist/entry/embed.js";
   ```

2. **Branch on `?embedded=1`.** Wrap your existing app boot in:

   ```js
   window.ReachyMini = ReachyMini;
   window.dispatchEvent(new Event("reachymini:ready"));

   if (new URLSearchParams(location.search).get("embedded") === "1") {
     // Inside the host iframe: the robot is already authenticated,
     // picked, and streaming. Skip your own gate / picker entirely.
     const handle = await connectToHost();
     const robot = handle.reachy;          // same object your app already uses
     // ...rest of your app code, unchanged
     handle.onLeave(() => { /* your cleanup */ });
   } else {
     // Standalone visit: render the shell. It will iframe this same
     // page with ?embedded=1 once the user picks a robot.
     mountHost({ appName: "Your App", appEmoji: "🤖" });
   }
   ```

3. **Delete the dead UI**: your sign-in card, your robot picker, any
   "Sign out" / "End session" affordance, any `robotsChanged` listener
   that drove a picker list. The host shell renders all of these
   outside your iframe.

4. **Keep your motion code exactly as-is.** `robot.setTarget(...)`,
   `robot.setMotorMode(...)`, `robot.gotoTarget(...)`, `robot.setHeadRpyDeg(...)`,
   `robot.setAntennasDeg(...)` - identical method signatures on both
   SDKs. The recorded-move API (`playMove`, `uploadAudio`, `cancelMove`)
   only exists on the modern SDK; if you weren't using it, nothing
   changes. If you reached into private fields like `robot._pc`
   (RTCPeerConnection), prefer the public alternatives:
   `robot.attachVideo(videoEl)` for video, `enableMicrophone: true` on
   `mountHost` for bidirectional audio.

Your `README.md` frontmatter doesn't need any changes: `sdk: static`
and `hf_oauth: true` work for both variants. You can still skip
`app_build_command` / `app_file` since you're not adding a build step.

**When to graduate to Vite + TS (§11.1-§11.4)**

Switch to the bundled path when any of these become true:
- You want TypeScript with strict checks.
- The app loads more than ~10 source files (the CDN waterfall adds up).
- You want hot reload in dev (`npm run dev`).
- You need npm packages other than the SDK (jsDelivr doesn't ship every package well).
- Page-weight matters and you want tree-shaking + Brotli on hashed bundles.

---

## 12. FAQ and common pitfalls

### "I see a `Robot is busy` error even though no one is using the robot"

The host's SDK and the embed's SDK both claim a peer at the
central. The host **must** disconnect when the embed boots; if it
doesn't, the central sees two peers with the same `appName` and
rejects the embed.

This is handled automatically by `@pollen-robotics/reachy-mini-sdk/host`
(see [§13.5.1 Single live SDK per tab](#1351-single-live-sdk-per-tab)).
If you see this in dev, you likely have **two tabs** open on the
same Space - that's expected behaviour.

### "My app loads React + MUI even though I wrote vanilla TS"

The **host shell** is React + MUI. It runs **only outside your
iframe** (sign-in screen, picker, top bar). Once your app is live,
the host's React tree is idle.

Your iframe content is whatever you wrote. Vanilla TS apps stay
slim inside the iframe.

### "Vite warns about React being installed in two places"

You're using the legacy `file:./vendor/reachy-mini-host` dep
pattern (now unsupported — the host ships from npm as part of
`@pollen-robotics/reachy-mini-sdk`). Migrate to the npm dep and,
if you still see the warning, add to your `vite.config.ts`:

```ts
export default defineConfig({
  resolve: {
    dedupe: ['react', 'react-dom', 'react/jsx-runtime',
             '@emotion/react', '@emotion/styled',
             '@mui/material', '@mui/icons-material'],
  },
  optimizeDeps: {
    include: ['@pollen-robotics/reachy-mini-sdk',
              '@pollen-robotics/reachy-mini-sdk/host',
              '@pollen-robotics/reachy-mini-sdk/host/auto',
              '@pollen-robotics/reachy-mini-sdk/host/embed'],
  },
});
```

### "I want a different sign-in flow"

Not supported in v1. The host owns OAuth. If you need a custom
flow, the standalone shell isn't for you - publish your Space
with the host disabled (just don't call `mountHost()`) and roll
your own.

### "I want a different theme than the bundled MUI one"

The host shell's look is fixed (light + dark MUI bundle). Apps
own their own theme **inside the iframe** - use the
`handle.theme` value as your mode signal and wrap your app in
whatever ThemeProvider you want.

```ts
const handle = await connectToHost();
// Mirror `handle.theme` ('dark' | 'light') in your own
// ThemeProvider. The host pushes updates via onThemeChange().
```

### "The icon doesn't show up in the top bar"

Check the three sources in priority order (§6):

1. Is `/icon.svg` reachable at the deployed URL? Open
   `https://<space>.hf.space/icon.svg` directly.
2. Is the file's MIME type `image/svg+xml`? The host's probe
   checks the response's `content-type`.
3. Did you pass `appIconUrl: '/icon.svg'` to `mountHost()`?

If 1 + 2 + 3 are correct and it still fails, file a bug.

### "My Space serves the bundle but the OAuth login redirects loop"

HF only substitutes `__OAUTH_CLIENT_ID__` when `hf_oauth: true` is
set in the README frontmatter **and** the file is HTML. Common
mistakes:

- `hf_oauth: true` missing → placeholder stays as literal
  `"__OAUTH_CLIENT_ID__"`; the SDK falls back to no client ID and
  the login never resolves.
- You pushed a built `dist/index.html` that already had the
  placeholder replaced locally (e.g. you ran with `.env.local` and
  some bundler hardcoded the value). HF only substitutes
  `__...__` literals; if the file already has a real ID baked in
  for a different OAuth client, the redirect targets the wrong app.
- The Space pre-dates the `hf_oauth` substitution (very old
  Spaces). Re-create the Space.

### "My app doesn't appear in the mobile catalog"

Three things must be true simultaneously:

1. The Space tags include the exact string `reachy_mini_js_app` (see
   the [§11.1 frontmatter](#111-required-frontmatter)).
2. `public/icon.svg` exists in the **committed repo tree** (not just
   in `dist/`). The catalog probe inspects `siblings`, which is a
   listing of committed files, not served URLs.
3. The Space is public (or the requesting user has access).

### "My HF build failed - where are the logs?"

The Space's **Logs** tab. With `sdk: static` + `app_build_command`, HF
streams the build container's stdout/stderr there in real time. Common
failures and where to look:

- **`npm ci` peer / lockfile mismatch**: regenerate `package-lock.json`
  locally (`rm package-lock.json && npm install`), commit, push again.
- **`tsc` errors**: same TS errors as `npm run build` locally - reproduce
  by running the exact `app_build_command` on your laptop.
- **Missing `app_file` after build**: HF can't find what you told it to
  serve. Verify `app_file: dist/index.html` matches your Vite output
  path; if you customised `build.outDir`, update `app_file` too.
- **`short_description` over 60 chars**: rejected by the pre-receive
  hook before the build even starts. Trim and re-push.

While the build is failing, HF keeps serving the **previous** successful
build. Your URL doesn't 404 unless you've never successfully built.

### "Where do I see if my app crashed at boot?"

Three places, in order:

1. Browser console of the standalone Space tab (mountHost errors).
2. Browser console of the iframe (embed errors). The embed
   `postMessage`s any boot error back to the host as
   `embed:error` - the host surfaces fatal ones via a banner.
3. HF Space "Logs" tab - only build-time errors show up here for
   static Spaces (no runtime container).

### "How do I test the mobile-handoff mode locally?"

Hit your dev server at:

```
http://localhost:5173/?embedded=1#creds=<base64({"hfToken":"hf_xxx","userName":"you","robotPeerId":"abc","signalingUrl":"https://...","theme":"dark","config":null,"hostName":"Reachy Mini","appName":"My App"})>
```

The dispatcher will skip the shell and go straight to your embed.
Useful for testing the embed path without spinning up the mobile
app. The exact bundle shape is documented at
[§13.3 Two boot modes](#133-two-boot-modes-one-url-surface).

---

## 13. Architecture reference (host ↔ embed contract)

> **You don't need this section to ship an app.** §1-§12 plus §14
> (Robotics best practices) are enough. This appendix is the canonical
> contract between the **app**, the **host shell**, and the **embed
> adapter** - useful when you're debugging a weird boot, considering an
> unusual deployment, or contributing to the host shell itself.

### 13.1 Roles: app · host · embed

Three actors, one app repository:

| Actor      | Lives in                                              | Owns                                                                       |
|------------|-------------------------------------------------------|----------------------------------------------------------------------------|
| **App**    | `index.html` + `src/dispatch.ts` + `src/embed.{ts,tsx}` | UI, app-specific UX, **full freedom over framework / tooling choices**     |
| **Host**   | `@pollen-robotics/reachy-mini-sdk/host/auto`          | OAuth, robot discovery, robot picker, connecting overlay, end-session flow |
| **Embed**  | `@pollen-robotics/reachy-mini-sdk/host/embed`         | SDK lifecycle inside the iframe (`startSession`, `ensureAwake`, teardown)  |

The **App** consumes the Reachy Mini SDK (imported in
`src/dispatch.ts` and self-assigned to `window.ReachyMini`) plus the
`@pollen-robotics/reachy-mini-sdk/host` subpath exports. It contains
**zero auth code, zero picker code, zero session-lifecycle code**.

#### Why React + MUI for the host shell (and only the host)

The host shell needs a real component library: sign-in forms, robot
picker lists, connecting overlays, top bar, dark-mode toggles. It's
built with **React 19 + MUI 7 + Emotion**.

- The shell renders **only outside your iframe** and only between
  sessions; once your app is live, the shell's React tree is idle.
- Apps written in another framework still load the shell's bundle
  for sign-in / picker UI. That's the cost of the iframe model and
  we accept it.

The trade-off favours **fast host iteration + tech freedom for
apps** over a slimmer host shell.

### 13.2 App identity & official apps

A Reachy Mini app is uniquely identified by its **Hugging Face Space
ID**, of the form `owner/space` (e.g. `pollen-robotics/emotions`).
Everything downstream of identity flows from this single string:

- The catalog API filters and dedupes apps by `space.id`.
- The mobile app stores and recalls "last opened" apps by
  `owner/space`.
- The host shell does **not** need this ID at runtime (it lives
  inside the app it renders); it's used solely by the discovery
  surface.

**An app is "official" if and only if its Space ID starts with
`pollen-robotics/`.** No allowlist, no separate registry, no
`official: true` field. Adding `pollen-robotics/` to your URL is the
entire qualification. Where the distinction surfaces:

| Surface                | Behaviour                                                       |
|------------------------|-----------------------------------------------------------------|
| Mobile catalog         | "Official" badge / sort priority on `pollen-robotics/*` Spaces  |
| Website `/api/js-apps` | Returns `isOfficial: true` for `pollen-robotics/*`              |
| Host shell             | **No notion of "official"**. Renders the app the same way always |

### 13.3 Two boot modes, one URL surface

The same `index.html` is served for both modes. The dispatcher
(`src/dispatch.ts`) picks between them based on the URL.

| Mode                  | URL shape                                                            | What happens                                         |
|-----------------------|----------------------------------------------------------------------|------------------------------------------------------|
| **A. Hub standalone** | `https://<space>.hf.space/`                                          | Full host shell (OAuth → picker → iframe with app)   |
| **B. Mobile handoff** | `https://<space>.hf.space/?embedded=1#creds=<base64(CredsBundle)>`   | Skip shell; app boots directly, creds come via hash  |

#### Dispatch rule

```
if (URL.searchParams.has("embedded") && URL.hash.startsWith("#creds=")):
    boot embed  → import("./embed")
else:
    boot host   → import("@pollen-robotics/reachy-mini-sdk/host/auto").mountHost({...})
```

`?embedded=1` without creds is an invalid mode - the embed shows an
`ErrorView`.

#### `CredsBundle` (lives only in the URL hash, never in search)

The bundle has the shape:

```ts
{
  hfToken: string;       // short-lived HF bearer (15 min TTL)
  userName: string;
  robotPeerId: string;
  signalingUrl: string;
  theme: 'dark' | 'light';
  config: unknown | null;
  hostName: string;
  appName: string;
}
```

The hash is **never sent to a server**. The embed wipes it with
`history.replaceState` on the very first synchronous tick of
`connectToHost()`, **before any `await`** - see
[§13.5.2 Hash-only creds + immediate wipe](#1352-hash-only-creds--immediate-wipe).

#### Mode A standalone flow (the long story)

Phase machine inside the host:

```
booting → (signed-out | authenticated) → connecting → connected →
picking → handing-off → live → stopping → picking
```

- `booting`: wait for `window.ReachyMini`, instantiate the SDK,
  call `authenticate()`.
- `signed-out`: render the OAuth sign-in screen.
- `authenticated` → `connecting` → `connected` (SSE welcome) →
  `picking`.
- During `picking`, the robot list reacts live to the SDK's
  `robotsChanged` event.
- On robot selection, the host mounts the iframe at
  `<same-origin>?embedded=1#creds=<base64>` and overlays
  `ConnectingView` (3-step stepper: `link` → `session` → `wake`).
- When the embed reaches `phase: 'live'`, the overlay fades out.

Top bar layout while `live`:

```
[icon] [app name] ........ [robot status] [end-session] [oauth menu]
```

The top bar stays rendered through every phase of Mode A and does
**not** render at all in Mode B.

End-session flow:

1. Triggered by the End-session button, `embed:request-leave`, or
   `pagehide`.
2. Host → phase `stopping`, `LeavingView` overlay, posts
   `host:leaving`.
3. Embed runs `onLeave` callbacks → `reachy.stopSession()` → acks.
4. Host receives ack (or hits `timeoutMs`) → unmounts iframe →
   phase `picking`.

#### Mode B mobile handoff flow

The mobile app opens the Space in a WebView with a pre-built URL
containing creds in the hash. The dispatcher loads `./embed`
directly. **No host shell is mounted.** The user sees:

- No sign-in view (mobile already authenticated).
- No robot picker (mobile already picked).
- No welcome-back animation.
- No host top bar - if your app wants one, it draws it itself.

There is **no end-session button** in Mode B. Closing the WebView
triggers `pagehide`, which fires `onLeave` and stops the session.

### 13.4 Host phase state machine + handoff sequence

#### Host phase machine (Mode A only)

```mermaid
stateDiagram-v2
    [*] --> booting
    booting --> sdk_missing: SDK load timeout
    booting --> signed_out: authenticate() false
    booting --> authenticated: authenticate() true
    booting --> error: ctor / clientId error

    signed_out --> booting: user clicks Sign in
    authenticated --> connecting: auto
    connecting --> connected: SSE welcome
    connecting --> error: HTTP / network fail
    connected --> picking
    picking --> handing_off: selectRobot()
    handing_off --> live: embed reports phase=live
    handing_off --> error: embed:error { fatal: true }

    live --> stopping: endSession() OR embed:request-leave
    stopping --> picking: leave-ack OR timeout

    error --> picking: retry() if SDK authed
    error --> booting: retry() if no SDK
    error --> signed_out: retry() if auth expired
```

#### Handoff sequence (host → embed)

Showcases the single-SDK-per-tab invariant
([§13.5.1](#1351-single-live-sdk-per-tab)).

```mermaid
sequenceDiagram
    participant U as User
    participant H as Host
    participant HS as Host SDK
    participant C as Central
    participant E as Embed (iframe)
    participant ES as Embed SDK

    U->>H: click robot card
    H->>H: phase = handing-off, mount iframe
    E->>E: parse #creds, replaceState wipe
    E->>H: embed:ready
    H->>HS: disconnect() [§13.5.1]
    H->>E: host:init
    E->>ES: connect() → startSession() → ensureAwake()
    ES->>C: claim robot
    C-->>ES: session live
    E->>H: embed:app-state { phase: 'live' }
    H->>H: hide ConnectingView overlay
```

### 13.5 Engineering invariants

Four hard invariants. A failure here is a regression in the host
shell - it must produce a defined observable behaviour and must
stay covered by tests.

#### 13.5.1 Single live SDK per tab

**Contract**: at any instant, exactly one SDK instance per tab is
registered at the central.

**Mechanism**:
- Host mounts the SDK in `booting` and uses it for the picker.
- As soon as the embed posts `embed:ready`, the host calls
  `disconnect()` on its SDK and zeroes its references.
- On `leave-ack` (or timeout), the host calls `connect()` again to
  refresh the fleet and lands back on `picking`.

**Why**: removes the entire class of "Robot is busy" false
positives where the central thinks the shell still owns the robot.

**Defence in depth**: the host's SDK registers as `<appName> (shell)`
at the central; the embed keeps the clean `appName`. This protects
the narrow window where both SDKs overlap (between `embed:ready`
and `disconnect()`).

#### 13.5.2 Hash-only creds + immediate wipe

**Contract**: HF tokens never appear in URL search, never in
referer, never in HF Spaces access logs.

**Mechanism**:
- Creds are serialised as base64 JSON in the URL hash fragment
  (`#creds=...`).
- The embed wipes the hash with `history.replaceState` on the
  first synchronous tick of `connectToHost()`, before any `await`.
- On `host:leaving`, the embed clears `sessionStorage.hf_*` keys
  before sending the `leave-ack`.

**Token TTL**: `hf_token_expires` is set to **15 min** at seed
time. The SDK refreshes on demand.

#### 13.5.3 Bundle and SDK pinning

**Contract**: a fix in the host reaches every Space within one
cache cycle, with no per-app rebuild. A fix in the SDK can be
rolled out by the SDK team, not by every app team.

**Pinning rules**:
- App bundles (`index-<hash>.js`): hashed by Vite, cache-busted
  on deploy.
- `@pollen-robotics/reachy-mini-sdk` in `package.json`: pinned to
  an **exact prerelease build** (today: `1.8.0-rc1-main.fd4354c`),
  not a range. See [§10 SDK version pinning](#10-sdk-version-pinning).
- `@pollen-robotics/reachy-mini-sdk/host` subpath imports: same
  pin, same package.

On detected mismatch (e.g. unknown protocol version, structurally
invalid `host:init`), the host's `ErrorView` primary button calls
`window.location.reload(true)` to bypass any intermediary cache.

#### 13.5.4 React Strict Mode safety

**Contract**: every effect in the host package survives a double
mount in `<React.StrictMode>` without doubling network I/O, SDK
instances, or postMessage traffic.

**Why this matters**: React 18+ in dev intentionally mounts every
component, runs every effect, runs every cleanup, then mounts and
runs effects again. Code that "looked fine" in prod will fire two
parallel `connect()` calls, leak two SSE subscribers, post
`embed:ready` twice. In Mode A this surfaces as ghost sessions at
the central; in Mode B as two competing WebRTC peer connections.

**Mechanisms**:
- **Boot guard refs**: `useReachyHost()` uses a `bootStartedRef`
  set on first mount; the second StrictMode invocation early-
  returns instead of re-instantiating the SDK.
- **Subscription cleanup**: every `useEffect` returns its tear-
  down. `robotsChanged`, `phaseChanged`, `welcome` are all
  unsubscribed in cleanup.
- **Idempotent SDK calls**: `connect()` is a no-op when the SDK is
  already `connected` / `streaming`; the host relies on this rather
  than gating with a flag.
- **`connectToHost()` is one-shot**: a module-level
  `bootPromiseRef` returns the same promise for a second call,
  rather than re-running the handshake.

### 13.6 Protocol v1 messages

Full type definitions in
[`ts/host/src/lib/protocol.ts`](./host/src/lib/protocol.ts).
Envelopes are JSON, carry `source: 'reachy-mini'` and `version: 1`.
Both sides validate `event.origin` against `window.location.origin`
before trusting the payload.

| Direction       | Type                          | Purpose                                       |
|-----------------|-------------------------------|-----------------------------------------------|
| embed → host    | `embed:ready`                 | "I'm alive, send creds"                       |
| host  → embed   | `host:init`                   | Theme, signaling URL, hfToken, robotPeerId, config |
| embed → host    | `embed:app-state`             | Lifecycle phase + connecting sub-step         |
| host  → embed   | `host:theme-changed`          | Theme switched (no reload)                    |
| host  → embed   | `host:config-changed`         | Config updated (no reload, mobile-driven)     |
| host  → embed   | `host:leaving`                | Tear-down request with `timeoutMs`            |
| embed → host    | `embed:request-leave`         | App requests end-of-session                   |
| embed → host    | `embed:error`                 | Error report (`{ message, fatal, detail? }`)  |

Intentionally **not** in the v1 protocol:

- `embed:request-config-update` (apps don't push config upstream).
- `host:custom` / `embed:custom` (no free-form channel; new needs
  land as typed messages via a major bump).
- Any heartbeat / ping-pong messages.

#### Versioning policy

- `version: 1` is the only version today.
- **Additive changes** (new optional field, new message type) ship
  without a version bump.
- **Breaking changes** (removed field, changed semantics, removed
  message type) bump to `version: 2`. The host MAY support both
  versions for one release cycle, then drop v1.
- On unknown version, the receiver logs a warning and ignores the
  message. No negotiation handshake.

#### Idempotency

- `host:leaving` may arrive twice; the embed runs `onLeave`
  callbacks **once** (gated by a `pendingLeaveTokenRef`) and acks
  every time.
- `host:init` may arrive twice (rare: bridge re-arm); the embed
  treats the latest as authoritative and re-applies theme / config.

### 13.7 Non-goals

To stay simple and auditable, the host shell explicitly does NOT do:

- **App discovery / listing / catalog**. The shell renders exactly
  **one** app: the one it ships with. Listing apps lives in the
  Reachy Mini mobile app, fed by the website's `/api/js-apps`
  endpoint (filtered on `reachy_mini_js_app`).
- **"Official app" badging in the shell**. Officialness is derived
  from the Space ID prefix and surfaces only in the mobile catalog.
- **Multi-robot per tab**. One robot at a time per Space session.
- **Hot reload of the app code without reloading the iframe**.
  Code updates require unmounting + remounting.
- **App ↔ app communication**. Apps are sandboxed by design.
- **Offline mode**. The central is required for every session.
- **Automatic WebRTC retry**. On a session drop, the user goes
  back to the picker manually.
- **Queue or persistence of postMessage events** if the bridge is
  down. The bridge is best-effort; both sides re-converge on the
  next message.
- **Server-side rendering**. The host is a CSR shell, deliberately.
- **Cross-origin iframe**. The embed is same-origin with the host
  (both served by the Space); the origin check relies on this.
- **Imposing a framework on app authors**. Apps inside the iframe
  are completely free of framework constraints.

### 13.8 Threat model

The host runs in a HF Spaces container; the embed runs in a
same-origin iframe within the same Space.

| Asset                    | Threat                                   | Mitigation                                     |
|--------------------------|------------------------------------------|------------------------------------------------|
| HF bearer token          | Leak via URL log / referer               | Hash-only + immediate wipe (§13.5.2), 15 min TTL |
| HF bearer token          | Leak via sessionStorage to other origin  | Same-origin embed, no cross-origin postMessage |
| `config` payload         | Attacker controls URL → malformed JSON   | App MUST validate at the boundary (typed cast is not enough) |
| postMessage channel      | Random extension posts a forged message  | Receivers check `source === 'reachy-mini'` AND `event.origin === window.location.origin` |
| Central session          | Tab crashes, robot stays claimed         | `pagehide` triggers `stopSession()`; central also enforces idle timeout |

**Out of scope** for this iteration:
- Defence against a malicious app that the user explicitly loaded.
  We trust apps published under the `pollen-robotics/*` namespace.
- Defence against a compromised central. The signaling URL is
  configurable per-Space; the central is the trust root.

---

## 14. Robotics best practices

Subpath import: `@pollen-robotics/reachy-mini-sdk/animation`

Pure-TypeScript helpers that every Reachy Mini JS app used to copy-paste
from Rémi's `reachy-mini-js-practices` bench (the same source referenced
inline in `ts/animation/safe-return.ts`, `distance.ts`, `presets.ts`).
**No daemon changes** - everything routes through existing data-channel
commands (`setMotorMode`, `setTarget`, `gotoTarget`). Three concrete
callers today (`reachy_mini_emotions`, `reachy_mini_marionette` v1 + v2)
still carry their own copies; consolidate onto this lib for new apps.

See [`ts/animation/DESIGN.md`](./animation/DESIGN.md) for the two-phase
roadmap: Phase 1 (everything below) ships now; Phase 2 is the video-game-style
animation graph (named layers, masking, crossfades, procedural clips) - not
in this release.

### 14.1 Pose types: `Pose` vs `PartialPose`

```ts
import type { Pose, PartialPose } from "@pollen-robotics/reachy-mini-sdk/animation";

interface Pose         { head: number[]; antennas: [number, number]; body_yaw: number }
interface PartialPose  { head?: number[]; antennas?: [number, number] | number[]; body_yaw?: number }
```

- **`Pose`**: all three channels required. Use for hand-authored targets
  where you want the type system to nag if you forget a channel.
- **`PartialPose`**: any subset. Matches the SDK's `setTarget` /
  `gotoTarget` partial-update semantics, and is the shape of
  `reachy.robotState` (fields appear only after the daemon has emitted
  them).

Wire-format units, everywhere: `head` is a flat 16-float row-major 4×4
homogeneous matrix, `antennas` is `[right, left]` in **radians**,
`body_yaw` is a scalar in **radians**.

> Avoid carrying degrees around in your motion code. Convert at the UI
> boundary (`degToRad` / `radToDeg` from the SDK root) so everything
> below the boundary speaks wire units. Apps that drift between deg and
> rad ship subtle off-by-57 bugs.

### 14.2 Distance & scaled duration

The pure math behind "how big is this move" and "how long should it
take", computed **synchronously client-side** so apps can sync audio
cues, schedule streamer ticks, and live-tune constants without an extra
round-trip to the daemon.

```ts
import {
    distanceBetweenPoses,
    scaledDuration,
    DEFAULT_SCALED_DURATION_PRESET,
} from "@pollen-robotics/reachy-mini-sdk/animation";

const dist = distanceBetweenPoses(reachy.robotState, target);
// { head?: number /* magic-mm */,
//   antennas?: { right: number, left: number } /* deg */,
//   body_yaw?: number /* deg */ }

const plan = scaledDuration(reachy.robotState, target);
// { duration: number,
//   limiter: "head"|"antennaR"|"antennaL"|"body_yaw"|null,
//   perChannel: { head?, antennaR?, antennaL?, body_yaw? } }

reachy.gotoTarget({ ...target, duration: plan.duration });
```

The head metric is **magic-mm** - `translation_mm + rotation_deg` fused
into a single scalar that's monotonic with "how big does this move
feel". Mirror of the daemon's
`utils/interpolation.distance_between_poses`.

**Log `plan.limiter` in your motion paths.** It answers "why does the
home return take 1.2 s?" instantly without bisecting through three
channels by hand.

The default preset is calibrated on a real Reachy Mini (May 2026 bench):

| Field | Default | Rationale |
|---|---|---|
| `headSecPerMagicMm` | `0.015` | 0.02 reads as noticeably slow on the head. |
| `antennaSecPerDeg` | `0.005` | Antennas are light; PR [#952](https://github.com/pollen-robotics/reachy_mini/pull/952) resonance window penalises too-fast moves. |
| `bodyYawSecPerDeg` | `0.015` | Body sits between head (heavy) and antennas (light). |
| `minDurationSec` | `0.01` | Low floor so small in-app corrections feel snappy. |
| `maxDurationSec` | `1.5` | Matches the host shell's leave-protocol budget so `safelyReturnToPose` fits inside `onLeave`. |

Derive a custom preset by spreading (don't mutate - the constant is
deep-frozen):

```ts
const SLOWER_HEAD = {
    ...DEFAULT_SCALED_DURATION_PRESET,
    headSecPerMagicMm: 0.03,
};
const plan = scaledDuration(current, target, SLOWER_HEAD);
```

Edge case: when no channel overlaps between `current` and `target`,
`scaledDuration` returns `{ duration: minDurationSec, limiter: null,
perChannel: {} }`. The resulting `gotoTarget` is a safe no-op held over
the minimum dwell. Pass an explicit `duration` if you need a longer
dwell.

### 14.3 Safe return to home pose (`safelyReturnToPose`)

The canonical "return to safe rest" recipe in one call. Use it as your
`onLeave` body (see [§8 Cleaning up on leave](#8-cleaning-up-on-leave)):

```ts
import { safelyReturnToPose } from "@pollen-robotics/reachy-mini-sdk/animation";

handle.onLeave(() => safelyReturnToPose(handle.reachy));
```

What it does, in order:

1. `reachy.setMotorMode("enabled")` - safe since daemon PR
   [#1138](https://github.com/pollen-robotics/reachy_mini/pull/1138)
   (torque-on now pins all targets to the present pose before flipping).
2. Reads `reachy.robotState` for the current pose snapshot.
3. Computes `scaledDuration(current, target, preset)`.
4. Dispatches `reachy.gotoTarget({ ...target, duration })`.

Returns the `ScaledDurationResult` **synchronously after dispatching** -
does NOT await completion. If you need to await the motion finishing,
subscribe to the `state` event or `await sleep(plan.duration * 1000)`.

Default target is `INIT_POSE`:

```ts
import { INIT_POSE } from "@pollen-robotics/reachy-mini-sdk/animation";

// INIT_POSE.head     : identity 4×4 matrix (np.eye(4) in daemon parlance)
// INIT_POSE.antennas : [-0.1745, 0.1745]  ≈  ±10° outward (anti-resonance offset, PR #952)
// INIT_POSE.body_yaw : 0
```

Override for app-specific rest poses:

```ts
handle.onLeave(() =>
    safelyReturnToPose(handle.reachy, {
        target: { head: customHeadMatrix, antennas: [0, 0], body_yaw: 0 },
    })
);
```

**Safe to call when the data channel is closed.** Every underlying SDK
call swallows "channel closed" errors silently; the planned
`ScaledDurationResult` is still returned so you can log the move that
would have happened.

### 14.4 Standalone exit hooks (`installShutdownHandler`)

> **Host-shell apps: stop. Use `handle.onLeave()` from `connectToHost()`
> instead** - it integrates with the host's leave-protocol budget and
> avoids double-firing. Mixing the two dispatches `safelyReturnToPose`
> twice on close.

`installShutdownHandler` is for **standalone apps** (test benches,
custom dashboards, anything that doesn't go through `mountHost()`):

```ts
import { installShutdownHandler } from "@pollen-robotics/reachy-mini-sdk/animation";

const reachy = new ReachyMini({ appName: "my-bench" });
await reachy.authenticate();
await reachy.connect();
// ...pick robot, startSession...
installShutdownHandler(reachy);
```

What it does:

- Wires `pagehide` AND `beforeunload`. Both can fire on a real tab
  close, but in different scenarios - mobile Safari only fires
  `pagehide`. Wiring both is necessary for cross-browser coverage.
- Reentry-guards so `safelyReturnToPose` doesn't dispatch twice when
  both handlers fire on the same close.
- Defaults to `onlyWhenStreaming: true`: skips the goto when no session
  is live, so a stale tab where the user never picked a robot doesn't
  command anything on close.

### 14.5 Daemon parity warning

`INIT_POSE` and the magic-mm head coefficient mirror constants in the
**Python daemon**:

- `INIT_POSE.head` ↔ `Backend.INIT_HEAD_POSE` (`np.eye(4)`) in
  `src/reachy_mini/daemon/backend/abstract.py`.
- `INIT_POSE.antennas` ↔ `Backend.INIT_ANTENNAS_JOINT_POSITIONS`
  (`[-0.1745, 0.1745]`).
- The magic-mm head distance ↔
  `src/reachy_mini/daemon/utils/interpolation.py:distance_between_poses`.

**If a daemon PR changes any of these, the JS side must follow in the
same release.** `ts/animation/presets.ts` calls this out at the top of
each constant; respect it. Both `INIT_POSE` and
`DEFAULT_SCALED_DURATION_PRESET` are deep-frozen on the JS side so apps
can't accidentally drift via `INIT_POSE.head[0] = 0` (mutations throw in
strict mode, silently no-op otherwise).

### 14.6 Anti-patterns

| Don't | Do |
|---|---|
| Re-implement `distanceBetweenPoses` / `scaledDuration` in your app. | Import from `/animation`. The lib exists exactly because three apps did this and drifted. |
| Mix `installShutdownHandler` and `onLeave` in a host-shell app. | `onLeave` only. They both dispatch `safelyReturnToPose` and step on each other. |
| Call `reachy.stopSession()` inside your `onLeave`. | Let the host tear down. Do app-specific cleanup (return-to-pose, close audio, close sockets) and return. |
| Mutate `INIT_POSE` or `DEFAULT_SCALED_DURATION_PRESET`. | They're deep-frozen; spread to derive a variant. |
| `await safelyReturnToPose(...)` expecting the move to complete. | It resolves after **dispatch**, not after motion finishes. `await sleep(plan.duration * 1000)` or subscribe to `state` if you need to wait. |
| Carry degrees through your motion code. | Convert at the UI boundary; speak radians + magic-mm everywhere below it. |
| Pass `null` head / antennas / body_yaw to opt a channel out of a `gotoTarget`. | Use `PartialPose` and **omit** the channel. The SDK treats omission as "hold previous target". |
