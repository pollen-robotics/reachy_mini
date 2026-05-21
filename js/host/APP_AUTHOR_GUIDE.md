# Build a Reachy Mini app

How to ship a Hugging Face Space that runs on a Reachy Mini robot,
using `@pollen-robotics/reachy-mini-sdk/host` for OAuth, robot picking, session
lifecycle, and a top bar - so your code stays focused on **your
app's UI** and nothing else.

| Doc        | Purpose                              | Audience                          |
|------------|--------------------------------------|-----------------------------------|
| README     | Quick tour of the package            | First-time visitors               |
| **This**   | **Step-by-step app author guide**    | **You, building a new app**       |
| SPEC.md    | Behavioural contract + invariants    | Maintainers, reviewers            |

## Table of contents

1. [What you get for free](#1-what-you-get-for-free)
2. [Quickstart: clone a reference app](#2-quickstart-clone-a-reference-app)
3. [The 4-file contract](#3-the-4-file-contract)
4. [`mountHost()` API](#4-mounthost-api)
5. [`connectToHost()` API](#5-connecttohost-api)
6. [Visual identity: icon, name, emoji](#6-visual-identity-icon-name-emoji)
7. [Receiving an external config (deep-link, mobile)](#7-receiving-an-external-config)
8. [Cleaning up on leave](#8-cleaning-up-on-leave)
9. [Local dev: HF token vs OAuth redirect](#9-local-dev)
10. [Deploying to Hugging Face Spaces](#10-deploying-to-hugging-face-spaces)
11. [FAQ and common pitfalls](#11-faq-and-common-pitfalls)

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

- `index.html` + `Dockerfile` (boilerplate, ~10 lines each).
- `src/dispatch.ts` (~20 lines, picks shell vs embed mode).
- `src/embed.{ts,tsx}` - **your app's actual code**. You receive a
  live `ReachyMini` SDK handle and render whatever you want.

You can use **any framework** inside your `embed` entry: React,
Svelte, Vue, vanilla TS. The host runs outside your iframe and
doesn't care.

---

## 2. Quickstart: clone a reference app

Pick the reference closest to your needs and clone its `index.html`
+ `dispatch.ts` + `Dockerfile`:

| Reference app                       | Stack                       | Use it for                                  |
|-------------------------------------|-----------------------------|---------------------------------------------|
| `reachy_mini_emotions/`             | React 19 + MUI 7 + Vite     | UI-rich app with rich components / theming  |
| `reachy_mini_telepresence/`         | React 19 + MUI 7 + Vite     | App with camera / media streams             |
| `reachy_mini_minimal_conversation/` | **Vanilla TS + Vite**       | Smallest runtime, no framework              |

```bash
# Example: start from the vanilla TS template
cp -r reachy_mini_minimal_conversation/ my_new_app/
cd my_new_app
# Edit package.json `name`, README frontmatter (`title`, `emoji`),
# and src/embed.ts to your app.
npm install
npm run dev
# → http://localhost:5173
```

---

## 3. The 4-file contract

These four files are the entire integration surface. Don't add
mandatory files outside this list - keep the apps interchangeable.

### 3.1 `index.html`

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>My App</title>
    <link rel="icon" href="/icon.svg" type="image/svg+xml" />

    <!-- Theme bootstrap: paints the right palette before CSS lands -->
    <script>
      (function () {
        var params = new URLSearchParams(window.location.search);
        var raw = params.get("theme");
        var mode = (raw === "dark" || raw === "light")
          ? raw
          : (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
        document.documentElement.setAttribute("data-theme", mode);
        document.documentElement.style.backgroundColor = mode === "dark" ? "#101013" : "#fafafa";
      })();
    </script>

    <!-- HF Spaces injects OAuth client ID at container start -->
    <script>
      (function () {
        var clientId = "__OAUTH_CLIENT_ID__";
        if (clientId && clientId.indexOf("__") !== 0) {
          window.huggingface = { variables: { OAUTH_CLIENT_ID: clientId } };
        }
      })();
    </script>

    <!-- ReachyMini SDK, pinned to a MAJOR version (e.g. @1) for auto-patch
         updates, or to a TAG / SHA via `cdn.jsdelivr.net/gh/...` if you
         need bleeding-edge. The `+esm` suffix resolves the bare
         @huggingface/hub dependency recursively. -->
    <script type="module">
      import { ReachyMini } from "https://cdn.jsdelivr.net/npm/@pollen-robotics/reachy-mini-sdk@1/+esm";
      window.ReachyMini = ReachyMini;
      window.dispatchEvent(new Event("reachymini:ready"));
    </script>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/dispatch.ts"></script>
  </body>
</html>
```

### 3.2 `src/dispatch.ts`

```ts
const params = new URLSearchParams(window.location.search);

if (params.get('embedded') === '1') {
  void import('./embed');
} else {
  void import('@pollen-robotics/reachy-mini-sdk/host/auto').then(({ mountHost }) => {
    mountHost({
      appName: 'My App',
      appIconUrl: '/icon.svg',
      enableMicrophone: false,
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

### 3.4 `Dockerfile`

Standard nginx + node build. Copy from `reachy_mini_emotions/Dockerfile`
verbatim; the only thing to customise is the app name in comments.

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
   sdk: docker
   app_port: 7860
   pinned: false
   hf_oauth: true
   short_description: One-line description shown in the catalog.
   tags:
     - reachy_mini
     - reachy_mini_js_app
   ---
   ```

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
     inject the ID at container start.

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
handle.onLeave(async () => {
  player.cancel();        // stop streaming motion frames
  audioCtx?.close();      // release audio
  ws?.close();            // close any side channels
  await flushTelemetry(); // your async hooks
});
```

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

## 10. Deploying to Hugging Face Spaces

The reference apps' `Dockerfile` is ready to copy. Steps:

```bash
# 1. Init the Space repo
huggingface-cli repo create my-app --type space --space_sdk docker

# 2. Push
git remote add space git@hf.co:spaces/<your-username>/my-app
git push space main
```

HF Spaces will:

- Build your `Dockerfile` (the multi-stage `node:20-alpine` → `nginx:1.27-alpine` setup).
- At container start, run `docker-entrypoint.d/10-inject-hf-vars.sh`
  to patch `index.html` with the real OAuth client ID, space host,
  and space ID.
- Serve nginx on `app_port` (7860).

### Required Space frontmatter

```yaml
sdk: docker
app_port: 7860
hf_oauth: true
tags:
  - reachy_mini
  - reachy_mini_js_app   # critical: mobile app discovery
```

### Cache busting

If after a push your Space still serves the old bundle, push an
empty commit:

```bash
git commit --allow-empty -m "chore: bust HF Spaces cache"
git push space main
```

---

## 11. FAQ and common pitfalls

### "I see a `Robot is busy` error even though no one is using the robot"

The host's SDK and the embed's SDK both claim a peer at the
central. The host **must** disconnect when the embed boots; if it
doesn't, the central sees two peers with the same `appName` and
rejects the embed.

This is handled automatically by `@pollen-robotics/reachy-mini-sdk/host` (§8.1 of the
SPEC). If you see this in dev, you likely have **two tabs** open
on the same Space - that's expected behaviour.

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

### "How do I test the mobile-handoff mode locally?"

Hit your dev server at:

```
http://localhost:5173/?embedded=1#creds=<base64({"hfToken":"hf_xxx","userName":"you","robotPeerId":"abc","signalingUrl":"https://...","theme":"dark","config":null,"hostName":"Reachy Mini","appName":"My App"})>
```

The dispatcher will skip the shell and go straight to your embed.
Useful for testing the embed path without spinning up the mobile
app.
