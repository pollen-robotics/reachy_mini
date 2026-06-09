# Host shell

> Shell + iframe bridge for [Reachy Mini](https://huggingface.co/pollen-robotics) web apps.
> Handles Hugging Face OAuth, robot selection, session lifecycle, and the parent / iframe
> postMessage protocol so app authors only ship their app, not the plumbing.

[![npm version](https://img.shields.io/npm/v/@pollen-robotics/reachy-mini-sdk.svg)](https://www.npmjs.com/package/@pollen-robotics/reachy-mini-sdk)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](../LICENSE)

---

## What is this?

Subpath exports of [`@pollen-robotics/reachy-mini-sdk`](https://www.npmjs.com/package/@pollen-robotics/reachy-mini-sdk) that power every Reachy Mini app deployed as a
[Hugging Face Space](https://huggingface.co/docs/hub/spaces-overview). It exposes:

- **`mountHost()`** — the shell rendered around your app's iframe. Renders the
  top bar, sign-in screen, robot picker, and connecting / leaving overlays.
- **`connectToHost()`** — the vanilla-JS client that runs inside the iframe and
  hands your app a fully-connected `ReachyMini` SDK instance.

Two boot modes are supported (cf. [`APP_CREATION_GUIDE.md` §13.3](../APP_CREATION_GUIDE.md#133-two-boot-modes-one-url-surface)):

| Mode | Entry point | Use case |
|------|-------------|----------|
| **A. Standalone** | `mountHost()` in `dispatch.ts` | A user opens the Space directly in a browser tab. Full OAuth → picker → iframe sequence. |
| **B. Mobile handoff** | `connectToHost()` in `embed.ts` | The Reachy Mini mobile app launches the app inside a webview with credentials pre-injected via the URL hash. No shell, no picker. |

The same app code works in both modes; only the entry point differs.

## Documentation

| Document | Audience | Read it when… |
|----------|----------|----------------|
| **[`../APP_CREATION_GUIDE.md`](../APP_CREATION_GUIDE.md)** | app authors **and** host maintainers | Single source of truth: scaffold, `sdk: static` deploy, host ↔ embed contract, invariants, protocol v1. Today's SDK pin: `1.8.0`. |

App authors and library maintainers both start with the
**[App Creation Guide](../APP_CREATION_GUIDE.md)**: §1-§12 are the
app-author recipe, §13 is the host ↔ embed architecture reference.

## Installation

Most app authors load the package from a CDN at runtime; npm install is only
needed for TypeScript types and local dev:

```bash
npm install --save-dev @pollen-robotics/reachy-mini-sdk
```

```html
<!-- Loaded at runtime by every Reachy Mini app's index.html.
     Both bundles auto-install the SDK on window.ReachyMini at
     load time, so no separate `<script>` for the SDK is needed. -->
<script type="module"
  src="https://cdn.jsdelivr.net/npm/@pollen-robotics/reachy-mini-sdk@1/host/dist/entry/auto.js">
</script>
<script type="module"
  src="https://cdn.jsdelivr.net/npm/@pollen-robotics/reachy-mini-sdk@1/host/dist/entry/embed.js">
</script>
```

The CDN URL pins to the **major** version (`@1`), so patch + minor releases
land in every Space automatically; only a deliberate breaking change requires
each app to update its tag. The host shares its version with the SDK and the
daemon (single repo, single release, single npm package).

## 90-second tour

A complete Reachy Mini app, in 4 files:

**`index.html`** — picks the entry script based on `?embedded=1`:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>My App</title>
    <script src="https://reachy-mini-js.example/sdk@v1.0.0.js"></script>
    <script type="module" src="/src/dispatch.ts"></script>
  </head>
  <body><div id="root"></div></body>
</html>
```

**`src/dispatch.ts`** — routes to host or embed:

```ts
const isEmbed = new URLSearchParams(window.location.search).get('embedded') === '1';
if (isEmbed) {
  await import('./embed');
} else {
  await import('./host');
}
```

**`src/host.ts`** — mounts the shell:

```ts
import { mountHost } from '@pollen-robotics/reachy-mini-sdk/host/auto';

mountHost({
  appName: 'My App',
  appIconUrl: '/icon.svg',
  enableMicrophone: false,
});
```

**`src/embed.ts`** — connects and runs the app:

```ts
import { connectToHost } from '@pollen-robotics/reachy-mini-sdk/host/embed';

const handle = await connectToHost();
handle.reachy.setHeadRpyDeg(0, 10, 0);
handle.onLeave(() => {
  console.log('Session ending, cleaning up.');
});
```

That's the whole contract. The host does OAuth, picks the robot, mounts the
iframe; the embed wakes the robot and drives it. Everything else is your
app's logic. See the [App Creation Guide](../APP_CREATION_GUIDE.md)
for the detailed walk-through and reference apps.

## Reference apps

Three open-source apps live alongside this package and are kept in lockstep
with every release:

| App | Stack | What it shows |
|-----|-------|---------------|
| [`pollen-robotics/reachy_mini_minimal_conversation`](https://huggingface.co/spaces/pollen-robotics/reachy_mini_minimal_conversation) | Vanilla TypeScript | Smallest possible app. Tech-freedom proof. |
| [`pollen-robotics/reachy_mini_emotions`](https://huggingface.co/spaces/pollen-robotics/reachy_mini_emotions) | React + MUI | Plutchik emotion wheel + dance triggers. |
| [`pollen-robotics/reachy_mini_telepresence`](https://huggingface.co/spaces/pollen-robotics/reachy_mini_telepresence) | React + MUI | Live video + head / body teleop. |

App authors are free to use any UI framework they want inside the iframe; the
host doesn't care. This is a hard design rule, not an accident
(see [`APP_CREATION_GUIDE.md` §13.1](../APP_CREATION_GUIDE.md#131-roles-app--host--embed)
"Why React + MUI for the host (and only the host)").

## Versioning

The host ships inside `@pollen-robotics/reachy-mini-sdk` and shares its
version with the SDK and the Reachy Mini Python daemon (a single source of
truth, enforced by the npm publish CI). The major version of the **wire
protocol** is tracked separately in `PROTOCOL_VERSION` and bumped only on
incompatible postMessage changes
(see [`APP_CREATION_GUIDE.md` §13.6](../APP_CREATION_GUIDE.md#136-protocol-v1-messages)).

App authors should **pin to the exact version that the reference
apps use** - today `1.8.0`, see
[`APP_CREATION_GUIDE.md` §10](../APP_CREATION_GUIDE.md#10-sdk-version-pinning).

## License

Apache-2.0 - see [LICENSE](../LICENSE).
