# @pollen-robotics/reachy-mini-sdk

The JS package for [Reachy Mini](https://github.com/pollen-robotics/reachy_mini). It ships:

- The **browser SDK** (`ReachyMini` class) for direct WebRTC control — the default export.
- An optional **host shell** (OAuth + robot picker + iframe bridge) for apps deployed as Hugging Face Spaces, exposed under the `./host*` subpaths.

Both used to be separate npm packages (`@pollen-robotics/reachy-mini-sdk` + `@pollen-robotics/reachy-mini-host`); they were merged into a single entry point so app authors only install, version, and import from one place.

The full SDK API reference lives in the JSDoc header of [`reachy-mini-sdk.js`](./reachy-mini-sdk.js): constructor options, read-only properties, the `disconnected → connected → streaming` state machine, the event list, and every command helper.

## Install

### With a bundler / Node

```bash
npm install @pollen-robotics/reachy-mini-sdk
```

```js
// Low-level robot control
import { ReachyMini } from "@pollen-robotics/reachy-mini-sdk";

// Host shell for Hugging Face Spaces apps (OAuth + picker + iframe)
import { mountHost } from "@pollen-robotics/reachy-mini-sdk/host";
import { connectToHost } from "@pollen-robotics/reachy-mini-sdk/host/embed";
import { PROTOCOL_VERSION } from "@pollen-robotics/reachy-mini-sdk/host/protocol";
```

### From a browser, no build step (Hugging Face Spaces, static hosting…)

Use an ESM CDN. The `+esm` suffix tells jsdelivr to resolve bare specifiers (the `@huggingface/hub` dependency) recursively, so the import just works:

```js
import { ReachyMini } from "https://cdn.jsdelivr.net/npm/@pollen-robotics/reachy-mini-sdk@1/+esm";
```

Or via esm.sh:

```js
import { ReachyMini } from "https://esm.sh/@pollen-robotics/reachy-mini-sdk@1";
```

Track bleeding-edge from `main`:

```js
import { ReachyMini } from "https://cdn.jsdelivr.net/npm/@pollen-robotics/reachy-mini-sdk@main/+esm";
```

The host CDN bundles live under the same package:

```html
<!-- Standalone host shell entry -->
<script type="module"
  src="https://cdn.jsdelivr.net/npm/@pollen-robotics/reachy-mini-sdk@1/host/dist/entry/auto.js">
</script>

<!-- Embed-side client (inside the iframe) -->
<script type="module"
  src="https://cdn.jsdelivr.net/npm/@pollen-robotics/reachy-mini-sdk@1/host/dist/entry/embed.js">
</script>
```

Both bundles auto-install the SDK on `window.ReachyMini` at load time, so an app that uses the host no longer needs a separate `<script type="module">` to expose the SDK.

## Quick start (SDK only)

```js
import { ReachyMini } from "@pollen-robotics/reachy-mini-sdk";

const robot = new ReachyMini();

// 1. Auth (HuggingFace OAuth — required by the signaling server)
if (!await robot.authenticate()) { robot.login(); return; }

// 2. Connect to the signaling server
await robot.connect();

// 3. Pick a robot once the list arrives
robot.addEventListener("robotsChanged", (e) => {
    const robots = e.detail.robots;  // [{ id, meta: { name } }, ...]
});

// 4. Start a WebRTC session
const detach = robot.attachVideo(document.querySelector("video"));
await robot.startSession(robotId);

// 5. Send commands — degree-friendly helpers
robot.setHeadRpyDeg(0, 10, -5);
robot.setAntennasDeg(30, -30);
robot.setBodyYawDeg(15);
robot.playSound("wake_up.wav");
```

See the JSDoc header in [`reachy-mini-sdk.js`](./reachy-mini-sdk.js) for the full surface (events, raw `setTarget`, audio controls, embedded-mode auto-start, etc.).

## Host shell

For Hugging Face Spaces apps that need OAuth + a robot picker + iframe lifecycle management:

- [`host/README.md`](./host/README.md) — quickstart and integration overview
- [`host/APP_AUTHOR_GUIDE.md`](./host/APP_AUTHOR_GUIDE.md) — full guide for app authors
- [`host/SPEC.md`](./host/SPEC.md) — host ↔ embed protocol spec

## Migration from `@pollen-robotics/reachy-mini-host`

```diff
- import { mountHost }      from '@pollen-robotics/reachy-mini-host/auto';
- import { connectToHost }  from '@pollen-robotics/reachy-mini-host/embed';
- import { PROTOCOL_VERSION } from '@pollen-robotics/reachy-mini-host/protocol';
+ import { mountHost }      from '@pollen-robotics/reachy-mini-sdk/host/auto';
+ import { connectToHost }  from '@pollen-robotics/reachy-mini-sdk/host/embed';
+ import { PROTOCOL_VERSION } from '@pollen-robotics/reachy-mini-sdk/host/protocol';
```

You can drop `@pollen-robotics/reachy-mini-host` from your `package.json`. The host CDN bundles also moved — adjust the jsdelivr URL in your `index.html`:

```diff
- https://cdn.jsdelivr.net/npm/@pollen-robotics/reachy-mini-host@1/dist/entry/auto.js
+ https://cdn.jsdelivr.net/npm/@pollen-robotics/reachy-mini-sdk@1/host/dist/entry/auto.js
```

The host bundles now import the SDK directly and assign it on `window.ReachyMini` at load time, so apps no longer need a second `<script type="module">` for the SDK alone.

See the [CHANGELOG](./CHANGELOG.md) for the full release history.

## License

Apache-2.0 — see [LICENSE](./LICENSE).
