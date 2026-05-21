# @pollen-robotics/reachy-mini-sdk

Browser SDK + host shell for controlling a [Reachy Mini](https://github.com/pollen-robotics/reachy_mini) robot over WebRTC.

This package ships two surfaces from a single npm install:

- **SDK** (`@pollen-robotics/reachy-mini-sdk`) - the low-level `ReachyMini` class for direct robot control over WebRTC. Full API reference in the JSDoc header of [`reachy-mini-sdk.js`](./reachy-mini-sdk.js).
- **Host** (`@pollen-robotics/reachy-mini-sdk/host*`) - the shell rendered around every Reachy Mini app deployed as a Hugging Face Space: OAuth, robot picker, session lifecycle, and the parent/iframe postMessage bridge. See [`host/APP_AUTHOR_GUIDE.md`](./host/APP_AUTHOR_GUIDE.md) for the full integration guide and [`host/SPEC.md`](./host/SPEC.md) for the behavioural contract.

## Install

### With a bundler / Node

```bash
npm install @pollen-robotics/reachy-mini-sdk
```

```js
import { ReachyMini } from "@pollen-robotics/reachy-mini-sdk";
```

### From a browser, no build step (Hugging Face Spaces, static hosting…)

Use an ESM CDN. The `+esm` suffix tells jsdelivr to resolve bare specifiers (the `@huggingface/hub` dependency) recursively, so the import just works:

```js
import { ReachyMini } from "https://cdn.jsdelivr.net/npm/@pollen-robotics/reachy-mini-sdk@1.7.3/+esm";
```

Or via esm.sh:

```js
import { ReachyMini } from "https://esm.sh/@pollen-robotics/reachy-mini-sdk@1.7.3";
```

## Quick start

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

## Host shell (apps deployed as Hugging Face Spaces)

If you're building a Reachy Mini app and want the prebuilt OAuth + robot picker + session shell, import from the `/host*` subpaths instead of writing the plumbing yourself:

```ts
// src/host.ts — mounts the shell around your app's iframe
import { mountHost } from '@pollen-robotics/reachy-mini-sdk/host/auto';

mountHost({
  appName: 'My App',
  appIconUrl: '/icon.svg',
  enableMicrophone: false,
});
```

```ts
// src/embed.ts — runs inside the iframe, receives a connected SDK instance
import { connectToHost } from '@pollen-robotics/reachy-mini-sdk/host/embed';

const handle = await connectToHost();
handle.reachy.setHeadRpyDeg(0, 10, 0);
```

Same as the SDK, both entries are CDN-loadable for build-less Spaces:

```ts
import('https://cdn.jsdelivr.net/npm/@pollen-robotics/reachy-mini-sdk@1/host/dist/entry/auto.js');
import('https://cdn.jsdelivr.net/npm/@pollen-robotics/reachy-mini-sdk@1/host/dist/entry/embed.js');
```

The complete step-by-step recipe (index.html, dispatch.ts, Dockerfile, deploying to HF Spaces, FAQ) lives in [`host/APP_AUTHOR_GUIDE.md`](./host/APP_AUTHOR_GUIDE.md).

## License

Apache-2.0 — see [LICENSE](./LICENSE).
