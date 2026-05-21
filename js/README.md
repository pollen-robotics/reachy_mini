# @pollen-robotics/reachy-mini

Browser SDK for controlling a [Reachy Mini](https://github.com/pollen-robotics/reachy_mini) robot over WebRTC.

The full API reference lives in the JSDoc header of [`reachy-mini.js`](./reachy-mini.js): constructor options, read-only properties, the `disconnected → connected → streaming` state machine, the event list, and every command helper.

## Install

### With a bundler / Node

```bash
npm install @pollen-robotics/reachy-mini
```

```js
import { ReachyMini } from "@pollen-robotics/reachy-mini";
```

### From a browser, no build step (Hugging Face Spaces, static hosting…)

Use an ESM CDN. The `+esm` suffix tells jsdelivr to resolve bare specifiers (the `@huggingface/hub` dependency) recursively, so the import just works:

```js
import { ReachyMini } from "https://cdn.jsdelivr.net/npm/@pollen-robotics/reachy-mini@1.7.3/+esm";
```

Or via esm.sh:

```js
import { ReachyMini } from "https://esm.sh/@pollen-robotics/reachy-mini@1.7.3";
```

## Quick start

```js
import { ReachyMini } from "@pollen-robotics/reachy-mini";

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

See the JSDoc header in [`reachy-mini.js`](./reachy-mini.js) for the full surface (events, raw `setTarget`, audio controls, embedded-mode auto-start, etc.).

## License

Apache-2.0 — see [LICENSE](./LICENSE).
