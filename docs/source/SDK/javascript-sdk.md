# JavaScript SDK & Web Apps

Reachy Mini supports **full JavaScript web apps** that run entirely in the browser. No install, no server, no Python — just open a URL and control your robot from any device, including your phone.

## Why Web Apps?

The Python SDK is powerful but requires installation, GStreamer dependencies, and a capable machine. Web apps take a different approach:

- **Zero install** — open a link, you're in. Save disk space and setup time.
- **Cross-platform** — works on any device with a browser: laptop, tablet, phone.
- **Run from anywhere** — control your robot from the other side of the world.
- **Leverage device hardware** — use your phone's microphone, speakers, and touchscreen.
- **Instant sharing** — send someone a link, they can use the app immediately.

Web apps are deployed as **static Hugging Face Spaces** (`sdk: static`). There is no server-side code — the browser connects directly to the robot over WebRTC via a central signaling server.

> Python apps are not going away. Web apps are a complementary option, especially suited for lightweight control, remote access, and quick demos.

## Architecture

```
┌─────────────────────────────────┐
│  Browser                        │
│  (your app + reachy-mini.js)    │
└───────┬────────────┬────────────┘
        │ SSE/HTTP   │ WebRTC (peer-to-peer)
        │ signaling  │ video + audio + data
┌───────▼──────┐     │
│  Signaling   │     │
│  Server      │     │
│  (HF Space)  │     │
└───────┬──────┘     │
        │            │
┌───────▼────────────▼────────────┐
│  Robot                          │
│  GStreamer WebRTC daemon        │
│  camera · mic · motors          │
└─────────────────────────────────┘
```

1. **Your app** is a static HTML/JS page hosted on Hugging Face Spaces.
2. **reachy-mini.js** handles authentication, signaling, and WebRTC negotiation.
3. The **signaling server** relays SDP offers/answers and ICE candidates. It also validates Hugging Face OAuth tokens.
4. Once the WebRTC connection is established, **video, audio, and commands flow peer-to-peer** — the signaling server is no longer in the path.

## Quick Start

### 1. Create a Hugging Face Space

Create a new Space on [huggingface.co](https://huggingface.co/new-space) with `sdk: static`.

Your `README.md` front matter should look like:

```yaml
---
title: My Reachy Mini App
emoji: 🤖
sdk: static
pinned: false
hf_oauth: true
hf_oauth_expiration_minutes: 480
---
```

`hf_oauth: true` is required — it enables the Hugging Face login button that the signaling server uses for authentication.

### 2. Add the SDK

In your `index.html`, import the SDK as an ES module:

```html
<script type="module">
import { ReachyMini } from "./reachy-mini.js";

const robot = new ReachyMini();
</script>
```

You can grab `reachy-mini.js` from the [reference example](https://huggingface.co/spaces/cduss/webrtc_example) or from the npm CDN:

```js
import { ReachyMini } from "https://cdn.jsdelivr.net/npm/@anthropic-robotics/reachy-mini/+esm";
```

### 3. Connect to your robot

```js
// Authenticate with Hugging Face
if (!await robot.authenticate()) {
    robot.login();  // redirects to HF login page
    return;
}

// Connect to the signaling server
await robot.connect();

// Wait for robots to appear
robot.addEventListener("robotsChanged", (e) => {
    const robots = e.detail.robots;
    console.log("Available robots:", robots);
});

// Start a session with a specific robot
const detach = robot.attachVideo(document.querySelector("video"));
await robot.startSession(robotId);

// You're live — video is streaming, data channel is open
```

### 4. Control the robot

```js
// Move the head (roll, pitch, yaw in degrees)
robot.setHeadRpyDeg(0, 10, -5);

// Move the antennas (right, left in degrees)
robot.setAntennasDeg(30, -30);

// Rotate the body (yaw in degrees)
robot.setBodyYawDeg(15);

// Atomic raw-units update (single datachannel message; no XYZ loss)
robot.setTarget({
    head: rpyToMatrix(0, 10, -5).flat(),    // number[16] flat 4×4
    antennas: [degToRad(30), degToRad(-30)],
    body_yaw: degToRad(15),
});

// Play a sound file on the robot
robot.playSound("wake_up.wav");

// Send any JSON command via the data channel
robot.sendRaw({ my_custom_command: "hello" });
```

### 5. Receive robot state

```js
// Emitted every ~500ms while streaming. Wire-shape, raw units —
// use the exported math utilities for human conversions.
robot.addEventListener("state", (e) => {
    const { head, antennas, body_yaw, motor_mode, is_move_running } = e.detail;
    // head:            number[16]            — flat row-major 4×4
    // antennas:        [rightRad, leftRad]
    // body_yaw:        number                — radians
    // motor_mode:      "enabled" | "disabled" | "gravity_compensation"
    // is_move_running: boolean
    const rpy = matrixToRpy(head);   // { roll, pitch, yaw } in degrees
});
```

### 6. Audio

```js
// Unmute robot speaker (muted by default in browser)
robot.setAudioMuted(false);

// Unmute your microphone (bidirectional audio, if robot supports it)
robot.setMicMuted(false);

// Check if bidirectional audio is available
robot.addEventListener("micSupported", (e) => {
    console.log("Mic supported:", e.detail.supported);
});
```

### 7. Cleanup

```js
detach();                    // remove video binding
await robot.stopSession();   // back to 'connected' state
robot.disconnect();          // close signaling (keeps auth)
robot.logout();              // clear HF credentials
```

## API Reference

### Constructor

```js
new ReachyMini({
    signalingUrl: "https://cduss-reachy-mini-central.hf.space",  // default
    enableMicrophone: true,  // default — request mic on startSession()
})
```

### State Machine

```
'disconnected' ──connect()──▸ 'connected' ──startSession()──▸ 'streaming'
     ▴ disconnect()                ▴ stopSession()
     └─────────────────────────────┘
```

### Properties (read-only)

| Property | Type | Description |
| :--- | :--- | :--- |
| `state` | `string` | `"disconnected"`, `"connected"`, or `"streaming"` |
| `robots` | `Array` | Available robots: `[{ id, meta: { name } }]` |
| `robotState` | `Object` | Latest `state` event detail — `{ head: number[16], antennas: [rRad, lRad], body_yaw, motor_mode, is_move_running }` (wire shape) |
| `username` | `string\|null` | HF username after `authenticate()` |
| `isAuthenticated` | `boolean` | True if a valid HF token is available |
| `micSupported` | `boolean` | True if robot offers bidirectional audio |
| `micMuted` | `boolean` | Your microphone mute state |
| `audioMuted` | `boolean` | Robot speaker mute state (local only) |

### Methods

| Method | Returns | Description |
| :--- | :--- | :--- |
| `authenticate()` | `Promise<boolean>` | Check for existing HF OAuth token |
| `login()` | — | Redirect to HF login page |
| `connect()` | `Promise` | Open SSE connection, receive robot list |
| `startSession(robotId)` | `Promise` | Negotiate WebRTC, resolves when video + data ready |
| `stopSession()` | `Promise` | End session, back to `connected` |
| `disconnect()` | — | Close signaling (keeps auth) |
| `logout()` | — | Clear HF credentials |
| `attachVideo(videoEl)` | `() => void` | Bind video stream to element; returns cleanup function |
| `setTarget({ head?, antennas?, body_yaw? })` | `boolean` | Atomic raw-units update — `head` is `number[16]` (flat 4×4), `antennas` is `[rRad, lRad]`, `body_yaw` is radians |
| `setHeadRpyDeg(roll, pitch, yaw)` | `boolean` | Set head orientation in degrees (wraps `setTarget`) |
| `setAntennasDeg(right, left)` | `boolean` | Set antenna positions in degrees (wraps `setTarget`) |
| `setBodyYawDeg(yaw)` | `boolean` | Set body yaw in degrees (wraps `setTarget`) |
| `playSound(filename)` | `boolean` | Play a sound file on the robot |
| `sendRaw(data)` | `boolean` | Send arbitrary JSON via data channel |
| `requestState()` | `boolean` | Request a state snapshot |
| `setAudioMuted(muted)` | — | Mute/unmute robot speaker (local) |
| `setMicMuted(muted)` | — | Mute/unmute your microphone |

### Events

Use `robot.addEventListener(name, handler)` — the SDK extends `EventTarget`.

| Event | Detail | Description |
| :--- | :--- | :--- |
| `connected` | `{ peerId }` | Signaling connection established |
| `disconnected` | `{ reason }` | Signaling connection lost |
| `robotsChanged` | `{ robots }` | Robot list updated |
| `streaming` | `{ sessionId, robotId }` | WebRTC session active |
| `sessionStopped` | `{ reason }` | Session ended |
| `state` | `{ head, antennas, body_yaw, motor_mode, is_move_running }` | Robot state update (~500ms; wire shape — see "Receive robot state" above) |
| `videoTrack` | `{ track, stream }` | Video track available |
| `micSupported` | `{ supported }` | Bidirectional audio availability |
| `error` | `{ source, error }` | Error from `signaling`, `webrtc`, or `robot` |

### Math Utilities

```js
import { rpyToMatrix, matrixToRpy, degToRad, radToDeg } from "./reachy-mini.js";

rpyToMatrix(roll, pitch, yaw)  // degrees → 4×4 rotation matrix (ZYX)
matrixToRpy(matrix)            // 4×4 matrix → { roll, pitch, yaw } in degrees
```

## Security

- Authentication goes through Hugging Face OAuth — only users logged in to HF can access the signaling server.
- By default, you can only connect to robots registered under your own HF account.
- WebRTC connections are encrypted (DTLS/SRTP).

## Prerequisites

- Your robot must be running the wireless firmware and connected to the central signaling server.
- The robot must have a valid Hugging Face token configured (see [Usage](../platforms/reachy_mini/usage)).
- Currently supported on **wireless versions** only.

## Example

A full working example is available as a Hugging Face Space:
[cduss/webrtc_example](https://huggingface.co/spaces/cduss/webrtc_example)

It demonstrates video streaming, head/antenna control, bidirectional audio, and sound playback — all from a single static HTML page.
