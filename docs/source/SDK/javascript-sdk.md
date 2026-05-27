# JavaScript SDK runtime reference

> **Building a Reachy Mini JS app?** The single source of truth is
> [`ts/APP_CREATION_GUIDE.md`](../../../ts/APP_CREATION_GUIDE.md) at
> the repo root. It covers scaffolding, `public/icon.svg`, the host
> shell, `sdk: static` (default) / `sdk: docker` (fallback) deploy,
> `mountHost()` / `connectToHost()` API, local dev, FAQ, and the host
> ↔ embed contract. **Pin the SDK to
> `@pollen-robotics/reachy-mini-sdk@1.8.0-rc1-main.fd4354c`** (the
> version validated against the host shell + daemon).
>
> **This file** is the runtime API surface of the `ReachyMini` class
> you receive from `handle.reachy` once `connectToHost()` resolves:
> methods, events, properties, state machine, and the daemon-side
> recorded-move playback API. Bookmark it after you've shipped a
> first app from the guide.

Reachy Mini ships a browser SDK that drives a robot over WebRTC.
The npm package `@pollen-robotics/reachy-mini-sdk` exposes:

- The `ReachyMini` class (the SDK runtime documented below).
- The host shell + embed adapter under the `./host*` subpath
  exports (`./host`, `./host/auto`, `./host/embed`, `./host/protocol`).
  See [`ts/APP_CREATION_GUIDE.md`](../../../ts/APP_CREATION_GUIDE.md)
  for the integration recipe.

## Architecture

```
┌─────────────────────────────────┐
│  Browser                        │
│  (your app + reachy-mini-sdk.js)│
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

1. Your app is a static HTML/JS page hosted on Hugging Face Spaces.
2. The SDK handles authentication, signaling, and WebRTC negotiation.
3. The signaling server relays SDP offers/answers and ICE candidates
   and validates Hugging Face OAuth tokens.
4. Once the WebRTC connection is established, video, audio, and
   commands flow peer-to-peer; the signaling server is no longer in
   the path.

When you use the host shell (`mountHost()` + `connectToHost()`,
documented in [`ts/APP_CREATION_GUIDE.md`](../../../ts/APP_CREATION_GUIDE.md)),
the steps below are handled for you. The class-level API documented
here is what you use **after** `connectToHost()` resolves, or what
you call directly if you opted out of the host shell.

## API Reference

### Constructor

```js
new ReachyMini({
    signalingUrl: "https://pollen-robotics-reachy-mini-central.hf.space",  // default
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
| `playMove(motion, opts?)` | `Promise<{finished?, cancelled?, error?, has_audio?}>` | Upload + play a recorded move (optionally with audio) on the daemon's local clock; resolves when playback ends — see [Daemon-side recorded-move playback](#daemon-side-recorded-move-playback) |
| `cancelMove()` | `boolean` | Cancel an in-flight `playMove` |
| `uploadAudio(blob, opts?)` | `Promise<string>` | Upload a standalone audio slot, returns `uploadId` — pair with `playUploadedAudio` for record-time sync |
| `playUploadedAudio(uploadId, opts?)` | `Promise<{started: true, ...}>` | Trigger daemon-side standalone audio playback; resolves on the daemon's `started` broadcast (use as a sync anchor) |
| `cancelAudio()` | `boolean` | Cancel an in-flight `playUploadedAudio` |

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
import { rpyToMatrix, matrixToRpy, degToRad, radToDeg } from "@pollen-robotics/reachy-mini-sdk";

rpyToMatrix(roll, pitch, yaw)  // degrees → 4×4 rotation matrix (ZYX)
matrixToRpy(matrix)            // 4×4 matrix → { roll, pitch, yaw } in degrees
```

## Daemon-side recorded-move playback

Long recorded moves (and any move with audio) should play **server-side on the daemon's local clock**, not by streaming `set_target` frames from the browser. The browser uploads the move once over the WebRTC data channel and the daemon ticks the inner loop at the requested frequency — no per-frame round-trip, smooth on wireless robots. When audio is attached the daemon plays it on the same GStreamer pipeline, so motion and audio share a single clock (no cross-network drift).

### Combined motion + audio

```js
const result = await robot.playMove(motion, {
    audioBlob,                    // optional, 16 kHz mono PCM WAV
    audioLeadMs: -100,            // system-wide default
    description: "happy wave",
    onProgress: (p) => console.log(p.phase, p.sent, p.total),
    onStarted: ({ duration_s, has_audio }) => { /* sync anchor */ },
});
// result is { finished: true } | { cancelled: true } | { error: "..." }

// Cancel at any time from another code path:
robot.cancelMove();
```

`motion` is the shape the Python `RecordedMove` parser expects:
```js
{ time: [0, 0.01, 0.02, …], set_target_data: [{ head, antennas, body_yaw }, …] }
```

`audioLeadMs` shifts audio relative to motion at the daemon:
- **Positive** — audio fires N ms BEFORE motion (compensates motor pickup).
- **Negative** — motion fires N ms BEFORE audio (compensates GStreamer playbin warmup).
- **Default `-100`** is the empirical system-wide constant (combined motor + pipeline). Tune only after measuring.

The encoded wire form defaults to `gzip+base64` (typically ~3× smaller for recorded-move JSON). Falls back to plain JSON if the browser lacks `CompressionStream`.

### Record-time audio (sync anchor)

For recording flows that want the SAME audio pipeline at capture AND replay (so pipeline latency cancels out and one `audioLeadMs` works for all recordings):

```js
// 1. During the countdown — upload the source audio.
const audioId = await robot.uploadAudio(audioBlob, { description: "song" });

// 2. At the GO! moment — kick off daemon-side playback, await the
//    started broadcast, then start motion capture.
await robot.playUploadedAudio(audioId);
const captureT0 = performance.now();
startMyMotionCapture();

// 3. On stop / cancel / restart — stop the audio.
robot.cancelAudio();
```

The daemon does NOT emit a `finished` event for standalone audio; callers know the duration from the WAV header and call `cancelAudio()` when done.

### Audio format

Audio must be canonical **16 kHz mono 16-bit PCM WAV**. Apps are responsible for normalizing before upload — the daemon does not transcode. Format mismatch is a frequent cause of "audio is silent / wrong speed" on inherited datasets.

### Backpressure & cancellation

`playMove` and `uploadAudio` pace chunk sends on the data channel's `bufferedAmount` so multi-megabyte uploads (a 3-min song's WAV is ~6 MB base64) don't degrade other channels on the same peer connection. There's no separate `pause` — to stop a long upload mid-way, close the session.

## Security

- Authentication goes through Hugging Face OAuth — only users logged in to HF can access the signaling server.
- By default, you can only connect to robots registered under your own HF account.
- WebRTC connections are encrypted (DTLS/SRTP).

## Prerequisites

- Your robot must be running the wireless firmware and connected to the central signaling server.
- The robot must have a valid Hugging Face token configured (see [Usage](../platforms/reachy_mini/usage)).
- Currently supported on **wireless versions** only.

## Working examples

The three reference apps maintained alongside the SDK are the canonical worked examples. They all use the host shell pattern and the current SDK pin:

- [`pollen-robotics/reachy_mini_minimal_conversation`](https://huggingface.co/spaces/pollen-robotics/reachy_mini_minimal_conversation) — vanilla TS + Vite.
- [`pollen-robotics/reachy_mini_emotions`](https://huggingface.co/spaces/pollen-robotics/reachy_mini_emotions) — React 19 + MUI 7 + Vite.
- [`pollen-robotics/reachy_mini_telepresence`](https://huggingface.co/spaces/pollen-robotics/reachy_mini_telepresence) — React 19 + MUI 7 + Vite with camera + media streams.

Clone the closest one and trim. See [`ts/APP_CREATION_GUIDE.md`](../../../ts/APP_CREATION_GUIDE.md) for the step-by-step.
