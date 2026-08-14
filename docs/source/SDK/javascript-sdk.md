# JavaScript SDK runtime reference

> **Building a Reachy Mini JS app?** The single source of truth is
> [`ts/APP_CREATION_GUIDE.md`](../../../ts/APP_CREATION_GUIDE.md) at
> the repo root. It covers scaffolding, `public/icon.svg`, the host
> shell, `sdk: static` deploy,
> `mountHost()` / `connectToHost()` API, local dev, FAQ, and the host
> ↔ embed contract. **Pin the SDK to
> `@pollen-robotics/reachy-mini-sdk@1.8.0`** (the stable release
> validated against the host shell + daemon).
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
    clientId: "my-app",              // optional — OAuth client / app id
    appName: "My App",               // optional — label advertised to the robot
    videoJitterBufferTargetMs: 0,    // optional — 0 = render ASAP (teleop); omit for default browser buffering
    autoStartFromUrl: false,         // optional — auto startSession() when the URL carries a robot_peer_id hint
    // NOTE: `enableMicrophone` is deprecated and ignored. The SDK never grabs
    // the user's mic; it always wires a silent placeholder audio sender you
    // can replaceTrack() with your own audio (TTS, files, or the user's mic).
})
```

### State Machine

```
'disconnected' ──connect()──▸ 'connected' ──startSession()──▸ 'streaming'
     ▴ disconnect()                ▴ stopSession()
     └─────────────────────────────┘
```

> **One-shot bring-up.** `autoConnect(options?)` runs the whole
> auth → `connect()` → robot pick → `startSession()` → `ensureAwake()`
> chain and resolves to `{ robotId, robotName, isEmbedded }`. It
> auto-picks when exactly one robot is free, otherwise calls your
> `options.pickRobot(robots)` callback. This is the recommended entry
> point when you're **not** using the host shell.

### Properties (read-only)

| Property | Type | Description |
| :--- | :--- | :--- |
| `state` | `string` | `"disconnected"`, `"connected"`, or `"streaming"` |
| `robots` | `Array` | Available robots: `[{ id, meta: { name } }]` |
| `robotState` | `Object` | Latest `state` event detail — `{ head: number[16], antennas: [rRad, lRad], head_joint_positions: number[7], antennas_joint_positions: [rRad, lRad], body_yaw, motor_mode, is_move_running }` (wire shape). `head_joint_positions` is per-motor radians: body yaw at `[0]`, the 6 Stewart-platform neck motors at `[1..6]`; `antennas_joint_positions` is `[right, left]` in radians. Both fields appear only once the daemon emits them. |
| `username` | `string\|null` | HF username after `authenticate()` |
| `isAuthenticated` | `boolean` | True if a valid HF token is available |
| `micSupported` | `boolean` | True if robot offers bidirectional audio |
| `micMuted` | `boolean` | Your microphone mute state |
| `audioMuted` | `boolean` | Robot speaker mute state (local only) |
| `preselectedRobotId` | `string\|null` | Robot id read from `?robot_peer_id=` / `#robot_peer_id=` (set by the host iframe) |
| `isEmbedded` | `boolean` | `true` iff `preselectedRobotId !== null` — UX branching helper (embed vs standalone) |

### Methods

| Method | Returns | Description |
| :--- | :--- | :--- |
| `authenticate()` | `Promise<boolean>` | Check for existing HF OAuth token |
| `login()` | — | Redirect to HF login page |
| `connect()` | `Promise` | Open SSE connection, receive robot list |
| `autoConnect(opts?)` | `Promise<{robotId, robotName, isEmbedded}>` | One-shot bring-up: auth → connect → pick robot → session → wake. Auto-picks a single free robot, else calls `opts.pickRobot(robots)` |
| `startSession(robotId)` | `Promise` | Negotiate WebRTC, resolves when video + data ready |
| `stopSession()` | `Promise` | End session, back to `connected` |
| `disconnect()` | — | Close signaling (keeps auth) |
| `logout()` | — | Clear HF credentials |
| `attachVideo(videoEl)` | `() => void` | Bind video stream to element; returns cleanup function |
| `setTarget({ head?, antennas?, body_yaw? })` | `boolean` | Atomic raw-units update — `head` is `number[16]` (flat 4×4), `antennas` is `[rRad, lRad]`, `body_yaw` is radians |
| `gotoTarget({ head?, antennas?, body_yaw?, duration })` | `boolean` | Smooth daemon-side interpolation to a target pose over `duration` seconds (same wire units as `setTarget`). Throws `TypeError` on invalid input |
| `setHeadRpyDeg(roll, pitch, yaw)` | `boolean` | Set head orientation in degrees (wraps `setTarget`) |
| `setAntennasDeg(right, left)` | `boolean` | Set antenna positions in degrees (wraps `setTarget`) |
| `setBodyYawDeg(yaw)` | `boolean` | Set body yaw in degrees (wraps `setTarget`) |
| `startHeadTracking(weight?)` / `stopHeadTracking()` | `boolean` | Enable / disable daemon-side visual head tracking. `weight` in `[0,1]` blends tracking with app motion (default `1.0`) |
| `getTrackedFace()` | `Promise<FaceTarget\|null>` | Latest face seen by head tracking — `{ detected, x, y, roll }` |
| `setMotorMode(mode)` | `boolean` | `"enabled"` (position control), `"disabled"` (limp), or `"gravity_compensation"` (float by hand) |
| `setMotorTorque(on, ids?)` | `boolean` | Toggle torque; per-motor when `ids` is given, else global |
| `wakeUp(opts?)` / `gotoSleep(opts?)` | `Promise<void>` | Play the wake-up / sleep trajectory; resolves on daemon completion (rejects after `opts.timeoutMs`, default 8000). `wakeUp` enables motors first |
| `isAwake()` | `boolean` | Awake state derived from the cached `motor_mode` (`gravity_compensation` counts as awake) |
| `ensureAwake(timeoutMs?)` | `Promise<boolean>` | Idempotent bring-up to position control: awaits the wake trajectory when asleep, flips `gravity_compensation` back to `enabled` (no emote), no-op when already there. Never rejects |
| `playSound(filename)` | `boolean` | Play a sound file on the robot |
| `clearIncomingAudio()` | `boolean` | Drop audio queued for the robot speaker (barge-in) |
| `sendRaw(data)` | `boolean` | Send arbitrary JSON via data channel |
| `requestState()` | `boolean` | Request a state snapshot |
| `setAudioMuted(muted)` | — | Mute/unmute robot speaker (local) |
| `setMicMuted(muted)` | — | Mute/unmute your microphone |
| `getVolume()` / `setVolume(v)` | `Promise<number\|null>` | Read / set speaker volume (integer `0..100`, clamped); resolves the applied value or `null` if unavailable |
| `getMicrophoneVolume()` / `setMicrophoneVolume(v)` | `Promise<number\|null>` | Read / set microphone gain (integer `0..100`, clamped) |
| `getVersion()` | `Promise<string\|null>` | Daemon version string |
| `getHardwareId()` | `Promise<string\|null>` | Hardware ID (USB serial); `null` on developer machines |
| `applyAudioConfig(config, opts?)` | `Promise<boolean>` | Write a batch of XVF3800 audio-board parameters (`[{ name, values }]`); `opts.verify` reads them back (default `true`). Wireless only |
| `readAudioParameter(name)` | `Promise<number[]\|null>` | Read one XVF3800 parameter by name |
| `subscribeLogs({ onLine, onError? })` | `() => void` | Stream the daemon's `journalctl` logs over the data channel; returns an unsubscribe fn |
| `playRecordedMove(name, opts?)` | `boolean` | Play a named move (motion + bundled sound) from a HF dataset, daemon-side. Fire-and-forget (resolves on send, not on playback end). `opts.dataset` defaults to the pre-downloaded emotions library; `opts.initialGotoDuration` sets the lead-in goto |
| `playRecordedMoveAndWait(name, opts?)` | `Promise<boolean\|null>` | Same command, but resolves on the daemon's dispatch ack: `true` once the move is loaded and starting, `false` when it couldn't be loaded (unknown name, missing dataset), `null` on the fail-open timeout. Prefer it when the dataset may be cold — the daemon downloads before acking, so `opts.timeoutMs` defaults to 120 s |
| `stopMove()` | `boolean` | Stop whatever move is playing (recorded, uploaded or goto) and silence its bundled sound. Idempotent: a stop with nothing running is acked as a no-op |
| `preloadDataset(dataset)` | `boolean` | Ask the daemon to pre-download a recorded-move dataset into its HF cache, so a later `playRecordedMove` starts instantly. Fire-and-forget; the daemon only preloads the official `pollen-robotics` libraries by itself |
| `preloadDatasetAndWait(dataset, opts?)` | `Promise<boolean\|null>` | Same, but resolves when the dataset is actually cached: `true` on success, `false` on a daemon-reported download failure, `null` on the fail-open timeout |
| `playMove(motion, opts?)` | `Promise<{finished?, cancelled?, error?, has_audio?}>` | Upload + play a recorded move (optionally with audio) on the daemon's local clock; resolves when playback ends — see [Daemon-side recorded-move playback](#daemon-side-recorded-move-playback) |
| `cancelMove()` | `boolean` | Cancel an in-flight `playMove` |
| `uploadAudio(blob, opts?)` | `Promise<string>` | Upload a standalone audio slot, returns `uploadId` — pair with `playUploadedAudio` for record-time sync |
| `playUploadedAudio(uploadId, opts?)` | `Promise<{started: true, ...}>` | Trigger daemon-side standalone audio playback; resolves on the daemon's `started` broadcast (use as a sync anchor) |
| `cancelAudio()` | `boolean` | Cancel an in-flight `playUploadedAudio` |
| `subscribePose()` / `unsubscribePose()` | `boolean` | Opt into a ~30 Hz daemon-pushed pose stream (unreliable/unordered data channel) that fires `state` events, instead of the 500 ms `requestState` poll. Refcounted — pair each subscribe with one unsubscribe. No-op on daemons without the pose channel |
| `getFirstWakeUp()` / `setFirstWakeUp(done)` | `Promise<boolean\|null>` | Read / persist the robot-wide "first wake-up wizard completed" flag. `null` = channel closed or daemon predates the command (fail-open: skip the wizard) |
| `getRobotName()` / `setRobotName(name)` | `Promise<string\|null>` | Read / persist the robot display name. Applied live by the daemon (status + central relay + mDNS), no restart needed. Resolves the stored name, or `null` on error / unsupported |
| `signOut()` | `Promise<boolean\|null>` | Sign the robot out of Hugging Face (daemon deletes its stored HF token, de-registering it from central). Reaches the robot remotely over the data channel. The session may drop right after the ack — expected |

> **`setTarget` head-vs-body coupling.** The `head` matrix is in the
> world frame. Sending `setTarget({ body_yaw })` alone rotates the
> body *but not the head's commanded world yaw* — the head's gaze
> stays fixed in world frame, so visually it appears to counter-rotate
> as the body turns. For tank-style "head follows body", add the body
> yaw delta to the head RPY's yaw and ship `head` + `body_yaw` in the
> same `setTarget` call. The baseline for the head yaw must be the
> last *commanded* value you tracked yourself, not `state.head` from
> the telemetry event — telemetry lags one WebRTC RTT and cumulative
> deltas computed against it stall under rapid input.

### Events

Use `robot.addEventListener(name, handler)` — the SDK extends `EventTarget`.

| Event | Detail | Description |
| :--- | :--- | :--- |
| `connected` | `{ peerId }` | Signaling connection established |
| `disconnected` | `{ reason }` | Signaling connection lost |
| `robotsChanged` | `{ robots }` | Robot list updated |
| `streaming` | `{ sessionId, robotId }` | WebRTC session active |
| `sessionStopped` | `{ reason }` | Session ended |
| `state` | Same shape as the `robotState` property (see above) | Robot state update (~500 ms polled, or ~30 Hz when subscribed via `subscribePose()`; wire shape) |
| `videoTrack` | `{ track, stream }` | Video track available |
| `micSupported` | `{ supported }` | Bidirectional audio availability |
| `error` | `{ source, error }` | Error from `signaling`, `webrtc`, or `robot` |
| `sessionRejected` | `{ reason, activeApp }` | The robot refused the session (e.g. busy with another app) |
| `iceStateChange` | `{ state }` | Granular ICE transition (`RTCIceConnectionState`); a transient `disconnected` is debounced before it escalates to `error` |
| `networkOnline` / `networkOffline` | `{}` | Browser connectivity regained / lost |
| `networkChange` | `{ effectiveType?, downlink?, rtt?, saveData? }` | Transport swap (Wi-Fi ↔ cellular) via the NetworkInformation API; best-effort, Chromium-only |

### Math Utilities

```js
import { rpyToMatrix, matrixToRpy, degToRad, radToDeg } from "@pollen-robotics/reachy-mini-sdk";

rpyToMatrix(roll, pitch, yaw)  // degrees → 4×4 rotation matrix (ZYX)
matrixToRpy(matrix)            // 4×4 matrix → { roll, pitch, yaw } in degrees
```

### Advanced: JSON-RPC & daemon admin

Beyond the typed surface above, the runtime object exposes lower-level hooks (not part of the host-shell typed handle — reach for these only when you need them):

- `rpcCall(method, params?, { timeoutMs? })` — send a JSON-RPC request over the data channel and await the correlated result (e.g. app-defined methods).
- `onNotification(method, cb)` — subscribe to one-way JSON-RPC notifications pushed by the robot/app (e.g. `conversation.turn`); returns an unsubscribe fn.
- `startDaemonUpdate({ preRelease?, onProgress? })` — trigger a PyPI update of the daemon. It restarts on success (which tears the session down), so treat a successful reconnect as the "done" signal; `onProgress` fires with `status: "failed"` if the install errors first.

### Debug logging

Every SDK log line is prefixed `[reachy:<ns>]` (`reachy:sdk`, `reachy:session` for auto-reconnect, `reachy:embed` / `reachy:host` for the host shell), so a devtools console filter on `reachy:` isolates the whole stack. The default level is `info`: a handful of lifecycle lines per session (boot phases, wake/sleep steps, reconnects). To see per-message traffic (every command, reply, and SSE event):

```js
// From the devtools console (persists across reloads):
localStorage.setItem("reachy-log", "debug"); location.reload();

// Or from code:
import { setLogLevel } from "@pollen-robotics/reachy-mini-sdk";
setLogLevel("debug"); // "debug" | "info" | "warn" | "error" | "silent"
```

Debug lines use `console.debug`, so also enable the "Verbose" level in the devtools console filter to see them.

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

## Live pose stream (30 Hz)

By default the SDK refreshes `robotState` by polling `requestState()` every
500 ms. For anything that mirrors the robot in real time (a 3D view, a
"wait for the move to finish" watcher), opt into the daemon's **pose
stream** instead: it *pushes* the state at ~30 Hz over a dedicated
unreliable/unordered data channel, firing the same `state` events.

```js
robot.subscribePose();               // start the push stream
robot.addEventListener("state", (e) => {
    // e.detail.head_joint_positions → per-motor radians (body yaw + 6 neck)
    render(e.detail);
});

// Later, when this consumer no longer needs the stream:
robot.unsubscribePose();
```

The subscription is **refcounted**: multiple consumers share one daemon-side
stream, so pair every `subscribePose()` with exactly one `unsubscribePose()`
— the daemon only stops pushing once the last consumer releases. Frames that
arrive out of order are dropped (the channel is unordered). Against an older
daemon that has no pose channel this is a no-op; fall back to `requestState()`
polling there.

## Robot onboarding & management

These daemon-side commands power first-run setup and remote administration.
Each resolves `null` when the data channel isn't open or the daemon predates
the command, so **fail-open**: treat `null` as "unsupported" and skip the
gated UI.

```js
// First wake-up wizard (robot-wide, persisted on the robot).
if ((await robot.getFirstWakeUp()) === false) {
    await runWakeUpWizard();
    await robot.setFirstWakeUp(true);
}

// Robot display name — applied live (status + central relay + mDNS), no restart.
const name = await robot.getRobotName();
await robot.setRobotName("Marvin");

// Sign the robot out of Hugging Face (deletes its stored token; the robot
// disappears from its owner's list until set up again). The session may
// tear down right after the ack — that's expected.
await robot.signOut();
```

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
