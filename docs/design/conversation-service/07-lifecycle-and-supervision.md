# 06 - Lifecycle and supervision

Status: draft / design proposal.

The cross-cutting rules that govern both the conversation engine and the motion
service: how they are kept warm and supervised, how they arbitrate robot
ownership, how power state and failures are handled. These are settled decisions,
not notes.

## Public-facing behaviors

- **Ownership errors** surface as `robot_busy` on `conversation.start`.
- **Auth errors** surface as `missing_credentials`.
- **Crashes** surface as `conversation.error { fatal: true }` + `phase: error`,
  then settle back to `idle` (a retry is just another `start`).
- **Inactivity** stops a session unless a control transport is connected.

Everything below is how those behaviors are produced; it has no separate client
surface.

## Warm supervision

A `ConversationSupervisor`:

- Keeps code/deps **warm** at daemon boot but the **runtime dormant**: no session,
  no motion loop until a trigger (`conversation.start`, or NFC/wake-word later).
  The motion service is lazy (started on first `motion.*` or on conversation
  start). No kiosk auto-start in v1.
- Runs the engine as a supervised asyncio task; on unhandled exception it logs and
  restarts the task gracefully without bouncing the daemon.
- Runs a **memory watchdog**: polls RSS below the OOM point and recycles the
  session before the kernel intervenes (workload has no sudden large allocation -
  vision is cloud).
- Keeps all queues **bounded**.

Rule: a **heavy local model** (future on-device vision) gets its own process,
OOM-scored. The engine does not; it stays in-process.

## Robot ownership: the app lock arbitrates

`robot_app_lock` is the single arbiter. Binary by design (conversation OR store app
OR remote session, never two), so no hardware conflict and no extra feature flag
for safety.

- **Extension required**: a **conversation holder not bound to the triggering
  transport**. `REMOTE_SESSION` dies with the WebRTC session and `LOCAL_APP` is a
  subprocess; a conversation survives client disconnect, so it needs its own holder
  identity/state, reusing the existing acquire / release / evict / `on_became_free`
  machinery.
- `conversation.start` while a store app holds the lock -> `robot_busy`; a store app
  starting while a conversation holds it -> eviction or `robot_busy` per policy.
- The lazy motion service operates under the current lock holder.
- The lock already prevents the old store conversation app and the new in-daemon
  service from running together, so hiding the old app behind a flag is optional
  cosmetics, not a safety requirement.

## Power state: wake / sleep owned by the daemon

- `conversation.start`: if asleep, **wake first** (enable motors + wake-up
  trajectory), then open the session.
- `conversation.stop`, fatal error, inactivity stop: **`goto_sleep`** + release /
  disable motors.
- Motors enabled on demand, released when idle, never held awake across sessions.

## Failure surfacing

The **supervisor** (not the dead task) owns failure -> client surfacing:

1. broadcast `conversation.error { fatal: true }` + `phase: error`;
2. tear down the session (release lock, sleep/release motors, stop audio);
3. settle to `idle`;
4. re-arm the warm engine.

`error` stays transient: every transport sees it via broadcast.

## Inactivity auto-stop

A prolonged inactivity window stops the session, **except when a control transport
is connected** (e.g. the mobile app observing keeps it alive indefinitely). Any
user/assistant turn or `say` resets the window.

## Auth (HF credentials)

`conversation.start` needs the HF token; absent -> `missing_credentials`. The
daemon already owns OAuth device-code login + token refresh + sign-out. The engine
just consumes the daemon-managed token; no new auth flow.

## Starting animation (latency mask)

Opening the S2S session takes a couple of seconds. During `starting` the controller
plays a wake-up animation via `motion.*` so the robot feels responsive. Session
pre-warm/pool is optional, only if the animation is not enough.

## Camera sharing (non-issue)

The camera is served by the daemon over IPC (unixfd socket), a read-only fan-out.
Face tracking, the `look`/camera tool and remote WebRTC video all consume
concurrently. No exclusive lock like audio.

## PR1 scope

Supervisor with warm-code/dormant-runtime, lazy motion, graceful task restart +
memory watchdog + bounded queues; lifespan/teardown wiring; transport-independent
conversation lock holder; wake-on-start / sleep-on-stop; crash -> error -> idle
surfacing; inactivity stop gated by a connected transport; consume the daemon HF
token; `starting` animation hook.
