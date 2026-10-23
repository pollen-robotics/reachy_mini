# 01 - Conversation engine

Status: draft / design proposal.

The engine is the in-process component that runs one speech-to-speech (S2S)
session at a time. It owns the cloud realtime connection, its audio I/O, and its
public `conversation.*` control surface. It is glue: it dispatches tool calls and
drives the body through `motion.*`, but never touches motors directly.

This doc covers the engine, the protocol it exposes, and its audio path - the
three faces of the same block.

## Public interface

### `conversation.*` (JSON-RPC 2.0 over the WebRTC DataChannel)

Rides the existing relay (#1266); no bespoke transport. Requests are dispatched
by direct in-process calls to the controller.

| method | params | blocking | resolves with |
|---|---|---|---|
| `conversation.start` | `config` | yes, until `running` or `start_timeout` | phase snapshot |
| `conversation.stop` | `session_id?` | no | phase snapshot (`idle`) |
| `conversation.restart` | `config`, `session_id?` | yes | phase snapshot (fresh `session_id`) |
| `conversation.status` | - | no | phase snapshot |
| `conversation.say` | `text`, `session_id?` | no (on accept) | result |
| `conversation.interrupt` | `session_id?` | yes, until playback stops | result |

### Events (one-way notifications, broadcast to every connected transport)

- `conversation.phase` - `{ session_id, phase, reason, origin{by}, config }`. Source of truth, also returned by `status`.
- `conversation.turn` - `{ state: listening|thinking|speaking, reason? }`.
- `conversation.transcript` - `{ role, text, final }`.
- `conversation.level` - `{ rms }` (~10 Hz).

### Error `reason` codes (stable, UI branches on them)

`robot_busy`, `missing_credentials`, `already_running`, `not_running`,
`start_timeout`.

### Consumes

- `motion.*` ([`02-motion-service.md`](./02-motion-service.md)) for embodiment.
- Tools ([`04-tools.md`](./04-tools.md)) on model tool-calls.
- The daemon's mic/speaker hardware for audio I/O.

## Lifecycle (controller FSM)

```
idle -> starting -> running -> stopping -> idle
                 \-> error (transient) -> idle
```

- `start` resolves config, takes the robot lock, wakes the robot if asleep, opens
  the S2S session, goes `running`.
- `stop` tears down, sleeps the robot, releases the lock. Idempotent.
- `restart` = stop + start (v1 may be non-atomic).
- `say` injects an assistant turn; `interrupt` is barge-in (cancel response, flush
  playback, back to `listening`).

Wake/sleep and the lock are owned by the supervisor
([`07-lifecycle-and-supervision.md`](./07-lifecycle-and-supervision.md)).

## Audio

- Mic capture + speaker playback happen in-process (GStreamer), low latency, no
  WebRTC renegotiation.
- The mic/speaker hardware has a **single owner** at a time; the engine
  coordinates acquire/release with the daemon media server.
- Barge-in flushes the player queue on user speech.
- Audio never rides the control DataChannel. v1 keeps mic + speaker on the robot
  (`local`); remote audio routes use dedicated WebRTC media tracks (later).

## Key decisions

- **In-process, warm, dormant.** Runs as a supervised asyncio task, deps ready at
  boot, no session until a trigger. Not a subprocess (see README rationale).
- **Reuse the relay, not a new transport.** `conversation.*` registers on #1266.
- **Ported concurrency core, dropped debt.** Keep the serialized `response.create`
  sender, `active_response` retry, debounced partial transcripts, barge-in flush,
  tool plumbing, and the `ConversationHandler` ABC as the real contract. Drop the
  `LocalStream` god-object, the multi-backend duck-typing, and the REST + SSE
  surface.
- **Stateless protocol.** No persistent server state; the full `config` is supplied
  on every `start` and echoed on `phase`. A late-joining transport resyncs via
  `status` (no replay buffer).
- **Optional heavy deps.** HF realtime deps are an optional extra, lazy-imported,
  so the daemon core stays lean.

## PR1 scope

Full FSM + `start/stop/status/say/interrupt`; ported S2S loop producing real
audio + transcripts + tool calls; `conversation.*` on the relay with `phase` /
`turn` / `transcript` / `level` broadcast and the five error codes; in-process
audio with clean hardware ownership and barge-in. `restart` may be non-atomic.
