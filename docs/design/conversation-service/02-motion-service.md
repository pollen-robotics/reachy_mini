# 02 - Motion service (embodiment)

Status: draft / design proposal.

The embodiment layer is a **standalone daemon capability**, not something buried
inside the conversation. It is exposed over WebRTC as `motion.*` so **any client
or JS-SDK app can drive the robot's body without a conversation session**. The
conversation engine is just one consumer.

## Public interface

### `motion.*` (over the relay)

Proposed v1 surface (minimal, extend later without breaking):

| RPC | Params | Effect |
|---|---|---|
| `motion.play` | `{ name }` | Play a recorded move, interrupt breathing, return to base |
| `motion.stop` | - | Stop the current move, resume idle |
| `motion.look_at` | `{ yaw, pitch } \| null` | Aim gaze (smoothed/held); `null` releases |
| `motion.set_listening` | `{ enabled }` | Freeze / blend-back the antennas |
| `motion.set_idle` | `{ enabled }` | Turn idle life on/off |
| `motion.status` | - | Current move, listening, loop stats |

Later (roadmap): raw `motion.set_target` / `goto`, `motion.set_offsets`,
`wake_up` / `goto_sleep`.

### Consumed by

- The conversation engine ([`01-conversation-engine.md`](./01-conversation-engine.md)):
  enqueues emotion/dance moves, toggles `set_listening` on speech events.
- The motion tools ([`04-tools.md`](./04-tools.md)): `play_emotion` / `dance` /
  `move_head` resolve to `motion.*` calls.

## Behavior model

Four layers fused in a single control loop, summed on top of each other and
clamped to the reachable envelope:

- **Primary**: sequential recorded moves (emotions, dances, gotos) + the idle
  breathing floor. The only layer that moves the antennas.
- **Gaze** (additive): aim from face tracking.
- **Offsets** (additive): speech-reactive wobble.

Invariants: a move wins over breathing; gaze coexists with breathing but yields to
expressive moves; direct poses pause idle and auto-resume; `set_listening` freezes
then blends the antennas back.

## Key decisions

- **Standalone and over WebRTC.** "Alive" is a property of the robot, not of an
  app. Every app gets the same breathing, never-frozen Reachy by sending
  intentions ("play this emotion", "look here"), never a 30 Hz keep-alive from the
  network.
- **Lazy, under the lock.** Not always-on: started on the first `motion.*` call or
  when a conversation starts, dormant otherwise. Runs under the current
  `robot_app_lock` holder; never fights a store app for the motors. See
  [`07-lifecycle-and-supervision.md`](./07-lifecycle-and-supervision.md).
- **Not a general arbitration engine (explicit non-goal).** We ship the constrained
  slice games have run for 20 years - a fixed layer set, trivial arbitration
  (fixed priority + last-writer-wins), clamp as safety only, additive fusion in one
  RT loop. No runtime-arbitrary blend graph, no "smart" command fusion. Tractable
  because Reachy Mini is low-DoF, position-controlled, no balance, no contact.

## Notes (implementation)

- **Port** the primary layer + idle breathing + listening freeze in-process,
  calling the backend motion API directly.
- **Reuse** the daemon-side gaze from the existing vision stack
  ([`03-vision.md`](./03-vision.md)) and the speech wobble already tapped in the
  media server. Do not reimplement head tracking.
- **New**: the fusion point (sum primary + gaze + offsets, clamp) and the `motion.*`
  RPC binding on the relay.

## PR1 scope

Primary layer + idle breathing + listening freeze ported in-process; fusion with
existing gaze and wobble; `motion.play` / `stop` / `set_listening` / `status` on
the relay; drivable standalone with no conversation session.
