# 03 - Tools

Status: draft / design proposal.

Tools are the verbs the model may call. The runtime routes every call, bounds it,
returns recoverable failures to the model, and cancels in-flight calls on
`interrupt` / `stop`. The client only observes (informational `conversation.tool`
events, later).

## Public interface

### Built-in set (v1)

| Tool | Effect | Resolves to |
|---|---|---|
| `move_head` | Aim the head | `motion.*` |
| `play_emotion` | Play an emotion clip | `motion.*` |
| `dance` | Play a dance | `motion.*` |
| `look` | Gaze / camera (when gaze is wired) | `motion.*` |

Memory tools (`save_memory` / `recall_memory`) are always-on system tools,
documented in [`05-memory.md`](./05-memory.md).

### Selection

- `config.tools` picks which built-ins are enabled for a session; unknown names
  degrade (skipped + reported), never fail `start`.
- `config.animations` scopes which clips the motion verbs may reach (AND with
  `tools`). See [`06-personas-and-config.md`](./06-personas-and-config.md).

### Execution contract

- Each call is bounded by a timeout; on error/timeout the runtime returns a failure
  the model can recover from.
- `interrupt` / `stop` cancel in-flight calls; a result landing after cancellation
  is dropped silently.
- Tools that opt out of a spoken follow-up keep that behavior.

## Consumed by / consumes

- Called by the engine ([`01-conversation-engine.md`](./01-conversation-engine.md))
  when the model emits a tool call.
- Motion tools call into [`02-motion-service.md`](./02-motion-service.md).
- The `camera` / `look` tools read a frame from the existing vision stack
  ([`03-vision.md`](./03-vision.md)); scene understanding is done by the cloud model.

## Key decisions

- **Fixed built-in registry for v1.** No per-profile Python file loading, no
  external dir scan, no remote MCP resolution at startup - that was a
  startup-latency source. Keep the `BackgroundToolManager` (async execution,
  cancel, timeout, cleanup); it is well built and tested - port close to as-is.
- **Degrade, never fail.** Unknown tool/animation names are dropped and reported,
  not fatal.
- **MCP tool-spaces are the later extension path**, re-wired behind the same
  `config.tools` shape.

## PR1 scope

`BackgroundToolManager` + `move_head` / `play_emotion` / `dance` wired to
`motion.*`, fixed built-in registry, timeout + cancel-on-interrupt.
