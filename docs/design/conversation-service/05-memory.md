# 04 - Memory

Status: draft / design proposal (V1 already implemented in the current app,
see PR #360 line).

Memory lets the robot recall facts across sessions, so the second interaction is
not a blank slate. It is a **capability the engine drives**, exposed to the model
as two always-on tools plus two automatic behaviors (logging + prompt injection).

## Public interface

### Model-facing tools (always-on system tools)

Registered like `task_status` / `task_cancel`: auto-loaded for every persona, not
listed in any `config.tools`.

| Tool | Params | Returns | Effect |
|---|---|---|---|
| `save_memory` | `fact` (string) | `{ status: "saved", fact }` | Append a concise fact to active memory (prompt-injected next session). |
| `recall_memory` | `log_ref` (string, or empty) | `{ log_ref, content }` \| `{ available_logs }` \| `{ error }` | Read a past session log; empty `log_ref` lists available logs. |

### Automatic behaviors (no client/model action)

- **Conversation logging.** Every user/assistant transcript and completed tool call
  is appended to a per-session log. Fire-and-forget; failures never crash the audio
  pipeline.
- **Prompt injection.** Non-empty active memory is appended to the system prompt on
  every session start and persona switch, so newly saved facts show up next time.

### Configuration

| Key | Default | Effect |
|---|---|---|
| `memory.enabled` | `true` | Turn the whole memory system on/off. |
| data directory | `~/.reachy_mini/data/` | Root for logs + active memory (outside the repo). |

Fits the session `config` object as an optional `memory` field
([`06-personas-and-config.md`](./06-personas-and-config.md)); absent = defaults.

## Two-tier model

1. **Active memory** - a small curated file of facts, injected into the prompt.
   Grows unbounded; a soft warning fires around ~1,500 tokens (comfortable headroom
   under the realtime model's instruction budget).
2. **Conversation logs** - full per-session transcripts, kept indefinitely, read on
   demand via `recall_memory`. Each active-memory fact carries the log filename it
   came from, so the model can pull the detailed conversation.

## Key decisions

- **Flat files, no vector DB (V1).** File-based grep/read matches the Letta
  Filesystem benchmark (~74% LoCoMo, beating vector approaches) and is sufficient
  below ~150 conversations. Vector search can slot in behind `recall_memory` later
  with no interface change.
- **LLM-driven save.** The model decides what to remember via `save_memory`, rather
  than automatic extraction - more selective, less noise.
- **Graceful degradation.** Enabled by default, disableable; any memory failure is
  logged, never fatal to the conversation.
- **Global, not per-persona.** One robot = one memory. Multi-user scoping is a
  future extension (namespace the data dir by user id).

## PR1 scope

`save_memory` / `recall_memory` as always-on tools, per-session transcript logging,
active-memory prompt injection, and the `memory.enabled` switch. Sleep-time
compute, summarization, vector recall and a `forget_memory` tool are later.
