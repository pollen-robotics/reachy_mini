# 05 - Personas and config

Status: draft / design proposal.

`config` is the single source of truth for a session, handed to
`conversation.start`. A **persona** is a named preset that resolves into config
fields (prompt / voice / tools). They belong together: config is the contract,
personas are the convenience layer over it.

## Public interface

### `config` object (v1 fields)

```jsonc
{
  "prompt": "You are a friendly desk robot...",
  "voice": "...",
  "language": "en",                 // soft hint, biases transcription
  "tools": ["move_head", "play_emotion", "dance"],
  "animations": { "emotions": ["happy", "curious"], "dances": ["wiggle", "spin"] }
}
```

Every field optional; absent fields fill from robot defaults. Deferred (documented
so they slot in without a protocol change): `memory`, `assets`, `sounds`,
`wobble`, `vision.gaze`, `vad`, `inactivity_timeout_ms`, `audio` route.

### `personalities.*` / `voices.*` (over the relay)

Discovery:

- `personalities.list` - choices + current + startup + lock state.
- `personalities.all` - every persona with full config + `avatar_id` (no inline SVG).
- `personalities.load` - one persona's instructions / greeting / tools / voice.
- `personalities.avatar` - SVG markup for a persona (lazy, cached by `avatar_id`).
- `voices.list` / `voices.current`.

CRUD and apply:

- `personalities.save` - create/update a user persona (persisted on the robot).
- `personalities.delete` - delete a user persona (never a built-in or the active/startup one).
- `personalities.apply` - apply now, optionally persist as the startup choice.
- `voices.apply` - change voice live.

## Config resolution (fill + degrade)

On `start` the runtime produces an **effective config** deterministically:

1. drop unknown fields, fill unset ones from robot defaults;
2. resolve references, dropping ones that degrade (unknown `tools` / `animations` /
   `voice` entry is removed, never fatal);
3. normalise to canonical form (scalars by value, reference lists as sets).

The effective config is echoed back on `conversation.phase`. Unknown field or
reference degrades + is reported, never fails `start` - this is what keeps a newer
client compatible with an older robot.

## Key decisions

- **Stateless protocol, config as resolver.** The full config is supplied on every
  `start`; the protocol holds no persistent state. A thin client may send just a
  persona name and let the robot resolve it, or send an inline config. Discovery of
  names is out-of-band (`personalities.*` / `voices.*`), never enumerated by
  `conversation.*`.
- **Works standalone.** Default personas are embedded on the robot and the startup
  persona is persisted, so the robot is usable with no client and no catalog.
- **Port personas as-is.** `PersonalityOps` is already transport-agnostic; port it
  and register on the relay. Avatar SVGs live on the robot (profile-local, built-in
  map, default fallback, stable `avatar_id` for client caching).

## PR1 scope

The five v1 config fields with fill-defaults + degrade-unknown resolution and
effective config echoed on `phase`; `personalities.*` / `voices.*` registered on
the relay with default personas + persisted startup; a persona resolves into
config.
