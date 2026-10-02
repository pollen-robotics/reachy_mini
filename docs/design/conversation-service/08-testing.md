# 07 - Testing

Status: draft / design proposal.

Method: the public contract is precise enough to become a conformance suite first,
then the implementation is written to green (docs -> tests -> agent).

## Control plane: conformance tests (PR1)

The control plane (RPC / events / config) is deterministic and mockable - unit
testable with a mocked S2S backend and a mocked backend/motion layer:

- **RPC subset**: `start` / `stop` / `status` / `say` / `interrupt` request-response
  shapes and blocking semantics.
- **Lifecycle**: `idle -> starting -> running -> stopping -> idle`, and `error` as a
  transient that settles back to `idle`.
- **Events**: `conversation.phase` (source of truth) and `conversation.turn`
  sub-states emitted on the right transitions and broadcast to all transports.
- **Config resolution**: fill-defaults, degrade-unknown-field,
  degrade-unknown-reference; effective config echoed on `phase`.
- **Error codes**: `robot_busy`, `not_running`, `missing_credentials`,
  `already_running`, `start_timeout`.
- **Observe-without-starting**: a second transport `status` mid-session sees
  `running` + `origin.by`.

Ported components keep their existing tests where they move over
(`BackgroundToolManager`, personality ops, memory manager, motion primary layer).

## Motion service

- **Unit**: fusion clamps to the reachable envelope; move-wins-over-breathing;
  `set_listening` freeze/blend; gaze yields to expressive moves.
- **Standalone**: `motion.*` drivable with no conversation session.

## End-to-end: full WebRTC (PR1)

PR1 is driven **entirely over WebRTC** (relay #1266 + JS SDK), no REST/SSE
fallback. The deliverable includes a **minimal WebRTC SDK harness** (a script or a
tiny page) that runs `start` -> observes `phase` / `turn` / `transcript` -> `say`
-> `interrupt` -> `stop`, and drives `motion.*` standalone. This harness is the
actual driver for on-robot manual validation until the reference thin client lands.

## Not unit-testable (human-in-the-loop)

The embodied, real-time feel - turn-taking, voice, S2S latency, motion feel - needs
HITL iteration on hardware. Validate on the robot over SSH
(`journalctl -u reachy_mini`), and land latency `metrics` early to measure
`first_audio_ms` / turn latency rather than guess.

## PR1 scope

The control-plane conformance suite for the v1 RPC/event/config/error subset, the
motion-service unit tests (green against mocked backends), and the minimal WebRTC
SDK harness for end-to-end validation.
