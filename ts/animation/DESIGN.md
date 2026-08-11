# Reachy Mini JS Motion — Scope

**Status:** draft, for iteration.
**Last updated:** 2026-05-28.

Two-phase plan for the JS-side motion tooling.

---

## Phase 1 — what ships in this PR

Pure TypeScript, subpath export `@pollen-robotics/reachy-mini-sdk/animation`.
No daemon changes.

### `ts/animation/`

**`pose.ts`** — wire-format types:

```ts
interface Pose         { head: number[]; antennas: [number, number]; body_yaw: number }
interface PartialPose  { head?: number[]; antennas?: [number, number] | number[]; body_yaw?: number }
```

**`presets.ts`** — canonical safe-rest pose + default scaled-duration tuning:

```ts
INIT_HEAD_POSE_FLAT          // identity 4×4, row-major flat
INIT_ANTENNAS_RAD            // [-0.1745, 0.1745]
INIT_BODY_YAW_RAD            // 0
INIT_POSE                    // aggregate of the three above
DEFAULT_SCALED_DURATION_PRESET // { 0.02 / 0.005 / 0.015 / 0.2 / 1.5 }
```

**`distance.ts`** — pure math:

```ts
distanceBetweenPoses(current, target)         → PoseDistance        // per-channel raw
scaledDuration(current, target, preset?)      → ScaledDurationResult // { duration, limiter, perChannel }
```

**`safe-return.ts`** — exit-handler convenience:

```ts
safelyReturnToPose(reachy, { target?, preset? })   // setMotorMode + scaledDuration + gotoTarget
installShutdownHandler(reachy, { onlyWhenStreaming? })   // wires pagehide + beforeunload
```

### Milestones

| ID    | Deliverable                                                       | Status |
|-------|-------------------------------------------------------------------|--------|
| P1.1  | `Pose` / `PartialPose` types                                       | ✅     |
| P1.2a | `distanceBetweenPoses` + `scaledDuration` + default preset         | ✅     |
| P1.2b | `INIT_POSE` constants + `safelyReturnToPose` + `installShutdownHandler` | ✅ |
| P1.3  | `composeWorldOffset` + `combinePrimaryAndOffsets` (pure math)      | ⏳ deferred to first real caller |
| P1.4  | Recorded-move player (Marionette-v1 streamer, 50 Hz `setTarget`, lead compensation) | ⏳ deferred to first real caller |
| P1.5  | Migrate `reachy_mini_emotions`, `reachy_mini_marionette` v1+v2 off their local copies | ⏳ follow-up PR per app |

`LEAD_PRESET_V1` for the future recorded-move player (`{ head: 0.205, antennas: 0.090, body_yaw: 0.205 }`, May 2026 marionette v1 calibration) is parked here so the constants aren't lost.

---

## Phase 2 — deferred

A video-game-style animation graph: named layers, masking, crossfades, procedural clips.
Would let the conversation app delete its boolean gates (`isPoseLocked`,
`isMovePlaying`, `isTrajectoryPlaying`) and express "speech sway while a
dance plays" declaratively.

Phase 1 is its foundation: Pose types, distance math, `composeWorldOffset`.
Ship those first, validate in real apps, then build the graph on top.

New PR when we start.
