/**
 * Canonical safe-rest pose + scaled-duration preset.
 *
 * Both are mirrored from the daemon (Python). The constants live on
 * both sides because apps need the values **synchronously** in JS:
 *
 *   - `INIT_POSE`: the "home" pose for `safelyReturnToPose`'s default
 *     target. Mirror of `Backend.INIT_HEAD_POSE` + `Backend.INIT_ANTENNAS_JOINT_POSITIONS`
 *     in `src/reachy_mini/daemon/backend/abstract.py`.
 *   - `DEFAULT_SCALED_DURATION_PRESET`: the per-channel "magic-mm × cost"
 *     tuning. Mirror of the constants the daemon already uses for
 *     `wake_up()` / `goto_sleep()` durations.
 *
 * **If you change a value here**, update the matching constant in
 * `src/reachy_mini/daemon/backend/abstract.py` in the same commit.
 */

import type { Pose } from "./pose.js";

// ─── Canonical safe-rest pose ────────────────────────────────────────────────

/**
 * Identity 4×4 head matrix, row-major, flattened.
 *
 * Mirror of `Backend.INIT_HEAD_POSE` (Python `np.eye(4)`) in
 * `abstract.py`. Frozen so apps can't accidentally mutate the shared
 * instance via `INIT_POSE.head[0] = 0`.
 */
export const INIT_HEAD_POSE_FLAT: readonly number[] = Object.freeze([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
]);

/**
 * Antennas at ~±10° outward (anti-resonance offset).
 *
 * Mirror of `Backend.INIT_ANTENNAS_JOINT_POSITIONS = [-0.1745, 0.1745]`
 * (rounded to match Python bit-for-bit) in `abstract.py`. Background:
 * straight-up (0°, 0°) antennas mechanically resonate with the
 * head-motor cogging frequency; the ~10° symmetric outward tilt damps
 * it. See [PR #952](https://github.com/pollen-robotics/reachy_mini/pull/952).
 */
export const INIT_ANTENNAS_RAD: readonly [number, number] = Object.freeze([
    -0.1745, 0.1745,
]) as readonly [number, number];

/** Body squared up: head and body share the world-frame yaw axis at rest. */
export const INIT_BODY_YAW_RAD: number = 0;

/**
 * Aggregate canonical "safe rest" pose.
 *
 * Default target for `safelyReturnToPose`. Deep-frozen, so spread it
 * if you need a mutable copy:
 * `{ ...INIT_POSE, head: INIT_POSE.head.slice() }`.
 */
export const INIT_POSE: Readonly<Pose> = Object.freeze({
    head: INIT_HEAD_POSE_FLAT as readonly number[] as number[],
    antennas: INIT_ANTENNAS_RAD as readonly [number, number] as [number, number],
    body_yaw: INIT_BODY_YAW_RAD,
});

// ─── Scaled-duration preset ──────────────────────────────────────────────────

/**
 * Tuning knobs for `scaledDuration` and friends.
 *
 * Each `*SecPer*` field is the per-unit cost of motion on that channel.
 * A move's duration is `max(channelDistance × channelCost)` across all
 * channels, clamped to `[minDurationSec, maxDurationSec]`. Slowest
 * channel wins.
 */
export interface ScaledDurationPreset {
    /** Head: `magicMm × this` = head channel candidate (sec). */
    readonly headSecPerMagicMm: number;
    /** Per antenna: `|Δ°| × this` = each antenna's candidate (sec). */
    readonly antennaSecPerDeg: number;
    /** Body yaw: `|Δ°| × this` = body channel candidate (sec). */
    readonly bodyYawSecPerDeg: number;
    /** Lower bound on the final duration (sec). */
    readonly minDurationSec: number;
    /** Upper bound on the final duration (sec). */
    readonly maxDurationSec: number;
}

/**
 * Default tuning, mirrored from RemiFabre's test bench
 * (`reachy-mini-js-practices`, May 2026) — the slider defaults he
 * validated on a real Reachy Mini.
 *
 * Per-channel rationale:
 *   - **head**: 0.02 s per "magic-mm" (translation_mm + rotation_deg).
 *     Same constant as `utils/interpolation.py:distance_between_poses`
 *     daemon-side.
 *   - **antenna**: 0.005 s/°. Antennas are light, but the mechanical
 *     resonance window (~PR #952) penalises overly fast moves.
 *   - **body_yaw**: 0.015 s/°. Body sits between head (heavy) and
 *     antennas (light).
 *
 * Hard bounds `[0.2, 1.5]` match the host shell's leave-protocol
 * budget, so `safelyReturnToPose` finishes within the iframe's
 * `pagehide` window.
 *
 * Frozen so apps can't mutate the shared instance. Spread to derive
 * a custom preset: `{ ...DEFAULT_SCALED_DURATION_PRESET, headSecPerMagicMm: 0.03 }`.
 */
export const DEFAULT_SCALED_DURATION_PRESET: Readonly<ScaledDurationPreset> = Object.freeze({
    headSecPerMagicMm: 0.02,
    antennaSecPerDeg: 0.005,
    bodyYawSecPerDeg: 0.015,
    minDurationSec: 0.2,
    maxDurationSec: 1.5,
});
