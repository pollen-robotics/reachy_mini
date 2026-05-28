/**
 * Pose distance + scaled duration math.
 *
 * Pure functions, no SDK calls, no clock reads. The caller passes a
 * `current` pose (typically from `reachy.robotState`) and a `target`
 * pose; the helpers return per-channel distances and a "feels right"
 * duration for `gotoTarget`.
 *
 * Mirror of the daemon's `utils/interpolation.distance_between_poses`
 * (head magic-mm) + the scaled-duration algorithm used internally for
 * `wake_up` / `goto_sleep`. Duplicated client-side so apps know the
 * duration **synchronously** for audio sync, streamer scheduling, and
 * live constant tuning.
 */

import { radToDeg } from "../lib/math.js";
import type { PartialPose } from "./pose.js";
import {
    DEFAULT_SCALED_DURATION_PRESET,
    type ScaledDurationPreset,
} from "./presets.js";

// ─── Distance ────────────────────────────────────────────────────────────────

/**
 * Per-channel "raw" distance between two poses.
 *
 * Channels missing from either input are absent from the result (the
 * caller's `current` and `target` may both be `PartialPose`).
 *
 * Units:
 *   - `head`: "magic-mm" — translation_mm + rotation_deg fused into one
 *     scalar. The Python SDK's `utils/interpolation.distance_between_poses`
 *     uses the same trick: two unrelated units summed into a single
 *     number that's monotonic with "how big does this motion feel".
 *   - `antennas.right` / `antennas.left`: `|Δ°|` per antenna.
 *   - `body_yaw`: `|Δ°|`.
 *
 * Useful in log lines:
 *   `head=3.2mm, antennaR=18°, body=14°, limiter=antennaR`
 */
export interface PoseDistance {
    head?: number;
    antennas?: { right: number; left: number };
    body_yaw?: number;
}

/**
 * Compute the per-channel distance between two poses.
 *
 * Pure function. Allocates only the output object.
 */
export function distanceBetweenPoses(
    current: PartialPose,
    target: PartialPose,
): PoseDistance {
    const out: PoseDistance = {};

    if (current.head !== undefined && target.head !== undefined) {
        out.head = headDistanceMagicMm(current.head, target.head);
    }

    if (current.antennas !== undefined && target.antennas !== undefined) {
        const curR = current.antennas[0] ?? 0;
        const curL = current.antennas[1] ?? 0;
        const tgtR = target.antennas[0] ?? 0;
        const tgtL = target.antennas[1] ?? 0;
        out.antennas = {
            right: Math.abs(radToDeg(tgtR - curR)),
            left: Math.abs(radToDeg(tgtL - curL)),
        };
    }

    if (
        typeof current.body_yaw === "number" &&
        typeof target.body_yaw === "number"
    ) {
        out.body_yaw = Math.abs(radToDeg(target.body_yaw - current.body_yaw));
    }

    return out;
}

/**
 * Fuse translation (mm) and rotation (deg) of a 4×4 head transform
 * delta into a single "magic-mm" scalar.
 *
 * Mirrors Python `distance_between_poses`:
 *   - Translation distance = `|t_a - t_b|` (Euclidean, mm).
 *   - Rotation distance = `arccos((trace(R_a^T · R_b) - 1) / 2)` in
 *     degrees, robust to numerical drift via clamping.
 *   - Magic-mm = `translation_mm + rotation_deg`.
 *
 * Assumes incoming matrices use **metres** for the translation column
 * (indices 3, 7, 11 in row-major), matching the daemon's wire format,
 * and converts to millimetres internally.
 */
function headDistanceMagicMm(headA: number[], headB: number[]): number {
    // Translation in metres at indices 3, 7, 11 (row-major).
    const dxM = (headA[3] ?? 0) - (headB[3] ?? 0);
    const dyM = (headA[7] ?? 0) - (headB[7] ?? 0);
    const dzM = (headA[11] ?? 0) - (headB[11] ?? 0);
    const translationMm = Math.sqrt(dxM * dxM + dyM * dyM + dzM * dzM) * 1000;

    // Top-left 3×3 of each, row-major: trace(A^T · B) = Σ_ij A[i,j] * B[i,j]
    const trace =
        (headA[0] ?? 0) * (headB[0] ?? 0) +
        (headA[1] ?? 0) * (headB[1] ?? 0) +
        (headA[2] ?? 0) * (headB[2] ?? 0) +
        (headA[4] ?? 0) * (headB[4] ?? 0) +
        (headA[5] ?? 0) * (headB[5] ?? 0) +
        (headA[6] ?? 0) * (headB[6] ?? 0) +
        (headA[8] ?? 0) * (headB[8] ?? 0) +
        (headA[9] ?? 0) * (headB[9] ?? 0) +
        (headA[10] ?? 0) * (headB[10] ?? 0);

    // Clamp to [-1, 1] before arccos to absorb floating-point drift
    // (matrices that should be orthonormal sometimes have trace numerically
    // slightly outside the valid range).
    let cosTheta = (trace - 1) / 2;
    if (cosTheta > 1) cosTheta = 1;
    else if (cosTheta < -1) cosTheta = -1;
    const rotationDeg = radToDeg(Math.acos(cosTheta));

    return translationMm + rotationDeg;
}

// ─── Scaled duration ─────────────────────────────────────────────────────────

/** Outcome of a scaled-duration calculation. */
export interface ScaledDurationResult {
    /** Duration to pass to `gotoTarget`, clamped to the preset (sec). */
    readonly duration: number;
    /** Which channel was the slowest (i.e. dictated the duration).
     *  `null` when `current` and `target` had no overlapping channels. */
    readonly limiter: "head" | "antennaR" | "antennaL" | "body_yaw" | null;
    /** Per-channel un-clamped candidate durations (sec), for log lines
     *  and "why is this slow?" diagnostics. */
    readonly perChannel: Readonly<{
        head?: number;
        antennaR?: number;
        antennaL?: number;
        body_yaw?: number;
    }>;
}

/**
 * Pick a `goto` duration that "feels right" for the move from
 * `current` to `target`.
 *
 * Algorithm: for each channel defined on both sides, compute its
 * candidate duration as `channelDistance × channelCost`. Take the max
 * (slowest channel limits). Clamp to
 * `[preset.minDurationSec, preset.maxDurationSec]`.
 *
 * If no channel overlaps, returns `{ duration: minDurationSec,
 * limiter: null, perChannel: {} }` — a safe fallback the caller can
 * still pass to `gotoTarget` (the resulting move is a no-op held over
 * the min duration).
 *
 * **Divergence from Rémi's `reachy-mini-js-practices` bench (intentional):**
 * his version returns `max` in the no-overlap case. We return `min`
 * instead — a no-op move doesn't need the maximum dwell time, and the
 * minimum is the smallest legal duration the daemon accepts. Callers
 * that need a longer dwell can pass an explicit duration.
 *
 * Pure function. No SDK reads, no clock reads.
 */
export function scaledDuration(
    current: PartialPose,
    target: PartialPose,
    preset: ScaledDurationPreset = DEFAULT_SCALED_DURATION_PRESET,
): ScaledDurationResult {
    const dist = distanceBetweenPoses(current, target);

    const perChannel: {
        head?: number;
        antennaR?: number;
        antennaL?: number;
        body_yaw?: number;
    } = {};

    if (dist.head !== undefined) {
        perChannel.head = dist.head * preset.headSecPerMagicMm;
    }
    if (dist.antennas !== undefined) {
        perChannel.antennaR = dist.antennas.right * preset.antennaSecPerDeg;
        perChannel.antennaL = dist.antennas.left * preset.antennaSecPerDeg;
    }
    if (dist.body_yaw !== undefined) {
        perChannel.body_yaw = dist.body_yaw * preset.bodyYawSecPerDeg;
    }

    let limiter: ScaledDurationResult["limiter"] = null;
    // Sentinel: -1 means "no channel seen yet". Distinguishes "no
    // overlap" (limiter stays null) from "overlap but zero motion"
    // (limiter = first overlapping channel, raw = 0).
    let rawDuration = -1;
    for (const key of ["head", "antennaR", "antennaL", "body_yaw"] as const) {
        const value = perChannel[key];
        if (value === undefined) continue;
        if (value > rawDuration) {
            rawDuration = value;
            limiter = key;
        }
    }

    let duration = rawDuration < 0 ? 0 : rawDuration;
    if (duration < preset.minDurationSec) duration = preset.minDurationSec;
    if (duration > preset.maxDurationSec) duration = preset.maxDurationSec;

    return { duration, limiter, perChannel };
}
