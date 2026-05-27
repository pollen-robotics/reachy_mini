/**
 * Reachy Mini — utility helpers for JS apps.
 *
 * The missing-from-the-SDK pieces that every JS app re-implements by
 * hand: wire-format pose helpers, distance metric + scaled-duration
 * goto, pose composition for primary+offset fusion, shutdown handler,
 * and one safe-return-to-pose sugar. Mirrors the equivalent helpers in
 * the Python SDK (`reachy_mini/utils/interpolation.py` and friends).
 *
 * Import:
 *   import { scaledDuration, installShutdownHandler, ... }
 *     from "@pollen-robotics/reachy-mini-sdk/utils";
 *
 * Items 1-3 (distance, scaled-duration, shutdown handler) are
 * battle-tested via the reachy-mini-js-practices HF Space app. The
 * fusion helpers (composeWorldOffset, combineFullBody) are direct ports
 * of the Python implementation and have not yet been exercised from JS;
 * use them, but expect a follow-up validation pass.
 *
 * Notes on safe torque-on: with PR #1138 merged on the daemon side,
 * `setMotorMode("enabled")` is safe by construction — the daemon pins
 * all targets to the present pose before flipping torque on. Apps no
 * longer need a JS-side pre-pin / sleep dance, so there is no
 * `safeEnableTorque` function here. The shutdown handler below assumes
 * a post-#1138 daemon.
 */

import type { ReachyMiniInstance } from './lib/types';

// ─── Constants ───────────────────────────────────────────────────────

/** Identity 4×4 — the canonical "head at rest" pose. */
export const INIT_HEAD_POSE: number[][] = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
];

/** ~10° outward antennas tilt. Straight-up (0,0) is mechanically
 *  resonant and visibly buzzes; the small tilt damps that out.
 *  Mirrored from `reachy_mini.INIT_ANTENNAS_JOINT_POSITIONS`. */
export const INIT_ANTENNAS_RAD: [number, number] = [-0.1745, 0.1745];

export const INIT_BODY_YAW_RAD = 0;

/** Per-channel seconds-per-unit for `scaledDuration`. Tuned on a real
 *  robot via the reachy-mini-js-practices test app. */
export const SCALED_DURATION_PRESET = {
    /** Head: sec per "magic-mm" (translation_mm + rotation_deg). */
    head: 0.015,
    /** Antennas: sec per degree (shared by both; each is its own
     *  channel for the max). */
    antenna: 0.005,
    /** Body yaw: sec per degree. */
    bodyYaw: 0.015,
} as const;

/** Lower bound prevents tiny moves from snapping; upper bound prevents
 *  huge moves from crawling. */
export const SCALED_DURATION_CLAMP = { min: 0.01, max: 1.5 } as const;

/** Per-channel lead offsets in seconds, used when replaying a recorded
 *  motion. Actuators have measurable, channel-dependent response
 *  latency; sample each channel a bit *ahead* of the audio so the
 *  physical motion lines up with the audio reference.
 *
 *  Calibrated on marionette v1 (May 2026): head 205 ms, antennas 90 ms.
 *  Body yaw assumed equal to head pending a dedicated bench. Steady-
 *  state residual post-comp ~1.3° head, ~0.25 cm translation. */
export const LEAD_COMPENSATION_S = {
    head: 0.205,
    antenna: 0.090,
    bodyYaw: 0.205,
} as const;


// ─── Wire-format pose helpers ────────────────────────────────────────
//
// The SDK exposes head poses on the wire as 16-element flat row-major
// arrays. Apps usually want nested 4×4 internally (matrix math reads
// better). These conversions are one-liners, but every app re-inlines
// them; collecting them here saves duplication and gives a single
// canonical convention.

/** Nested 4×4 → flat 16-element row-major array. */
export function flat16(nested4x4: number[][]): number[] {
    return nested4x4.flat();
}

/** Flat 16-element row-major array → nested 4×4. */
export function nest4x4(flat: number[]): number[][] {
    return [
        [flat[0]!, flat[1]!, flat[2]!, flat[3]!],
        [flat[4]!, flat[5]!, flat[6]!, flat[7]!],
        [flat[8]!, flat[9]!, flat[10]!, flat[11]!],
        [flat[12]!, flat[13]!, flat[14]!, flat[15]!],
    ];
}


// ─── Distance metric ─────────────────────────────────────────────────

/** Angle between two 3×3 rotation matrices, in radians. Equivalent to
 *  the angular distance in axis-angle space: angle = acos((tr(P·Qᵀ) - 1) / 2). */
export function deltaAngleBetweenRot(P: number[][], Q: number[][]): number {
    const trace =
        P[0]![0]! * Q[0]![0]! + P[0]![1]! * Q[0]![1]! + P[0]![2]! * Q[0]![2]! +
        P[1]![0]! * Q[1]![0]! + P[1]![1]! * Q[1]![1]! + P[1]![2]! * Q[1]![2]! +
        P[2]![0]! * Q[2]![0]! + P[2]![1]! * Q[2]![1]! + P[2]![2]! * Q[2]![2]!;
    let cos = (trace - 1) / 2;
    if (cos > 1) cos = 1;
    if (cos < -1) cos = -1;
    return Math.acos(cos);
}

/** Three distances between two 4×4 head poses. Mirrors the Python SDK's
 *  `distance_between_poses` (`reachy_mini/utils/interpolation.py:161`).
 *  `magicMm = translation_mm + rotation_deg` is an "arbitrary but
 *  emotionally satisfying" equivalence (1 mm ≈ 1°) that combines both
 *  channels into a single scalar — useful for picking a goto duration. */
export function distanceBetweenPoses(
    p1: number[][],
    p2: number[][],
): { translationM: number; angleRad: number; magicMm: number } {
    const dx = p1[0]![3]! - p2[0]![3]!;
    const dy = p1[1]![3]! - p2[1]![3]!;
    const dz = p1[2]![3]! - p2[2]![3]!;
    const translationM = Math.hypot(dx, dy, dz);
    const R1 = [p1[0]!.slice(0, 3), p1[1]!.slice(0, 3), p1[2]!.slice(0, 3)];
    const R2 = [p2[0]!.slice(0, 3), p2[1]!.slice(0, 3), p2[2]!.slice(0, 3)];
    const angleRad = deltaAngleBetweenRot(R1, R2);
    return {
        translationM,
        angleRad,
        magicMm: translationM * 1000 + angleRad * 180 / Math.PI,
    };
}


// ─── Scaled-duration goto ────────────────────────────────────────────
//
// Pick a goto duration that feels right no matter which dimension is
// furthest off. Compute a candidate per channel (head magic-mm, each
// antenna deg, body yaw deg), then take the max — the slowest-converging
// channel limits the move. Each candidate is clamped to [min, max].

export type ScaledDurationChannel = 'head' | 'antR' | 'antL' | 'body';

export interface ScaledDurationResult {
    /** Final clamped duration in seconds. */
    duration: number;
    /** Which channel was the slowest (and therefore the limiter), or
     *  `null` if no channels were provided. */
    winner: ScaledDurationChannel | null;
    /** Per-channel raw candidate durations (pre-clamp). Useful for
     *  logging "head=0.84s antR=0.12s antL=0.10s body=0.00s". */
    candidates: Array<{ ch: ScaledDurationChannel; d: number }>;
}

export interface ScaledDurationCurrent {
    head?: number[][] | null;
    /** [rightRad, leftRad]. Wider `number[]` type matches the SDK's
     *  `robotState.antennas`; only indices 0 and 1 are read. */
    antennas?: number[] | null;
    bodyYaw?: number | null;
}

export interface ScaledDurationTarget {
    head?: number[][] | null;
    antennas?: number[] | null;
    bodyYaw?: number | null;
}

export function scaledDuration(opts: {
    current: ScaledDurationCurrent;
    target: ScaledDurationTarget;
    constants?: Partial<typeof SCALED_DURATION_PRESET>;
    clamp?: Partial<typeof SCALED_DURATION_CLAMP>;
}): ScaledDurationResult {
    const k = { ...SCALED_DURATION_PRESET, ...opts.constants };
    const c = { ...SCALED_DURATION_CLAMP, ...opts.clamp };

    const candidates: Array<{ ch: ScaledDurationChannel; d: number }> = [];

    if (opts.current.head && opts.target.head) {
        const { magicMm } = distanceBetweenPoses(opts.current.head, opts.target.head);
        candidates.push({ ch: 'head', d: magicMm * k.head });
    }
    if (opts.current.antennas && opts.target.antennas
        && opts.current.antennas.length >= 2 && opts.target.antennas.length >= 2) {
        const dR = Math.abs(opts.target.antennas[0]! - opts.current.antennas[0]!) * 180 / Math.PI;
        const dL = Math.abs(opts.target.antennas[1]! - opts.current.antennas[1]!) * 180 / Math.PI;
        candidates.push({ ch: 'antR', d: dR * k.antenna });
        candidates.push({ ch: 'antL', d: dL * k.antenna });
    }
    if (opts.current.bodyYaw != null && opts.target.bodyYaw != null) {
        const dB = Math.abs(opts.target.bodyYaw - opts.current.bodyYaw) * 180 / Math.PI;
        candidates.push({ ch: 'body', d: dB * k.bodyYaw });
    }

    if (candidates.length === 0) {
        // No channels to compare; fall back to the upper clamp as a
        // safe-but-slow default rather than producing 0.
        return { duration: c.max, winner: null, candidates: [] };
    }

    let winner = candidates[0]!;
    for (const cand of candidates) if (cand.d > winner.d) winner = cand;
    const clamped = Math.min(Math.max(winner.d, c.min), c.max);
    return { duration: clamped, winner: winner.ch, candidates };
}


// ─── Pose fusion (primary + offset) ──────────────────────────────────
//
// Layering motions: a primary trajectory (a dance, a recorded move)
// plus an offset on top (a head wobble, a face-tracking correction).
// Naive Euler-angle sums are wrong as soon as the primary has any
// meaningful rotation; the right thing is to compose in the world
// frame. Port of `compose_world_offset` from the Python SDK
// (`reachy_mini/utils/interpolation.py:187`).
//
// Only meaningful for primary + offset. Don't use these to "average two
// trajectories" — the operation isn't commutative and the result of
// averaging arbitrary motions has no useful interpretation.

/** Compose a world-frame offset onto an absolute 4×4 pose:
 *
 *   t_final = t_abs + t_off    (translations add in world)
 *   R_final = R_off @ R_abs    (rotations compose in world)
 *
 * Note the rotation order: the offset rotates the frame in place about
 * its own origin by a rotation defined in world axes; it does NOT
 * compose in the body frame.
 */
export function composeWorldOffset(
    T_abs: number[][],
    T_off_world: number[][],
): number[][] {
    const R_abs = [
        [T_abs[0]![0]!, T_abs[0]![1]!, T_abs[0]![2]!],
        [T_abs[1]![0]!, T_abs[1]![1]!, T_abs[1]![2]!],
        [T_abs[2]![0]!, T_abs[2]![1]!, T_abs[2]![2]!],
    ];
    const R_off = [
        [T_off_world[0]![0]!, T_off_world[0]![1]!, T_off_world[0]![2]!],
        [T_off_world[1]![0]!, T_off_world[1]![1]!, T_off_world[1]![2]!],
        [T_off_world[2]![0]!, T_off_world[2]![1]!, T_off_world[2]![2]!],
    ];
    // R_final = R_off @ R_abs
    const R_final: number[][] = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
            R_final[i]![j] = R_off[i]![0]! * R_abs[0]![j]!
                           + R_off[i]![1]! * R_abs[1]![j]!
                           + R_off[i]![2]! * R_abs[2]![j]!;
        }
    }
    return [
        [R_final[0]![0]!, R_final[0]![1]!, R_final[0]![2]!, T_abs[0]![3]! + T_off_world[0]![3]!],
        [R_final[1]![0]!, R_final[1]![1]!, R_final[1]![2]!, T_abs[1]![3]! + T_off_world[1]![3]!],
        [R_final[2]![0]!, R_final[2]![1]!, R_final[2]![2]!, T_abs[2]![3]! + T_off_world[2]![3]!],
        [0, 0, 0, 1],
    ];
}

export interface FullBodyPose {
    /** 4×4 nested head pose. */
    head: number[][];
    /** [rightRad, leftRad]. */
    antennas: [number, number];
    /** Radians. */
    bodyYaw: number;
}

/** Combine a primary full-body pose with an offset. Head composes via
 *  `composeWorldOffset`; antennas and body yaw add component-wise (they
 *  are 1-DOF scalars, no axis ambiguity). When stacking multiple
 *  offsets, call this once per offset rather than summing offsets first
 *  and composing once — the order matters for the head channel. */
export function combineFullBody(
    primary: FullBodyPose,
    offset: FullBodyPose,
): FullBodyPose {
    return {
        head: composeWorldOffset(primary.head, offset.head),
        antennas: [
            primary.antennas[0] + offset.antennas[0],
            primary.antennas[1] + offset.antennas[1],
        ],
        bodyYaw: primary.bodyYaw + offset.bodyYaw,
    };
}


// ─── Lead-compensated playback sampler ───────────────────────────────
//
// Single helper to apply the lead offsets above. The caller provides a
// `sample(t)` that returns the recording's frame at recording-time t;
// the wrapper returns a new sampler that internally samples twice (at
// `t + head lead` and `t + antenna lead`) and recombines: head + body
// from the head sample, antennas from the antenna sample. Audio stays
// at plain `t` — it is the reference.

export interface RecordedFrame {
    /** 4×4 head pose (nested). */
    head?: number[][] | null;
    /** [rightRad, leftRad]. */
    antennas?: number[] | null;
    /** Radians. */
    bodyYaw?: number | null;
}

/** Wrap a recorded-motion sampler with per-channel lead compensation.
 *  Returns a new sampler that, at wall-clock t, produces a frame whose
 *  head + bodyYaw come from `sample(t + head lead)` and whose antennas
 *  come from `sample(t + antenna lead)`. */
export function withLeadCompensation(
    sample: (t: number) => RecordedFrame | null,
    leads: Partial<typeof LEAD_COMPENSATION_S> = {},
): (t: number) => RecordedFrame | null {
    const headLead = leads.head ?? LEAD_COMPENSATION_S.head;
    const antLead = leads.antenna ?? LEAD_COMPENSATION_S.antenna;
    const bodyLead = leads.bodyYaw ?? LEAD_COMPENSATION_S.bodyYaw;
    return (t: number) => {
        const fHead = sample(t + headLead);
        const fAnt = sample(t + antLead);
        if (fHead == null && fAnt == null) return null;
        // Body lead usually equals head lead, so reuse fHead in that
        // common case; only resample when it differs.
        const fBody = bodyLead === headLead ? fHead : sample(t + bodyLead);
        return {
            head: fHead?.head ?? null,
            antennas: fAnt?.antennas ?? null,
            bodyYaw: fBody?.bodyYaw ?? null,
        };
    };
}


// ─── Shutdown handler ────────────────────────────────────────────────
//
// Closing the tab gives us only milliseconds of synchronous JS before
// the runtime is torn down. Two data-channel sends, fire-and-forget;
// the daemon executes them in order on its own clock long after the
// page is gone:
//
//   1. setMotorMode("enabled") — daemon pins targets to present pose
//      (PR #1138). Safe by construction.
//   2. gotoTarget(targetPose, scaledDuration) — daemon smoothly
//      interpolates home from wherever the head is.
//
// Returns a disposer that unregisters the listeners.

export interface ShutdownHandlerOptions {
    /** Where to send the head on shutdown. Defaults to the rest pose
     *  (`INIT_HEAD_POSE` + `INIT_ANTENNAS_RAD` + 0 yaw). */
    targetPose?: FullBodyPose;
    /** Per-channel constants for the scaled-duration goto. */
    constants?: Partial<typeof SCALED_DURATION_PRESET>;
    /** Min/max clamp for the scaled-duration goto. */
    clamp?: Partial<typeof SCALED_DURATION_CLAMP>;
}

const REST_POSE: FullBodyPose = {
    head: INIT_HEAD_POSE,
    antennas: INIT_ANTENNAS_RAD,
    bodyYaw: INIT_BODY_YAW_RAD,
};

export function installShutdownHandler(
    robot: ReachyMiniInstance,
    opts: ShutdownHandlerOptions = {},
): () => void {
    const target = opts.targetPose ?? REST_POSE;
    let fired = false;

    const handler = () => {
        if (fired) return;
        fired = true;
        if (robot.state !== 'streaming') return;
        try {
            robot.setMotorMode('enabled');
            const s = robot.robotState;
            const current: ScaledDurationCurrent = {
                head: s.head ? nest4x4(s.head) : null,
                antennas: s.antennas ?? null,
                bodyYaw: s.body_yaw ?? null,
            };
            const { duration } = scaledDuration({
                current,
                target: { head: target.head, antennas: target.antennas, bodyYaw: target.bodyYaw },
                constants: opts.constants,
                clamp: opts.clamp,
            });
            robot.gotoTarget({
                head: flat16(target.head),
                antennas: target.antennas,
                body_yaw: target.bodyYaw,
                duration,
            });
        } catch {
            // Best-effort. The page is closing; nothing to surface.
        }
    };

    window.addEventListener('pagehide', handler);
    window.addEventListener('beforeunload', handler);
    return () => {
        window.removeEventListener('pagehide', handler);
        window.removeEventListener('beforeunload', handler);
    };
}


// ─── Convenience: safe return to a pose ──────────────────────────────
//
// One-liner that combines a safe enable (no-op on post-#1138 daemons)
// with a distance-scaled goto. The most common "get the head somewhere
// known" call after puppeteering, after a long-running move, or as
// any app's reset button. Returns the duration the goto was issued
// with.

export async function safeReturnToPose(
    robot: ReachyMiniInstance,
    target: FullBodyPose = REST_POSE,
    opts: {
        constants?: Partial<typeof SCALED_DURATION_PRESET>;
        clamp?: Partial<typeof SCALED_DURATION_CLAMP>;
    } = {},
): Promise<number> {
    robot.setMotorMode('enabled');
    const s = robot.robotState;
    const { duration } = scaledDuration({
        current: {
            head: s.head ? nest4x4(s.head) : null,
            antennas: s.antennas ?? null,
            bodyYaw: s.body_yaw ?? null,
        },
        target: {
            head: target.head,
            antennas: target.antennas,
            bodyYaw: target.bodyYaw,
        },
        constants: opts.constants,
        clamp: opts.clamp,
    });
    robot.gotoTarget({
        head: flat16(target.head),
        antennas: target.antennas,
        body_yaw: target.bodyYaw,
        duration,
    });
    return duration;
}
