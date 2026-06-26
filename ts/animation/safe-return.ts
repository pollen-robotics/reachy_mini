/**
 * Exit-handler helpers.
 *
 * Two thin convenience functions on top of the SDK's existing motion
 * primitives (`setMotorMode`, `gotoTarget`) + the lib's pose math
 * (`scaledDuration`). They exist so that:
 *
 *   1. App authors get a one-liner for the "return to safe pose on
 *      exit" pattern instead of copy-pasting the three-call sequence.
 *   2. The pattern is consistent across apps (same target, same
 *      scaled-duration tuning, same error handling for closed
 *      channels).
 *
 * No daemon-side primitive is introduced; everything routes through
 * existing data-channel commands.
 */

import type { PartialPose } from "./pose.js";
import {
    DEFAULT_SCALED_DURATION_PRESET,
    INIT_POSE,
    type ScaledDurationPreset,
} from "./presets.js";
import { scaledDuration, type ScaledDurationResult } from "./distance.js";
import type { ReachyMiniInstance } from "../lib/types.js";

/** Options for `safelyReturnToPose`. */
export interface SafelyReturnOptions {
    /** Where to return to. Defaults to `INIT_POSE`. */
    target?: PartialPose;
    /** Scaled-duration tuning. Defaults to `DEFAULT_SCALED_DURATION_PRESET`. */
    preset?: ScaledDurationPreset;
}

/**
 * Smooth, distance-scaled return to a target pose. Routes through the
 * existing SDK primitives:
 *
 *   1. `reachy.setMotorMode("enabled")` — safe by construction since
 *      daemon PR #1138 (the daemon pins all targets to the present
 *      pose before flipping torque on).
 *   2. Read `reachy.robotState` for the current pose snapshot.
 *   3. Compute `scaledDuration(current, target, preset)`.
 *   4. Call `reachy.gotoTarget({ ...target, duration })`.
 *
 * Returns **synchronously after dispatching** the goto — does NOT
 * await completion. Callers who want to await the move should
 * subscribe to the `state` event or
 * `await sleep(result.duration * 1000)`.
 *
 * Safe to call when the data channel is closed: every SDK call
 * swallows the "channel closed" error silently, and the planned
 * duration is still returned for log lines.
 *
 * The returned `ScaledDurationResult` carries the diagnostic data
 * (which channel was the slowest, per-channel candidate durations)
 * so apps can log "why does the home return take 1.2 s?".
 *
 * @example
 * // Host-shell embedded app:
 * onLeave(() => safelyReturnToPose(reachy));
 *
 * // Custom target:
 * safelyReturnToPose(reachy, {
 *   target: { head: customHeadMatrix, antennas: [0, 0], body_yaw: 0 },
 * });
 */
export function safelyReturnToPose(
    reachy: ReachyMiniInstance,
    options: SafelyReturnOptions = {},
): ScaledDurationResult {
    const target = options.target ?? INIT_POSE;
    const preset = options.preset ?? DEFAULT_SCALED_DURATION_PRESET;

    try {
        reachy.setMotorMode("enabled");
    } catch {
        // Already in the right mode, or data channel closed; both are
        // benign here — we still compute and emit the goto.
    }

    const current = reachy.robotState as PartialPose;
    const plan = scaledDuration(current, target, preset);

    const args: Parameters<ReachyMiniInstance["gotoTarget"]>[0] = {
        duration: plan.duration,
    };
    if (target.head !== undefined) args.head = target.head.slice();
    if (target.antennas !== undefined) {
        args.antennas = [target.antennas[0] ?? 0, target.antennas[1] ?? 0];
    }
    if (target.body_yaw !== undefined) args.body_yaw = target.body_yaw;

    try {
        reachy.gotoTarget(args);
    } catch {
        // Closed channel / shutting down / session torn down between
        // the read and the send. The plan is still useful for logs.
    }

    return plan;
}

/** Options for `installShutdownHandler`. */
export interface InstallShutdownHandlerOptions extends SafelyReturnOptions {
    /**
     * Skip when `reachy.state !== "streaming"`. Defaults to `true` —
     * the handler only fires when a session is actually live, so a
     * stale tab where the user never picked a robot doesn't try to
     * command anything on close.
     */
    onlyWhenStreaming?: boolean;
}

/**
 * Wire `pagehide` + `beforeunload` so the robot returns to its safe-rest
 * pose when the app closes.
 *
 * Use case: **standalone apps** (test benches, custom dashboards)
 * that don't have a host shell to manage their lifecycle. Host-shell
 * embedded apps should use the `onLeave` callback from
 * `connectToHost()` instead — it integrates with the host's
 * leave-protocol budget and avoids double-firing.
 *
 * @example
 * const reachy = new ReachyMini(...);
 * await reachy.authenticate();
 * await reachy.connect();
 * // ...pick robot, startSession...
 * installShutdownHandler(reachy);
 */
export function installShutdownHandler(
    reachy: ReachyMiniInstance,
    options: InstallShutdownHandlerOptions = {},
): void {
    const onlyWhenStreaming = options.onlyWhenStreaming ?? true;
    // Reentry guard: `pagehide` and `beforeunload` both fire on a real
    // tab close, so without this flag `safelyReturnToPose` would dispatch
    // twice (idempotent in practice — the second goto overrides the first
    // with the same duration — but wasteful and noisy in the daemon log).
    // Matches the `shuttingDown` boolean in Rémi's `reachy-mini-js-practices`
    // bench, which is the reference behaviour for this helper.
    let shuttingDown = false;
    const fire = (): void => {
        if (shuttingDown) return;
        shuttingDown = true;
        if (onlyWhenStreaming && reachy.state !== "streaming") return;
        safelyReturnToPose(reachy, {
            target: options.target,
            preset: options.preset,
        });
    };
    window.addEventListener("pagehide", fire);
    window.addEventListener("beforeunload", fire);
}
