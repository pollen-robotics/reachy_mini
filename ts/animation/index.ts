/**
 * `@pollen-robotics/reachy-mini-sdk/animation` — motion utilities.
 *
 * Phase 1 surface, split by concern:
 *
 *   - **Types** (`Pose`, `PartialPose`): the wire format every SDK
 *     motion call already uses.
 *   - **Canonical pose** (`INIT_POSE` + sub-constants): the "safe
 *     rest" pose, mirror of the daemon's `Backend.INIT_HEAD_POSE` +
 *     `Backend.INIT_ANTENNAS_JOINT_POSITIONS`.
 *   - **Distance math** (`distanceBetweenPoses`, `scaledDuration`):
 *     "how big is this move" + "how long should it take", computed
 *     synchronously client-side so apps can sync audio / scheduling
 *     against it.
 *   - **Exit handlers** (`safelyReturnToPose`, `installShutdownHandler`):
 *     thin convenience wrappers around the three-call SDK sequence
 *     (`setMotorMode` + read state + `gotoTarget`) for `pagehide` /
 *     `onLeave` call sites.
 *
 * Pure-TS: this lib does not introduce any daemon-side primitive. All
 * motion routes through existing data-channel commands. See
 * `DESIGN.md` for the rationale.
 *
 * @see DESIGN.md
 * @see APP_CREATION_GUIDE.md — "Robotics best practices" section
 */

// Types
export type { Pose, PartialPose } from "./pose.js";
export type { PoseDistance, ScaledDurationResult } from "./distance.js";
export type { ScaledDurationPreset } from "./presets.js";
export type {
    SafelyReturnOptions,
    InstallShutdownHandlerOptions,
} from "./safe-return.js";

// Constants
export {
    INIT_HEAD_POSE_FLAT,
    INIT_ANTENNAS_RAD,
    INIT_BODY_YAW_RAD,
    INIT_POSE,
    DEFAULT_SCALED_DURATION_PRESET,
} from "./presets.js";

// Distance + scaled duration
export { distanceBetweenPoses, scaledDuration } from "./distance.js";

// Exit-handler helpers
export { safelyReturnToPose, installShutdownHandler } from "./safe-return.js";
