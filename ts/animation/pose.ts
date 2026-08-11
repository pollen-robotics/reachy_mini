/**
 * Pose primitives.
 *
 * A `Pose` is the canonical "frame" the SDK sends and receives on the
 * data channel. The shape mirrors `setTarget` / `gotoTarget` / the
 * `state` event payload, named so it can be passed around without
 * anonymous-object soup.
 *
 * Channel units (wire format):
 *   - `head`     : flat 16-float row-major 4×4 homogeneous matrix.
 *   - `antennas` : [right, left] in **radians**.
 *   - `body_yaw` : scalar in **radians**.
 */

/**
 * Canonical pose carried over the data channel. All three channels
 * required.
 */
export interface Pose {
    /** Flat 16-float row-major 4×4 head pose. */
    head: number[];
    /** [right, left] in radians. */
    antennas: [number, number];
    /** Body yaw in radians. */
    body_yaw: number;
}

/**
 * Same shape as `Pose` but every channel is optional. Matches the
 * partial-update semantics of `setTarget` / `gotoTarget`: only the
 * channels you specify are commanded; the rest keep their previous
 * target.
 *
 * Also the shape `reachy.robotState` actually carries — fields appear
 * only once the daemon has emitted them.
 */
export interface PartialPose {
    head?: number[];
    antennas?: [number, number] | number[];
    body_yaw?: number;
}
