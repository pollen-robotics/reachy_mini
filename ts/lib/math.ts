/**
 * Math utilities for the ReachyMini SDK.
 *
 * These are the canonical conversions between human-readable
 * roll/pitch/yaw degrees and the wire format the daemon expects
 * (4×4 rotation matrices, ZYX convention).
 */

export function degToRad(deg: number): number {
    return deg * Math.PI / 180;
}

export function radToDeg(rad: number): number {
    return rad * 180 / Math.PI;
}

/**
 * Roll/pitch/yaw (degrees) → 4×4 rotation matrix (ZYX convention).
 * This is the wire format for the robot's `set_target` command.
 */
export function rpyToMatrix(
    rollDeg: number,
    pitchDeg: number,
    yawDeg: number,
): number[][] {
    const r = degToRad(rollDeg), p = degToRad(pitchDeg), y = degToRad(yawDeg);
    const cy = Math.cos(y), sy = Math.sin(y);
    const cp = Math.cos(p), sp = Math.sin(p);
    const cr = Math.cos(r), sr = Math.sin(r);
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, 0],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, 0],
        [-sp, cp * sr, cp * cr, 0],
        [0, 0, 0, 1],
    ];
}

/**
 * Rotation matrix (3×3 or 4×4) → { roll, pitch, yaw } in degrees.
 */
export function matrixToRpy(
    m: number[][],
): { roll: number; pitch: number; yaw: number } {
    return {
        roll: radToDeg(Math.atan2(m[2]![1]!, m[2]![2]!)),
        pitch: radToDeg(Math.asin(-m[2]![0]!)),
        yaw: radToDeg(Math.atan2(m[1]![0]!, m[0]![0]!)),
    };
}
