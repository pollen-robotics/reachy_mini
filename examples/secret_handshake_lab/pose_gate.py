"""Sleep-pose gate for the secret handshake. Pure, dependency-free.

Answers one question: is the head currently resting (roughly) in the sleep
pose? Checked once at ARMING time only; the real security comes from
torque-off + the two-round collision rhythm, so tolerances are generous
(see design spec section 6, derived from the recorded data).

Accepts any 4x4 row-major pose (numpy array or nested lists), robot frame:
translation in meters in the last column, rotation in the top-left 3x3.
"""

from __future__ import annotations

import math

# Copied from ReachyMiniBackend.SLEEP_HEAD_POSE in
# src/reachy_mini/daemon/backend/abstract.py so this lab stays import-free.
# If the daemon constant ever changes, update this copy (regression tests
# will still pass; only the gate's reference point moves).
SLEEP_HEAD_POSE = (
    (0.911, 0.004, 0.413, -0.021),
    (-0.004, 1.0, -0.001, 0.001),
    (-0.413, -0.001, 0.911, -0.044),
    (0.0, 0.0, 0.0, 1.0),
)

# Generous tolerances (spec section 6): x +-15 mm, y +-12 mm, z +-12 mm,
# roll +-8 deg, pitch +-15 deg, yaw ignored (head hangs loose around yaw).
TOL_X_M = 0.015
TOL_Y_M = 0.012
TOL_Z_M = 0.012
TOL_ROLL_RAD = math.radians(8.0)
TOL_PITCH_RAD = math.radians(15.0)


def _pitch_roll(pose) -> tuple[float, float]:
    """Pitch and roll from a 4x4 pose, ZYX (yaw-pitch-roll) convention."""
    s = -pose[2][0]
    s = max(-1.0, min(1.0, s))
    pitch = math.asin(s)
    roll = math.atan2(pose[2][1], pose[2][2])
    return pitch, roll


_REF_PITCH, _REF_ROLL = _pitch_roll(SLEEP_HEAD_POSE)


def sleep_pose_deviation(pose) -> dict:
    """Deviations from the sleep pose, for live display / debugging."""
    pitch, roll = _pitch_roll(pose)
    return {
        "dx_mm": (pose[0][3] - SLEEP_HEAD_POSE[0][3]) * 1000.0,
        "dy_mm": (pose[1][3] - SLEEP_HEAD_POSE[1][3]) * 1000.0,
        "dz_mm": (pose[2][3] - SLEEP_HEAD_POSE[2][3]) * 1000.0,
        "dpitch_deg": math.degrees(pitch - _REF_PITCH),
        "droll_deg": math.degrees(roll - _REF_ROLL),
    }


def head_in_sleep_pose(pose) -> bool:
    """True when `pose` (4x4) is within the generous sleep-pose tolerances."""
    if abs(pose[0][3] - SLEEP_HEAD_POSE[0][3]) > TOL_X_M:
        return False
    if abs(pose[1][3] - SLEEP_HEAD_POSE[1][3]) > TOL_Y_M:
        return False
    if abs(pose[2][3] - SLEEP_HEAD_POSE[2][3]) > TOL_Z_M:
        return False
    pitch, roll = _pitch_roll(pose)
    if abs(pitch - _REF_PITCH) > TOL_PITCH_RAD:
        return False
    if abs(roll - _REF_ROLL) > TOL_ROLL_RAD:
        return False
    return True
