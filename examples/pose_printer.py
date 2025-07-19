"""Stream Reachy Mini head pose.

Continuously prints the head’s Cartesian position (in millimetres) and
Euler orientation (roll‑pitch‑yaw in degrees). Stop with **Ctrl‑C**.
The motors are disabled so the user can move the head freely.
"""

from __future__ import annotations

import time
from typing import NoReturn

import numpy as np
from reachy_mini import ReachyMini
from scipy.spatial.transform import Rotation as R


def print_pose(pose: np.ndarray) -> None:
    """Pretty‑print a 4 × 4 homogeneous pose matrix.

    Parameters
    ----------
    pose
        Homogeneous transformation matrix whose top‑left 3 × 3 block is a
        rotation matrix (R) and whose last column is the translation
        vector (t) expressed in metres.

    Side effect
    -----------
    Writes one line to *stdout*:

    ```
    x=….. mm, y=….. mm, z=….. mm | roll=…..°, pitch=…..°, yaw=…..°
    ```
    """
    x_mm, y_mm, z_mm = pose[:3, 3] * 1_000  # metres → millimetres
    roll_deg, pitch_deg, yaw_deg = R.from_matrix(pose[:3, :3]).as_euler(
        "xyz",
        degrees=True,
    )

    print(
        f"x={x_mm:7.2f} mm, y={y_mm:7.2f} mm, z={z_mm:7.2f} mm | "
        f"roll={roll_deg:7.2f}°, pitch={pitch_deg:7.2f}°, yaw={yaw_deg:7.2f}°",
    )


def main() -> NoReturn:
    """Open Reachy Mini, disable motors, and stream pose."""
    with ReachyMini() as reachy:
        reachy.disable_motors()
        print("Streaming pose… (press Ctrl‑C to quit)")
        try:
            while True:
                head_joints, _ = reachy._get_current_joint_positions()
                pose = reachy.head_kinematics.fk(head_joints)
                print_pose(pose)
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
