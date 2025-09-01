"""Reachy Mini Analytical Kinematics GUI Example."""

import time
import tkinter as tk

import numpy as np
from placo_utils.tf import tf

from reachy_mini import ReachyMini
from reachy_mini.kinematics import ReachyMiniAnalyticKinematics


def main():
    """Run a GUI to set the head position and orientation of Reachy Mini."""
    with ReachyMini() as mini:
        analytic_solver = ReachyMiniAnalyticKinematics(
            urdf_path="src/reachy_mini/descriptions/reachy_mini/urdf/robot.urdf"
        )

        # initial joint angles
        joints = analytic_solver.ik(np.eye(4))

        # gui definition
        root = tk.Tk()
        root.title("Target Position and Orientation")
        pos_vars = [tk.DoubleVar(value=0.0) for _ in range(3)]
        rpy_vars = [tk.DoubleVar(value=0.0) for _ in range(4)]
        labels = [
            "X (m)",
            "Y (m)",
            "Z (m)",
            "Roll (rad)",
            "Pitch (rad)",
            "Yaw (rad)",
            "Body Yaw (rad)",
        ]
        for i, label in enumerate(labels):
            tk.Label(root, text=label).grid(row=i, column=0)
            var = pos_vars[i] if i < 3 else rpy_vars[i - 3]
            tk.Scale(
                root,
                variable=var,
                from_=-0.05 if i < 3 else -1.57,
                to=0.05 if i < 3 else 1.57,
                resolution=0.001,
                orient="horizontal",
                length=200,
            ).grid(row=i, column=1)

        while True:
            root.update()

            # Read GUI values
            px, py, pz = [v.get() for v in pos_vars]
            roll, pitch, yaw, body_yaw = [v.get() for v in rpy_vars]
            # Compute the target transformation matrix in the world frame
            target_pose = tf.translation_matrix((px, py, pz)) @ tf.euler_matrix(
                roll, pitch, yaw
            )

            joints = analytic_solver.ik(pose=target_pose, body_yaw=body_yaw)

            if len(joints) != 7 or not np.all(np.isfinite(joints)):
                print("joints IK failed for some motors, skipping iteration")
                continue

            # jacobian = analytic_solver.jacobian(robot.get_T_world_frame("head"))
            # print("singular values position:", 1/np.linalg.svd(jacobian[:,:3], compute_uv=False))
            # print("singular values orientation:", 1/np.linalg.svd(jacobian[:,3:], compute_uv=False))

            try:
                mini._set_joint_positions(
                    head_joint_positions=joints, antennas_joint_positions=None
                )
            except ValueError as e:
                print(f"Error: {e}")
                continue

            time.sleep(0.02)


if __name__ == "__main__":
    main()
