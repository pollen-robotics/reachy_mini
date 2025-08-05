"""Reachy Mini Analytical Kinematics GUI Example."""

import time
import tkinter as tk

import numpy as np
from placo_utils.tf import tf

from reachy_mini import ReachyMini
from reachy_mini.analytic_kinematics import ReachyMiniAnalyticKinematics


def main():
    """Run a GUI to set the head position and orientation of Reachy Mini."""
    with ReachyMini() as mini:
        analytic_solver = ReachyMiniAnalyticKinematics(urdf_path="src/reachy_mini/descriptions/reachy_mini/urdf/robot.urdf")

        # initial joint angles
        joints = analytic_solver.ik(np.eye(4))

        # gui definition
        root = tk.Tk()
        root.title("Target Position and Orientation")
        pos_vars = [tk.DoubleVar(value=0.0) for _ in range(3)]
        rpy_vars = [tk.DoubleVar(value=0.0) for _ in range(4)]
        labels = ["X (m)", "Y (m)", "Z (m)", "Roll (rad)", "Pitch (rad)", "Yaw (rad)", "Body Yaw (rad)"]
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
            T_home_head_target = (
                tf.translation_matrix((px, py, pz))
                @ tf.euler_matrix(roll, pitch, yaw)
            )

            T_home_head_current = mini.get_current_head_pose()
            jacobian = analytic_solver.jacobian(T_home_head_current)

            dM = np.linalg.inv(T_home_head_current) @ T_home_head_target
            error_p = dM[:3, 3]
            error_rpy = np.array(tf.euler_from_matrix(dM[:3, :3], axes="sxyz"))
            dq = jacobian @ np.concatenate((error_p, error_rpy))
            

            joints_dq = dq * 0.02
            # Update joint positions with the computed increments
            joints = [(j + j_dq) for j, j_dq in zip(joints, [0.0]+list(joints_dq))]
            joints[0] = body_yaw  # Set the body yaw joint position

            if len(joints) != 7 or not np.all(np.isfinite(joints)):
                print(joints)
                print("joints IK failed for some motors, skipping iteration")
                continue

            # jacobian = analytic_solver.jacobian(robot.get_T_world_frame("head"))
            # print("singular values position:", 1/np.linalg.svd(jacobian[:,:3], compute_uv=False))
            # print("singular values orientation:", 1/np.linalg.svd(jacobian[:,3:], compute_uv=False))

            try:
                mini._set_joint_positions(
                    head_joint_positions=joints,
                    antennas_joint_positions=None
                )
            except ValueError as e:
                print(f"Error: {e}")
                continue

            time.sleep(0.02)


if __name__ == "__main__":
    main()
