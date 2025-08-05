"""Reachy Mini Analytical Kinematics GUI Example."""

import os
import time
import tkinter as tk

from placo_utils.tf import tf
from placo_utils.visualization import frame_viz, robot_frame_viz, robot_viz

from reachy_mini.placo_kinematics import PlacoKinematics

urdf_path = os.path.abspath("src/reachy_mini/descriptions/reachy_mini/urdf/robot.urdf")
solver = PlacoKinematics(urdf_path, 0.02)
robot = solver.robot
robot.update_kinematics()


viz = robot_viz(robot)


def main():
    # initial joint angles
    joints = solver.ik(solver.head_starting_pose)

    # gui definition
    root = tk.Tk()
    root.title("Target Position and Orientation")
    pos_vars = [tk.DoubleVar(value=0.0) for _ in range(3)]
    rpy_vars = [tk.DoubleVar(value=0.0) for _ in range(3)]
    labels = ["X (m)", "Y (m)", "Z (m)", "Roll (rad)", "Pitch (rad)", "Yaw (rad)"]
    for i, label in enumerate(labels):
        tk.Label(root, text=label).grid(row=i, column=0)
        var = pos_vars[i] if i < 3 else rpy_vars[i - 3]
        tk.Scale(
            root,
            variable=var,
            from_=-0.05 if i < 3 else -3.1,
            to=0.05 if i < 3 else 3.1,
            resolution=0.001,
            orient="horizontal",
            length=200,
        ).grid(row=i, column=1)

    # Body yaw slider
    body_yaw_var = tk.DoubleVar(value=0.0)
    tk.Label(root, text="Body Yaw (rad)").grid(row=6, column=0)
    tk.Scale(
        root,
        variable=body_yaw_var,
        from_=-3.1,
        to=3.1,
        resolution=0.001,
        orient="horizontal",
        length=200,
    ).grid(row=6, column=1)

    # Collision checkbox
    collision_var = tk.BooleanVar(value=False)
    tk.Checkbutton(
        root,
        text="Enable Collisions",
        variable=collision_var,
        onvalue=True,
        offvalue=False,
    ).grid(row=7, column=0, columnspan=2)

    while True:
        root.update()

        # Read GUI values
        px, py, pz = [v.get() for v in pos_vars]
        roll, pitch, yaw = [v.get() for v in rpy_vars]
        # Compute the target transformation matrix in the world frame
        T_head_target = tf.translation_matrix((px, py, pz)) @ tf.euler_matrix(
            roll, pitch, yaw
        )

        joints = solver.ik(pose=T_head_target, body_yaw=body_yaw_var.get())

        if hasattr(solver, "robot_ik"):
            try:
                solver.fk(joints, collision_var.get())
            except Exception as e:
                print(f"FK failed: {e}")

        if len(joints) != 7:
            print("joints IK failed for some motors, skipping iteration")
            continue

        viz.display(robot.state.q)
        robot_frame_viz(robot, "head")
        # robot_frame_viz(robot, "dummy_torso_yaw")
        frame_viz(
            "target",
            (
                solver.head_starting_pose
                @ tf.translation_matrix((px, py, pz))
                @ tf.euler_matrix(roll, pitch, yaw)
            ),
        )

        time.sleep(0.02)


if __name__ == "__main__":
    main()
