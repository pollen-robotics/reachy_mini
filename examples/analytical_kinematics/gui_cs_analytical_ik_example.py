import time
import tkinter as tk

from placo_utils.tf import tf

from reachy_mini import ReachyMini
from reachy_mini.analytic_kinematics import ReachyMiniAnalyticKinematics


def main():
    with ReachyMini() as mini:
        analytic_solver = ReachyMiniAnalyticKinematics(robot=mini.head_kinematics.robot)

        # initial joint angles
        joints = analytic_solver.ik(analytic_solver.T_world_head_home)

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
            roll, pitch, yaw = [v.get() for v in rpy_vars]
            # Compute the target transformation matrix in the world frame
            T_world_target = (
                analytic_solver.T_world_head_home
                @ tf.translation_matrix((px, py, pz))
                @ tf.euler_matrix(roll, pitch, yaw)
            )
            joints = analytic_solver.ik(T_world_target)

            if len(joints) != 6:
                print("joints IK failed for some motors, skipping iteration")
                continue

            # jacobian = analytic_solver.jacobian(robot.get_T_world_frame("head"))
            # print("singular values position:", 1/np.linalg.svd(jacobian[:,:3], compute_uv=False))
            # print("singular values orientation:", 1/np.linalg.svd(jacobian[:,3:], compute_uv=False))

            try:
                mini._send_joint_command(
                    head_joint_positions=[
                        0.0,
                        joints["1"],
                        joints["2"],
                        joints["3"],
                        joints["4"],
                        joints["5"],
                        joints["6"],
                    ],
                    antennas_joint_positions=None,
                    check_collision=False,
                )
            except ValueError as e:
                print(f"Error: {e}")
                continue
            time.sleep(0.02)


if __name__ == "__main__":
    main()
